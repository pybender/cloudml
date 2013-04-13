import json
import logging
from copy import copy
from itertools import izip

from api import celery, db
from bson.objectid import ObjectId

from core.trainer.trainer import Trainer
from core.trainer.config import FeatureModel
from api.models import Test, Model
from helpers.weights import get_weighted_data


class InvalidOperationError(Exception):
    pass


@celery.task
def train_model(model_name, parameters):
    """
    Train new model
    """
    try:
        model = db.cloudml.Model.find_one({'name': model_name})
        if model.status == model.STATUS_TRAINED:
            raise InvalidOperationError("Model already trained")
        model.status = model.STATUS_TRAINING
        model.error = ""
        model.save()
        feature_model = FeatureModel(json.dumps(model.features),
                                     is_file=False)
        trainer = Trainer(feature_model)
        train_handler = model.get_import_handler(parameters)
        trainer.train(train_handler)
        trainer.clear_temp_data()
        model.set_trainer(trainer)
        model.set_weights(**trainer.get_weights())
        model.status = model.STATUS_TRAINED
        model.save()
    except Exception, exc:
        logging.error(exc)
        model.status = model.STATUS_ERROR
        model.error = str(exc)
        model.save()
        raise

    return "Model trained at %s" % trainer.train_time


@celery.task
def run_test(test_id):
    """
    Running tests for trained model
    """
    test = db.cloudml.Test.find_one({'_id': ObjectId(test_id)})
    model = test.model#Model(test.model)
    try:
        if model.status != model.STATUS_TRAINED:
            raise InvalidOperationError("Train the model before")

        test.status = test.STATUS_IN_PROGRESS
        test.error = ""
        test.save()

        parameters = copy(test.parameters)
        metrics, raw_data = model.run_test(parameters)
        test.accuracy = metrics.accuracy

        metrics_dict = metrics.get_metrics_dict()

        # TODO: Refactor this. Here are possible issues with conformity
        # between labels and values
        confusion_matrix = metrics_dict['confusion_matrix']
        confusion_matrix_ex = []
        for i, val in enumerate(metrics.classes_set):
            confusion_matrix_ex.append((val, confusion_matrix[i]))
        metrics_dict['confusion_matrix'] = confusion_matrix_ex
        n = len(raw_data) / 100 or 1
        metrics_dict['roc_curve'][1] = metrics_dict['roc_curve'][1][0::n]
        metrics_dict['roc_curve'][0] = metrics_dict['roc_curve'][0][0::n]
        metrics_dict['precision_recall_curve'][1] = \
            metrics_dict['precision_recall_curve'][1][0::n]
        metrics_dict['precision_recall_curve'][0] = \
            metrics_dict['precision_recall_curve'][0][0::n]
        test.metrics = metrics_dict
        test.classes_set = list(metrics.classes_set)
        test.status = Test.STATUS_COMPLETED

        if not model.comparable:
            # TODO: fix this
            model = db.cloudml.Model.find_one({'_id': model._id})
            model.comparable = True
            model.save()

        all_count = metrics._preds.size
        test.examples_count = all_count
        test.save()
        # store test examples
        from pmap import pmap
        from itertools import islice
        #ziped = izip(raw_data, metrics._labels,
        #                             metrics._preds)
        
        sliced = []
        concurency = 6
        n = all_count / concurency
        # for i in range(4):
        #     sliced.append(islice(ziped, i*n, (i+1)*n -1))
        for i in range(concurency):
            start = i*n
            stop = (i+1)*n -1
            sliced.append(izip(raw_data[start:stop],
                               metrics._labels[start:stop],
                               metrics._preds[start:stop]))
        def store(items):
            count = 0
            for row, label, pred in items:
                count += 1
                if count % 100 == 0:
                    logging.info('Stored %d rows' % count)
                row = decode(row)
                #weighted_data_input = get_weighted_data(model, row)
                example = db.cloudml.TestExample()
                example['data_input'] = row
                #example['weighted_data_input'] = dict(weighted_data_input)
                # TODO: Specify Example title column in raw data
                example['name'] = unicode(row['contractor.dev_profile_title'])
                example['label'] = unicode(label)
                example['pred_label'] = unicode(pred)
                #example['test'] = test
                example['test_name'] = test.name
                example['model_name'] = model.name
                example.save(check_keys=False)
        pmap(store, sliced)

    except Exception, exc:
        logging.error(exc)
        test.status = test.STATUS_ERROR
        test.error = str(exc)
        test.save()
        raise
    return 'Test completed'


def decode(row):
    for key, val in row.iteritems():
        try:
            if isinstance(val, basestring):
                row[key] = val.encode('ascii', 'ignore')
        except UnicodeDecodeError, exc:
            #logging.error('Error while decoding %s: %s', val, exc)
            row[key] = ""
    return row
