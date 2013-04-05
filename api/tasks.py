import json
from copy import copy
from api import celery, db

from core.trainer.trainer import Trainer
from core.trainer.config import FeatureModel


class InvalidOperationError(Exception):
    pass


@celery.task
def train_model(model_id, parameters):
    """
    Train new model
    """
    try:
        model = db.Model.find_one({'_id': model_id})
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
        model.status = model.STATUS_TRAINED
        model.save()
    except Exception, exc:
        model.status = model.STATUS_ERROR
        model.error = str(exc)
        model.save()
        raise

    return "Model trained at %s" % trainer.train_time


def run_test(test, model):
    """
    Running tests for trained model
    """
    pass
    # try:
    #     if model.status != Model.STATUS_TRAINED:
    #         raise InvalidOperationError("Train model before")

    #     test.status = Test.STATUS_IN_PROGRESS
    #     test.error = ""
    #     db.session.merge(test)
    #     db.session.commit()
    #     parameters = copy(test.parameters)
    #     group_by = parameters.pop('group_by')
    #     metrics, raw_data = model.run_test(parameters)
    #     test.accuracy = metrics.accuracy

    #     metrics_dict = metrics.get_metrics_dict()

    #     # TODO: Refactor this. Here are possible issues with conformity
    #     # between labels and values
    #     confusion_matrix = metrics_dict['confusion_matrix']
    #     confusion_matrix_ex = []
    #     for i, val in enumerate(metrics.classes_set):
    #         confusion_matrix_ex.append((val, confusion_matrix[i]))
    #     metrics_dict['confusion_matrix'] = confusion_matrix_ex
    #     n = len(raw_data) / 100 or 1
    #     metrics_dict['roc_curve'][1] = metrics_dict['roc_curve'][1][0::n]
    #     metrics_dict['roc_curve'][0] = metrics_dict['roc_curve'][0][0::n]
    #     metrics_dict['precision_recall_curve'][1] = \
    #         metrics_dict['precision_recall_curve'][1][0::n]
    #     metrics_dict['precision_recall_curve'][0] = \
    #         metrics_dict['precision_recall_curve'][0][0::n]
    #     test.metrics = metrics_dict
    #     test.classes_set = list(metrics.classes_set)
    #     test.status = Test.STATUS_COMPLETED

    #     if not model.comparable:
    #         model.comparable = True
    #         db.session.merge(model)

    #     db.session.merge(test)
    #     db.session.commit()
    #     # store test data in db
    #     Data.loads_from_raw_data(model, test, raw_data,
    #                              metrics._labels, metrics._preds,
    #                              group_by)
    # except Exception, exc:
    #     test.status = Test.STATUS_ERROR
    #     test.error = str(exc)
    #     db.session.rollback()
    #     db.session.merge(test)
    #     db.session.commit()
    #     return 'Failed'
    # return 'Test completed'
