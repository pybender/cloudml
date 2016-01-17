# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>

from mock import MagicMock
import numpy
import json
import unittest
import os
import logging

from cloudml.trainer.config import FeatureModel
from cloudml.trainer.trainer import Trainer, DEFAULT_SEGMENT, \
    _adjust_classifier_class
from jsonpath import jsonpath
from cloudml.trainer.store import store_trainer, load_trainer
from cloudml.trainer.streamutils import streamingiterload
from cloudml.tests.test_utils import get_iterator

TARGET = 'target'
FORMATS = ['csv', 'json']

BASEDIR = 'testdata'
TRAINER_NDIM_OUTCOME = os.path.join(
    BASEDIR, 'trainer', 'trainer.data.ndim_outcome.json')
FEATURES_NDIM_OUTCOME = os.path.join(
    BASEDIR, 'trainer', 'features.ndim_outcome.json')


class BaseTrainerTestCase(unittest.TestCase):
    FEATURES_FILE = 'features.json'
    FLOAT_ACCURACY = 15

    def setUp(self):
        logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
                            level=logging.DEBUG)
        self._config = FeatureModel(os.path.join(BASEDIR, 'trainer',
                                    self.FEATURES_FILE))
        self._trainer = None

    def _load_data(self, fmt='json'):
        """
        Load test data.
        """
        with open(os.path.join(BASEDIR, 'trainer',
                               'trainer.data.{}'.format(fmt))) as fp:
            self._data = list(streamingiterload(
                fp.readlines(), source_format=fmt))

        self._trainer = Trainer(self._config)
        self._trainer.train(self._data)

    def _train(self, fmt='json', store_vect_data=False,
               configure=None, start_training=True):
        with open(os.path.join(BASEDIR, 'trainer',
                               'trainer.data.segment.{}'.format(fmt))) as fp:
            self._data = list(streamingiterload(
                fp.readlines(), source_format=fmt))

        self._trainer = Trainer(self._config)
        if configure is not None:
            configure(self._trainer)
        if start_training:
            self._trainer.train(self._data, store_vect_data=store_vect_data)

    def _get_iterator(self, fmt='json'):
        with open(os.path.join(BASEDIR, 'trainer',
                               'trainer.data.segment.{}'.format(fmt))) as fp:
            self._data = list(streamingiterload(
                fp.readlines(), source_format=fmt))

        return self._data

    def assertFloatEqual(self, expected, original):
        return self.assertAlmostEqual(expected, original,
                                      places=self.FLOAT_ACCURACY)


class TrainerSegmentTestCase(BaseTrainerTestCase):
    FEATURES_FILE = 'features_segment.json'

    def test_get_nonzero_vectorized_data(self):
        self._train()
        vect_data = self._trainer.get_nonzero_vectorized_data()
        val = vect_data[''][u'contractor.dev_adj_score_recent']
        self.assertEquals(val, 1.0)

    def test_get_vectorized_data(self):
        self._train()
        vect_data = self._trainer._get_vectorized_data(
            '', self._trainer._train_prepare_feature)

    # def test_vect_data2csv(self):
    #     self._train(store_vect_data=True)
    #     self._trainer.vect_data2csv('vect_data.csv')

    #     self._train()
    #     self.assertRaises(
    #         ValueError, self._trainer.vect_data2csv, 'vect_data.csv')

    def test_trained_model_visualization(self):
        # for fmt in FORMATS:
        self._train()
        self.assertEquals(self._trainer._classifier[''].coef_.shape, (1, 15))
        self.assertEquals(self._trainer._classifier['USA'].coef_.shape,
                          (1, 14))
        self.assertEquals(self._trainer._classifier['Canada'].coef_.shape,
                          (1, 13))
        weights = {'': [[0.064919069751063666,
                         0.0,
                         0.19722302855611831,
                         0.79525860445052987,
                         0.093483460441531663,
                         0.16353058402912282,
                         0.064919069751063666,
                         0.064919069751063666,
                         0.064919069751063666,
                         0.0,
                         0.0,
                         0.0,
                         0.0,
                         0.0,
                         0.0]],
                   'Canada': [[0.0,
                               0.0,
                               0.0010504036212994055,
                               0.0010504036212994055,
                               1.4492042523793169,
                               1.2755465085032549,
                               0.0021008072425988118,
                               0.0021008072425988118,
                               0.11301245547569275,
                               0.0,
                               0.018907265183389307,
                               0.018907265183389307,
                               0.0084032289703952472]],
                   'USA': [[0.12371049253298733,
                            0.037212322240579243,
                            0.17299634058481614,
                            0.75354952537172959,
                            0.381054179743532,
                            0.17299634058481614,
                            0.17299634058481617,
                            0.17299634058481617,
                            0.037212322240579243,
                            0.0, 0.0, 0.0,
                            0.33491090016521313,
                            0.14884928896231697]]}
        for segment in weights:
            visualization = \
                self._trainer.get_visualization(segment)['weights'][1]
            print [item['feature_weight'] for item in
                   visualization['positive']]
            print [item['feature_weight'] for item in
                   visualization['negative']]
            # TODO:
            # self._trainer.weights[segment][0] = map(lambda x: round(x, 15),
            #    self._trainer._feature_weights[segment][0])
            # weights[segment][0] = map(lambda x: round(x, 15),
            #    weights[segment][0])
            # self.assertListEqual(self._trainer._feature_weights[segment][0],
            #    weights[segment][0])
        expected = {'feature_weight': 0.19563459457810858,
                    'name': u'contractor->skills->microsoft-word',
                    'weight': 0.19563459457810858}
        USA_weights = self._trainer.get_weights('USA')
        self.assertFloatEqual(
            expected['feature_weight'],
            USA_weights[1]['positive'][0]['feature_weight']
        )
        self.assertFloatEqual(
            expected['weight'],
            USA_weights[1]['positive'][0]['weight']
        )
        self.assertEqual(
            expected['name'],
            self._trainer.get_weights('USA')[1]['positive'][0]['name']
        )
        self.assertEqual(
            set(expected),
            set(self._trainer.get_weights('USA')[1]['positive'][0])
        )

    def test_train_and_test(self):
        # for fmt in FORMATS:
        self._train()
        self.assertEquals(self._trainer._classifier[''].coef_.shape, (1, 15))
        self.assertEquals(self._trainer._classifier['USA'].coef_.shape,
                          (1, 14))
        self.assertEquals(self._trainer._classifier['Canada'].coef_.shape,
                          (1, 13))
        title_feature = \
            self._trainer.features['Canada']['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertEquals(title_vectorizer.get_feature_names(), ['engineer',
                                                                 'python'])
        self.assertEqual(['0', '1'], self._trainer._get_labels())

        metr = self._trainer.test(self._get_iterator())

    def test_predict(self):
        self._train()
        results = self._trainer.predict(self._get_iterator())
        self.assertEqual(results['classes'].tolist(), ['0', '1'])

    def _train(self, fmt='json', store_vect_data=False):
        with open(os.path.join(BASEDIR, 'trainer',
                               'trainer.data.segment.{}'.format(fmt))) as fp:
            self._data = list(streamingiterload(
                fp.readlines(), source_format=fmt))

        self._trainer = Trainer(self._config)
        self._trainer.train(self._data, store_vect_data=store_vect_data)

    def _get_iterator(self, fmt='json'):
        with open(os.path.join(BASEDIR, 'trainer',
                               'trainer.data.segment.{}'.format(fmt))) as fp:
            self._data = list(streamingiterload(
                fp.readlines(), source_format=fmt))

        return self._data

    def test_transform(self):
        self._train()
        transform = self._trainer.transform(self._data)
        self.assertEqual(3, len(transform))
        self.assertEqual(transform['']['Y'], [1, 0])
        self.assertEqual(transform['USA']['Y'], [0, 1])
        self.assertEqual(transform['Canada']['Y'], [1, 0])
        self.assertTrue('X' in transform[''])
        self.assertTrue(transform['']['X'].shape[0], 3)
        self.assertTrue('X' in transform['USA'])
        self.assertTrue(transform['USA']['X'].shape[0], 2)
        self.assertTrue('X' in transform['Canada'])
        self.assertTrue(transform['Canada']['X'].shape[0], 1)

    def test_get_labels(self):
        segment1_mock = MagicMock()
        segment2_mock = MagicMock()
        classifier = {
            'Segment1': segment1_mock,
            'Segment2': segment2_mock
        }
        # [20, 10] - count of the elements in each segment
        segments = dict(zip(classifier.keys(), [20, 10]))
        segment1_mock._enc = 'something'
        segment1_mock.classes_.tolist.return_value = ['False', 'True']
        segment2_mock._enc = 'something'
        segment2_mock.classes_.tolist.return_value = ['False', 'True']

        trainer = Trainer(self._config)
        trainer.set_classifier(classifier)
        trainer._segments = segments
        trainer.feature_model.group_by = ['smth']
        self.assertEqual(['False', 'True'], trainer._get_labels())

        # Test assumption of segments having identical classes set violated
        segment1_mock.classes_.tolist.return_value = ['True', 'False']
        segment2_mock.classes_.tolist.return_value = ['False', 'True']
        trainer = Trainer(self._config)
        trainer.set_classifier(classifier)
        trainer._segments = segments
        trainer.feature_model.group_by = ['smth']
        self.assertRaises(AssertionError, trainer._get_labels)


class LogisticRegressionTrainerTestCase(BaseTrainerTestCase):
    FEATURES_FILE = 'features.json'

    def test_train(self):
        from sklearn.linear_model import LogisticRegression
        for fmt in FORMATS:
            self._load_data(fmt)
            self.assertEquals(
                type(self._trainer.classifier), LogisticRegression)
            print self._trainer.classifier.classes_.tolist()
            self.assertEquals(
                self._trainer._classifier[DEFAULT_SEGMENT].coef_.shape,
                (1, 19))
            title_feature = \
                self._trainer.features[DEFAULT_SEGMENT]['contractor.dev_title']
            title_vectorizer = title_feature['transformer']
            self.assertEquals(title_vectorizer.get_feature_names(),
                              ['engineer', 'python'])
            self.assertEqual(['0', '1'], self._trainer._get_labels())
            vis = self._trainer.get_visualization(segment='default')
            self.assertEquals(vis['classifier_type'], u'logistic regression')

    def test_train_class_weight(self):
        config = {
            'classifier': {
                'type': 'logistic regression',
                'params': {
                    'penalty': 'l2',
                    'class_weight': {
                        '0': 1,
                        '1': 2
                    }
                },
            }
        }
        self._config._process_classifier(config)
        self._load_data('json')
        self.assertEquals(
            self._trainer._classifier[DEFAULT_SEGMENT].coef_.shape, (1, 19))
        features = self._trainer.features[DEFAULT_SEGMENT]
        title_feature = features['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertEquals(
            title_vectorizer.get_feature_names(),
            ['engineer', 'python'])

    def test_store_feature_weights(self):
        for fmt in FORMATS:
            self._load_data(fmt)
            path = os.path.join(TARGET, 'test.weights.json')
            with open(path, 'wb') as fp:
                self._trainer.store_feature_weights(fp)

            self.assertTrue(os.path.exists(path), 'Weights were not stored!!!')

            expected = ['contractor->dev_is_looking',
                        'contractor->dev_adj_score_recent',
                        'contractor->dev_country->usa']

            with open(path) as fp:
                weights_dict = json.load(fp)
                # FIXME: JSON always dumps all keys as string!
                # Here should be int type
                weights = weights_dict['1']
                self.assertIn('positive', weights)
                self.assertIn('negative', weights)

                container = jsonpath(weights['negative'], '$.*.name') + \
                    jsonpath(weights['positive'], '$.*.name')
                for item in expected:
                    self.assertIn(item, container,
                                  'Item %s not in weights!' % item)

    def test_store_and_load_trainer(self):
        for fmt in FORMATS:
            self._load_data(fmt)
            path = os.path.join(TARGET, 'feature.model')
            with open(path, 'wb') as fp:
                store_trainer(self._trainer, fp)

            self.assertTrue(
                os.path.exists(path), 'Feature model not stored!!!')

            with open(path, 'rb') as fp:
                new_trainer = load_trainer(fp)

            old_model = self._trainer._feature_model
            new_model = new_trainer._feature_model
            self.assertEqual(
                old_model.target_variable, new_model.target_variable)
            self.assertEqual(old_model.schema_name, new_model.schema_name)

            # Check that two models have same feature types
            self.assertEqual(len(old_model._named_feature_types),
                             len(new_model._named_feature_types))
            for name, feature_type in old_model._named_feature_types.items():
                self.assertDictEqual(feature_type,
                                     new_model._named_feature_types[name])

            # Check that two models have same feature types
            self.assertEqual(len(old_model.features), len(new_model.features))
            self.assertEqual(
                old_model.features.keys(), new_model.features.keys())

    def test_test(self):
        """
        Tests a binary classifier for csv and binary formats
        """
        from numpy import ndarray
        from cloudml.trainer.metrics import ClassificationModelMetrics
        for fmt in FORMATS:
            self._load_data(fmt)
            metrics = self._trainer.test(self._data)
            self.assertIsInstance(metrics, ClassificationModelMetrics)
            self.assertEquals(metrics.accuracy, 1.0)
            self.assertIsInstance(metrics.confusion_matrix, ndarray)
            precision, recall = metrics.precision_recall_curve
            self.assertIsInstance(precision, ndarray)
            self.assertIsInstance(recall, ndarray)

            roc_curve = metrics.roc_curve
            pos_label = metrics.classes_set[1]
            self.assertTrue(pos_label in roc_curve)
            self.assertEqual(2, len(roc_curve[pos_label]))
            self.assertIsInstance(roc_curve[pos_label][0], ndarray)
            self.assertIsInstance(roc_curve[pos_label][1], ndarray)

            self.assertTrue(pos_label in metrics.roc_auc)
            self.assertEquals(metrics.roc_auc[pos_label], 1.0)

            self.assertEquals(metrics.average_precision, 0.0)
            # make sure we have tested all published metrics
            for key in ClassificationModelMetrics.BINARY_METRICS.keys():
                self.assertTrue(
                    hasattr(metrics, key),
                    'metric %s was not found or not tested' % (key))

            # make sure we can serialize the metric dictionary
            try:
                metrics_dict = metrics.get_metrics_dict()
                json.dumps(metrics_dict)
            except Exception, exc:
                self.fail(exc)

            #
            # Testing Weights, for a binary classifer
            #
            weights = self._trainer.get_weights()
            self.assertEqual(1, len(weights.keys()))
            for clazz_weights in weights.values():
                self.assertTrue('positive' in clazz_weights)
                self.assertTrue('negative' in clazz_weights)
                self.assertIsInstance(clazz_weights['positive'], list)
                self.assertIsInstance(clazz_weights['negative'], list)

    def test_test_ndim_outcome(self):
        """
        Tests a multiclass classifier
        """
        from numpy import ndarray
        from cloudml.trainer.metrics import ClassificationModelMetrics

        self._config = FeatureModel(os.path.join(BASEDIR, 'trainer',
                                                 'features.ndim_outcome.json'))
        with open(TRAINER_NDIM_OUTCOME) as fp:
            self._data = list(streamingiterload(
                fp.readlines(), source_format='json'))

        self._trainer = Trainer(self._config)
        self._trainer.train(self._data)

        #
        # Testing metrics
        #
        metrics = self._trainer.test(self._data)
        self.assertIsInstance(metrics, ClassificationModelMetrics)
        self.assertEquals(metrics.accuracy, 1.0)
        self.assertIsInstance(metrics.confusion_matrix, ndarray)
        # precision, recall = metrics.precision_recall_curve
        # self.assertIsInstance(precision, ndarray)
        # self.assertIsInstance(recall, ndarray)
        roc_curve = metrics.roc_curve
        for pos_label in metrics.classes_set:
            self.assertTrue(pos_label in roc_curve)
            self.assertEqual(2, len(roc_curve[pos_label]))
            self.assertIsInstance(roc_curve[pos_label][0], ndarray)
            self.assertIsInstance(roc_curve[pos_label][1], ndarray)

            self.assertTrue(pos_label in metrics.roc_auc)
            self.assertEquals(metrics.roc_auc[pos_label], 1.0)

        for key in ClassificationModelMetrics.MORE_DIMENSIONAL_METRICS.keys():
            self.assertTrue(hasattr(metrics, key),
                            'metric %s was not found or not tested' % (key))

        # make sure we can serialize the metric dictionary
        try:
            metrics_dict = metrics.get_metrics_dict()
            json.dumps(metrics_dict)
        except Exception, exc:
            self.fail(exc)

        #
        # Testing Weights
        #
        weights = self._trainer.get_weights()
        self.assertEqual(metrics.classes_count, len(weights.keys()))
        for clazz, clazz_weights in weights.iteritems():
            self.assertTrue(
                clazz in metrics.classes_set,
                "{0} not in {1}".format(clazz, metrics.classes_set))
            self.assertTrue('positive' in clazz_weights)
            self.assertTrue('negative' in clazz_weights)
            self.assertIsInstance(clazz_weights['positive'], list)
            self.assertIsInstance(clazz_weights['negative'], list)

    def test_transform(self):
        for fmt in ['json']:  # FORMATS:
            self._load_data(fmt)
            transform = self._trainer.transform(self._data)
            self.assertEqual(1, len(transform))
            self.assertEqual(
                transform[DEFAULT_SEGMENT]['Y'], [1, 0, 1, 0, 1])
            self.assertTrue('X' in transform[DEFAULT_SEGMENT])
            self.assertTrue(transform[DEFAULT_SEGMENT]['X'].shape[0], 6)

    def test_no_examples_for_label(self):
        """
        Tests the case there is no example for a given label
        """
        from numpy import ndarray
        from cloudml.trainer.metrics import ClassificationModelMetrics

        def do_train(exclude_labels):

            config = FeatureModel(FEATURES_NDIM_OUTCOME)

            with open(TRAINER_NDIM_OUTCOME) as fp:
                dataset = json.loads(fp.read())

            train_lines = json.dumps(dataset)
            test_lines = json.dumps(filter(
                lambda x: x['hire_outcome'] not in exclude_labels, dataset))

            train_data = list(streamingiterload(
                train_lines, source_format='json'))
            test_data = list(streamingiterload(
                test_lines, source_format='json'))

            self._trainer = Trainer(config)
            self._trainer.train(train_data)

            self._trainer = Trainer(config)
            self._trainer.train(train_data)
            return self._trainer.test(test_data)

        metrics = do_train(['class3'])
        self.assertEqual({DEFAULT_SEGMENT: [3]},
                         self._trainer._test_empty_labels)
        #
        # Testing metrics
        #
        self.assertIsInstance(metrics, ClassificationModelMetrics)
        self.assertEquals(metrics.accuracy, 1.0)
        self.assertIsInstance(metrics.confusion_matrix, ndarray)
        for pos_label in metrics.classes_set:
            self.assertFalse(numpy.all(numpy.isnan(
                metrics.roc_curve[pos_label][0])))
            self.assertFalse(numpy.all(numpy.isnan(
                metrics.roc_curve[pos_label][1])))
            if pos_label == 3:
                self.assertEquals(metrics.roc_auc[pos_label], 0.0)
            else:
                self.assertEquals(metrics.roc_auc[pos_label], 1.0)

        self.assertEqual(metrics.roc_auc, {1: 1.0, 2: 1.0, 3: 0.0})
        self.assertEqual(metrics.confusion_matrix.tolist(), [[2, 0, 0],
                                                             [0, 3, 0],
                                                             [0, 0, 0]])

        # exclude two labels
        metrics = do_train(['class1', 'class3'])
        self.assertEqual({DEFAULT_SEGMENT: [1, 3]},
                         self._trainer._test_empty_labels)
        #
        # Testing metrics
        #
        self.assertIsInstance(metrics, ClassificationModelMetrics)
        self.assertEquals(metrics.accuracy, 1.0)
        self.assertIsInstance(metrics.confusion_matrix, ndarray)
        for pos_label in metrics.classes_set:
            self.assertFalse(numpy.all(numpy.isnan(
                metrics.roc_curve[pos_label][0])))
            self.assertFalse(numpy.all(numpy.isnan(
                metrics.roc_curve[pos_label][1])))
            if pos_label in [1, 3]:
                self.assertEquals(metrics.roc_auc[pos_label], 0.0)
            else:
                self.assertEquals(metrics.roc_auc[pos_label], 0.0)

        self.assertEqual(metrics.roc_auc, {1: 0.0, 2: 0.0, 3: 0.0})
        self.assertEqual(metrics.confusion_matrix.tolist(), [[0, 0, 0],
                                                             [0, 3, 0],
                                                             [0, 0, 0]])

    def test_with_percent(self):
        data = get_iterator('trainer', 'test.data')

        self._train(start_training=False)
        self._trainer.clear_temp_data()
        self._trainer.train(self._data)
        metrics = self._trainer.test(data)
        self.assertEquals(self._trainer._count, 6)
        self.assertFloatEqual(metrics.accuracy, 0.66666666666666663)

        self._trainer.train(data, percent=40)
        self.assertEquals(self._trainer._count, 4)  # 6 - 40%
        metrics = self._trainer.test(data, percent=50)
        self.assertEquals(self._trainer._count, 3)  # 6 - 50%
        self.assertEquals(metrics.accuracy, 1)

        self.assertTrue(self._trainer._raw_data)
        self.assertTrue(self._trainer._vect_data)
        self._trainer.clear_temp_data()
        self.assertFalse(self._trainer._raw_data)
        self.assertFalse(self._trainer._vect_data)

    def test_corrupted_data(self):
        data = get_iterator('trainer', 'corrupted.data')

        self._train(start_training=False)

        from cloudml.trainer.exceptions import EmptyDataException
        with self.assertRaisesRegexp(
                EmptyDataException, "No rows found in the iterator"):
            self._trainer.train(data)

    def test_grid_search(self):
        data = get_iterator('trainer', 'trainer.data')
        test_data = get_iterator('trainer', 'test.data')
        self._train(start_training=False)
        params = {'penalty': ['l1', 'l2']}
        clf = self._trainer.grid_search(
            params, data, test_data, score='accuracy')['default']
        grids = []
        for item in clf.grid_scores_:
            grids.append({'parameters': item.parameters,
                          'mean': item.mean_validation_score})
        self.assertEquals(len(grids), 2)


class HelpersTestCase(unittest.TestCase):

    def setUp(self):
        self.feature_model = FeatureModel('./testdata/features.json')

    def test_adjust_classifier_class_boolean(self):
        f = self.feature_model.features['contractor.dev_is_looking']

        self.assertTrue(_adjust_classifier_class(f, 'True'))
        self.assertTrue(_adjust_classifier_class(f, 'true'))
        self.assertTrue(_adjust_classifier_class(f, '1'))

        self.assertFalse(_adjust_classifier_class(f, 'False'))
        self.assertFalse(_adjust_classifier_class(f, 'false'))
        self.assertFalse(_adjust_classifier_class(f, '0'))
        self.assertFalse(_adjust_classifier_class(f, '-1'))

    def test_adjust_classifier_class_int(self):
        f = self.feature_model.features['employer.op_tot_jobs_filled']

        self.assertEqual(1, _adjust_classifier_class(f, '1'))
        self.assertEqual(0, _adjust_classifier_class(f, '0'))
        self.assertEqual(-1, _adjust_classifier_class(f, '-1'))

    def test_adjust_classifier_class_float(self):
        f = self.feature_model.features['contractor.dev_adj_score_recent']

        self.assertEqual(1.1, _adjust_classifier_class(f, '1.1'))
        self.assertEqual(0.01, _adjust_classifier_class(f, '0.01'))
        self.assertEqual(-1.001, _adjust_classifier_class(f, '-1.001'))

    def test_adjust_classifier_class_ordinal(self):
        f = self.feature_model.features['hire_outcome']

        self.assertEqual(1, _adjust_classifier_class(f, '1'))
        self.assertEqual(0, _adjust_classifier_class(f, '0'))


class LinearSVRTestCase(BaseTrainerTestCase):
    FEATURES_FILE = 'features-svr.json'

    def test_train(self):
        self._load_data()
        self.assertEquals(
            self._trainer._classifier[DEFAULT_SEGMENT].coef_.shape, (1, 19))
        features = self._trainer.features
        title_feature = features[DEFAULT_SEGMENT]['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertEquals(
            title_vectorizer.get_feature_names(), ['engineer', 'python'])

        vis = self._trainer.get_visualization(segment='default')
        self.assertEquals(vis['classifier_type'], u'support vector regression')

        weights = vis['weights']
        self.assertTrue(weights)
        w = weights['default']['positive'][0]
        self.assertEquals(w['name'], u'contractor->skills->microsoft-word')
        self.assertFloatEqual(w['feature_weight'], 0.019903856547956178)
        self.assertFloatEqual(w['weight'], 0.049759641369890445)

    def test_test(self):
        from cloudml.trainer.metrics import RegressionModelMetrics
        self._load_data()
        metrics = self._trainer.test(self._data)
        self.assertIsInstance(metrics, RegressionModelMetrics)

        self.assertFloatEqual(
            metrics.explained_variance_score, 0.95987382954092304)
        self.assertFloatEqual(metrics.mean_absolute_error, 0.10015075257744899)
        self.assertFloatEqual(
            metrics.mean_squared_error,  0.010030280715664325)
        self.assertFloatEqual(metrics.r2_score, 0.95820716368473202)


class PolySVRTestCase(BaseTrainerTestCase):
    FEATURES_FILE = 'features-poly-svr.json'

    def test_train(self):
        self._load_data()
        features = self._trainer.features
        title_feature = features[DEFAULT_SEGMENT]['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertEquals(
            title_vectorizer.get_feature_names(), ['engineer', 'python'])

        vis = self._trainer.get_visualization(segment='default')
        self.assertEquals(vis['classifier_type'], u'support vector regression')
        self.assertFalse('weights' in vis)

    def test_test(self):
        from cloudml.trainer.metrics import RegressionModelMetrics
        self._load_data()
        metrics = self._trainer.test(self._data)
        self.assertIsInstance(metrics, RegressionModelMetrics)

        self.assertFloatEqual(
            metrics.explained_variance_score, 0.44692958439628583)
        self.assertFloatEqual(metrics.mean_absolute_error, 0.24808252972759073)
        self.assertFloatEqual(metrics.mean_squared_error,  0.14911762146334478)
        self.assertFloatEqual(metrics.r2_score, 0.37867657723606352)


class DecisionTreeClfTestCase(BaseTrainerTestCase):
    FEATURES_FILE = 'features-decision-tree-classifier.json'

    def test_train(self):
        self._load_data()
        features = self._trainer.features
        title_feature = features[DEFAULT_SEGMENT]['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertEquals(
            title_vectorizer.get_feature_names(), ['engineer', 'python'])

        vis = self._trainer.get_visualization(segment='default')
        self.assertEquals(vis['classifier_type'], u'decision tree classifier')
        self.assertTrue(vis['tree'], 'Decision Tree was not generated')

        weights = vis['weights']
        self.assertTrue(weights)
        w = weights[1]['positive'][0]
        self.assertEquals(w['name'], 'contractor->dev_title->engineer')
        self.assertFloatEqual(w['feature_weight'], 0.50609621254893444)
        self.assertFloatEqual(w['weight'], 1.0)

    def test_test(self):
        from cloudml.trainer.metrics import ClassificationModelMetrics
        self._load_data()
        metrics = self._trainer.test(self._data)
        self.assertIsInstance(metrics, ClassificationModelMetrics)
        self.assertEquals(metrics.accuracy, 1)
        self.assertEqual(metrics.roc_auc, {1: 1.0})


class ExtraTreesClfTestCase(BaseTrainerTestCase):
    FEATURES_FILE = 'features-extra-trees-classifier.json'

    def test_train(self):
        self._load_data()
        features = self._trainer.features
        title_feature = features[DEFAULT_SEGMENT]['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertEquals(
            title_vectorizer.get_feature_names(), ['engineer', 'python'])

        vis = self._trainer.get_visualization(segment='default')
        self.assertEquals(vis['classifier_type'], u'extra trees classifier')
        self.assertTrue(vis['trees'], 'Decision Trees was not generated')

        weights = vis['weights']
        self.assertTrue(weights)
        w = weights[1]['positive'][0]
        # TODO: why every time we have different weights?
        # self.assertEquals(w['name'], 'contractor->skills->microsoft-word')
        # self.assertEquals(w['feature_weight'], 0.02777777777777778)
        # self.assertEquals(w['weight'], 0.5)

    def test_test(self):
        from cloudml.trainer.metrics import ClassificationModelMetrics
        self._load_data()
        metrics = self._trainer.test(self._data)
        self.assertIsInstance(metrics, ClassificationModelMetrics)
        self.assertEquals(metrics.accuracy, 1)
        self.assertEqual(metrics.roc_auc, {1: 1.0})


class RandomForestClfTestCase(BaseTrainerTestCase):
    FEATURES_FILE = 'features-random-forest-classifier.json'

    def test_train(self):
        self._load_data()
        features = self._trainer.features
        title_feature = features[DEFAULT_SEGMENT]['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertEquals(
            title_vectorizer.get_feature_names(), ['engineer', 'python'])

        vis = self._trainer.get_visualization(segment='default')
        self.assertEquals(vis['classifier_type'], u'random forest classifier')
        self.assertTrue(vis['trees'], 'Decision Trees was not generated')

        weights = vis['weights']
        self.assertTrue(weights)
        w = weights[1]['positive'][0]
        # TODO: why every time we have different weights?
        # self.assertEquals(w['name'], 'contractor->skills->microsoft-word')
        # self.assertEquals(w['feature_weight'], 0.02777777777777778)
        # self.assertEquals(w['weight'], 0.5)

    def test_test(self):
        from cloudml.trainer.metrics import ClassificationModelMetrics
        self._load_data()
        metrics = self._trainer.test(self._data)
        self.assertIsInstance(metrics, ClassificationModelMetrics)


# TODO: sparse matrix was passed, but dense data is required.
# Use X.toarray() to convert to a dense numpy array.
# class GradientBoostingClfTestCase(BaseTrainerTestCase):
#     FEATURES_FILE = 'features-gradient-boosting-classifier.json'

#     def test_train(self):
#         self._load_data()
#         features = self._trainer.features
#         title_feature = features[DEFAULT_SEGMENT]['contractor.dev_title']
#         title_vectorizer = title_feature['transformer']
#         self.assertEquals(
#             title_vectorizer.get_feature_names(), ['engineer', 'python'])

#         vis = self._trainer.get_visualization(segment='default')
#         self.assertEquals(
#           vis['classifier_type'], u'gradient boosting classifier')

#         weights = vis['weights']
#         self.assertTrue(weights)
#         w = weights[1]['positive'][0]
#         # TODO: why every time we have different weights?
#         # self.assertEquals(w['name'], 'contractor->skills->microsoft-word')
#         # self.assertEquals(w['feature_weight'], 0.02777777777777778)
#         # self.assertEquals(w['weight'], 0.5)

#     def test_test(self):
#         from core.trainer.metrics import ClassificationModelMetrics
#         self._load_data()
#         metrics = self._trainer.test(self._data)
#         self.assertIsInstance(metrics, ClassificationModelMetrics)


# class SGDClfTestCase(BaseTrainerTestCase):
#     FEATURES_FILE = 'features-sgd-classifier.json'

#     def test_train(self):
#         self._load_data()
#         features = self._trainer.features
#         title_feature = features[DEFAULT_SEGMENT]['contractor.dev_title']
#         title_vectorizer = title_feature['transformer']
#         self.assertEquals(
#             title_vectorizer.get_feature_names(), ['engineer', 'python'])

#         vis = self._trainer.get_visualization(segment='default')
#         self.assertEquals(
#             vis['classifier_type'], u'random forest classifier')
#         self.assertTrue(vis['trees'], 'Decision Trees was not generated')

#         weights = vis['weights']
#         self.assertTrue(weights)
#         w = weights[1]['positive'][0]
#         # TODO: why every time we have different weights?
#         # self.assertEquals(w['name'], 'contractor->skills->microsoft-word')
#         # self.assertEquals(w['feature_weight'], 0.02777777777777778)
#         # self.assertEquals(w['weight'], 0.5)

#     def test_test(self):
#         from core.trainer.metrics import ClassificationModelMetrics
#         self._load_data()
#         metrics = self._trainer.test(self._data)
#         self.assertIsInstance(metrics, ClassificationModelMetrics)


class ModelWithPretrainedTransformer(BaseTrainerTestCase):
    """
    Tests the model with pretrained transformer defined.
    """
    FEATURES_FILE = 'features-with-pretrained-transformer.json'

    def setUp(self):
        super(ModelWithPretrainedTransformer, self).setUp()
        from cloudml.transformers.tests import _get_config, Transformer
        config = _get_config('transformer.json')
        self.transformer = Transformer(config)

    def test_train(self):
        # need to set get_transformer method
        from cloudml.trainer.exceptions import TransformerNotFound
        with self.assertRaisesRegexp(
                TransformerNotFound,
                "Transformer with name bestmatch not found"):
            self._train()

        from sklearn.utils.validation import NotFittedError
        with self.assertRaisesRegexp(
                NotFittedError, "TfidfVectorizer - Vocabulary wasn't fitted."):
            self._train(configure=self._get_configure_fn())

        self._train_transformer()
        self._train(configure=self._get_configure_fn())
        features = self._trainer.features
        title_feature = features[DEFAULT_SEGMENT]['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertItemsEqual(
            title_vectorizer.get_feature_names(), ['python'])

    def test_test(self):
        self._train_transformer()
        self._train(configure=self._get_configure_fn())

        from cloudml.trainer.metrics import ClassificationModelMetrics
        metrics = self._trainer.test(self._data)
        self.assertIsInstance(metrics, ClassificationModelMetrics)

    def _train_transformer(self):
        self.transformer.train(self._get_iterator())

    def _get_configure_fn(self):
        def configure(trainer):
            def get_transformer(name):
                return self.transformer.feature['transformer']

            trainer.set_transformer_getter(get_transformer)
        return configure
