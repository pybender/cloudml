__author__ = 'ifouk'


import json
import unittest
import os
import logging
from StringIO import StringIO


from core.trainer.config import FeatureModel
from core.trainer.trainer import Trainer, DEFAULT_SEGMENT
from jsonpath import jsonpath
from core.trainer.store import store_trainer, load_trainer
from core.trainer.streamutils import streamingiterload

BASEDIR = 'testdata'
TARGET = 'target'
FORMATS = ['csv', 'json']


class TrainerSegmentTestCase(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
                            level=logging.DEBUG)
        self._config = FeatureModel(os.path.join(BASEDIR, 'trainer',
                                    'features_segment.json'))
        self._trainer = None

    def test_train_and_test(self):
        #for fmt in FORMATS:
        self._train()
        self.assertEquals(self._trainer._classifier[''].coef_.shape, (1, 15))
        self.assertEquals(self._trainer._classifier['USA'].coef_.shape, (1, 14))
        self.assertEquals(self._trainer._classifier['Canada'].coef_.shape, (1, 13))
        title_feature = self._trainer.features['Canada']['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertEquals(title_vectorizer.get_feature_names(), ['engineer',
                                                                 'python'])

        metr =  self._trainer.test(self._get_iterator())


    def test_predict(self):

        self._train()
        results = self._trainer.predict(self._get_iterator())
        self.assertEqual(results['classes'].tolist(),['0', '1'])


    def _train(self, fmt='json'):
        with open(os.path.join(BASEDIR, 'trainer',
                               'trainer.data.segment.{}'.format(fmt))) as fp:
            self._data = list(streamingiterload(
                fp.readlines(), source_format=fmt))

        self._trainer = Trainer(self._config)
        self._trainer.train(self._data)

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
        self.assertTrue(transform[''].has_key('X'))
        self.assertTrue(transform['']['X'].shape[0], 3)
        self.assertTrue(transform['USA'].has_key('X'))
        self.assertTrue(transform['USA']['X'].shape[0], 2)
        self.assertTrue(transform['Canada'].has_key('X'))
        self.assertTrue(transform['Canada']['X'].shape[0], 1)


class TrainerTestCase(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
                            level=logging.DEBUG)
        self._config = FeatureModel(os.path.join(BASEDIR, 'trainer',
                                    'features.json'))
        self._trainer = None

    def test_train(self):
        for fmt in FORMATS:
            self._load_data(fmt)
            self.assertEquals(self._trainer._classifier[DEFAULT_SEGMENT].coef_.shape, (1, 19))
            title_feature = self._trainer.features[DEFAULT_SEGMENT]['contractor.dev_title']
            title_vectorizer = title_feature['transformer']
            self.assertEquals(title_vectorizer.get_feature_names(), ['engineer',
                                                                     'python'])

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
        self.assertEquals(self._trainer._classifier[DEFAULT_SEGMENT].coef_.shape, (1, 19))
        title_feature = self._trainer.features[DEFAULT_SEGMENT]['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertEquals(title_vectorizer.get_feature_names(), ['engineer',
                                                                 'python'])

    def test_store_feature_weights(self):
        for fmt in FORMATS:
            self._load_data(fmt)
            path = os.path.join(TARGET, 'test.weights.json')
            with open(path, 'wb') as fp:
                self._trainer.store_feature_weights(fp)

            self.assertTrue(os.path.exists(path), 'Weights were not stored!!!')

            positive_expected = ['contractor->dev_adj_score_recent',
                                 'contractor->dev_is_looking',
                                 ]

            # This is unintentional. Truly!
            negative_expected = ['contractor->dev_country->usa']

            with open(path) as fp:
                weights_dict = json.load(fp)
                weights = weights_dict['1']
                self.assertIn('positive', weights)
                self.assertIn('negative', weights)

                container = jsonpath(weights['positive'], '$.*.name')
                for item in positive_expected:
                    self.assertIn(item, container,
                                  'Item %s not in weights!' % item)

                container = jsonpath(weights['negative'], '$.*.name')
                for item in negative_expected:
                    self.assertIn(item, container,
                                  'Item %s not in weights!' % item)

    def test_store_and_load_trainer(self):
        for fmt in FORMATS:
            self._load_data(fmt)
            path = os.path.join(TARGET, 'feature.model')
            with open(path, 'wb') as fp:
                store_trainer(self._trainer, fp)

            self.assertTrue(os.path.exists(path), 'Feature model not stored!!!')

            with open(path, 'rb') as fp:
                new_trainer = load_trainer(fp)

            old_model = self._trainer._feature_model
            new_model = new_trainer._feature_model
            self.assertEqual(old_model.target_variable, new_model.target_variable)
            self.assertEqual(old_model.schema_name, new_model.schema_name)

            # Check that two models have same feature types
            self.assertEqual(len(old_model._named_feature_types),
                             len(new_model._named_feature_types))
            for name, feature_type in old_model._named_feature_types.items():
                self.assertDictEqual(feature_type,
                                     new_model._named_feature_types[name])

            # Check that two models have same feature types
            self.assertEqual(len(old_model.features), len(new_model.features))
            self.assertEqual(old_model.features.keys(), new_model.features.keys())

    def test_test(self):
        """
        Tests a binary classifier for csv and binary formats
        """
        from numpy import ndarray
        from core.trainer.metrics import ClassificationModelMetrics
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
            self.assertTrue(roc_curve.has_key(pos_label))
            self.assertEqual(2, len(roc_curve[pos_label]))
            self.assertIsInstance(roc_curve[pos_label][0], ndarray)
            self.assertIsInstance(roc_curve[pos_label][1], ndarray)

            self.assertTrue(metrics.roc_auc.has_key(pos_label))
            self.assertEquals(metrics.roc_auc[pos_label], 1.0)

            self.assertEquals(metrics.avarage_precision, 0.0)
            # make sure we have tested all published metrics
            for key in ClassificationModelMetrics.BINARY_METRICS.keys():
                self.assertTrue(hasattr(metrics, key),
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
            for clazz, clazz_weights in weights.iteritems():
                self.assertTrue(clazz_weights.has_key('positive'))
                self.assertTrue(clazz_weights.has_key('negative'))
                self.assertIsInstance(clazz_weights['positive'], list)
                self.assertIsInstance(clazz_weights['negative'], list)


    def test_test_ndim_outcome(self):
        """
        Tests a multiclass classifier
        """
        from numpy import ndarray
        from core.trainer.metrics import ClassificationModelMetrics

        self._config = FeatureModel(os.path.join(BASEDIR, 'trainer',
                                                 'features.ndim_outcome.json'))

        with open(os.path.join(BASEDIR, 'trainer', 'trainer.data.ndim_outcome.json')) as fp:
            self._data = list(streamingiterload(fp.readlines(), source_format='json'))

        self._trainer = Trainer(self._config)
        self._trainer.train(self._data)

        #
        # Testing metrics
        #
        metrics = self._trainer.test(self._data)
        self.assertIsInstance(metrics, ClassificationModelMetrics)
        self.assertEquals(metrics.accuracy, 1.0)
        self.assertIsInstance(metrics.confusion_matrix, ndarray)
        #precision, recall = metrics.precision_recall_curve
        #self.assertIsInstance(precision, ndarray)
        #self.assertIsInstance(recall, ndarray)
        roc_curve = metrics.roc_curve
        for pos_label in metrics.classes_set:
            self.assertTrue(roc_curve.has_key(pos_label))
            self.assertEqual(2, len(roc_curve[pos_label]))
            self.assertIsInstance(roc_curve[pos_label][0], ndarray)
            self.assertIsInstance(roc_curve[pos_label][1], ndarray)

            self.assertTrue(metrics.roc_auc.has_key(pos_label))
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
            self.assertTrue(clazz in metrics.classes_set)
            self.assertTrue(clazz_weights.has_key('positive'))
            self.assertTrue(clazz_weights.has_key('negative'))
            self.assertIsInstance(clazz_weights['positive'], list)
            self.assertIsInstance(clazz_weights['negative'], list)

    def test_transform(self):
        for fmt in ['json']: #FORMATS:
            self._load_data(fmt)
            transform = self._trainer.transform(self._data)
            self.assertEqual(1, len(transform))
            self.assertEqual(transform[DEFAULT_SEGMENT]['Y'], [1, 0, 0, 1, 0, 1])
            self.assertTrue(transform[DEFAULT_SEGMENT].has_key('X'))
            self.assertTrue(transform[DEFAULT_SEGMENT]['X'].shape[0], 6)

    def _load_data(self, fmt):
        """
        Load test data.
        """
        with open(os.path.join(BASEDIR, 'trainer',
                               'trainer.data.{}'.format(fmt))) as fp:
            self._data = list(streamingiterload(
                fp.readlines(), source_format=fmt))

        self._trainer = Trainer(self._config)
        self._trainer.train(self._data)

if __name__ == '__main__':
    unittest.main()
