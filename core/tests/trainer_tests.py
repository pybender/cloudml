__author__ = 'ifouk'


import json
import unittest
import os
import logging

from core.trainer.config import FeatureModel
from core.trainer.trainer import Trainer
from jsonpath import jsonpath
from core.trainer.store import store_trainer, load_trainer

BASEDIR = 'testdata'
TARGET = 'target'


class TrainerTestCase(unittest.TestCase):

    def setUp(self):
        logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
                            level=logging.DEBUG)
        self._config = FeatureModel(os.path.join(BASEDIR, 'trainer',
                                    'features.json'))
        self._load_data()
        self._trainer = Trainer(self._config)
        self._trainer.train(self._data)

    def test_train(self):
        self.assertEquals(self._trainer._classifier.coef_.shape, (1, 4))
        title_feature = self._config.features['contractor.dev_title']
        title_vectorizer = title_feature['transformer']
        self.assertEquals(title_vectorizer.get_feature_names(), ['engineer',
                                                                 'python'])

    def test_store_feature_weights(self):
        path = os.path.join(TARGET, 'test.weights.json')
        with open(path, 'wb') as fp:
            self._trainer.store_feature_weights(fp)

        self.assertTrue(os.path.exists(path), 'Weights were not stored!!!')

        positive_expected = ['contractor.dev_adj_score_recent',
                             'contractor.dev_is_looking',
                             'contractor.dev_title.engineer']

        # This is unintentional. Truly!
        negative_expected = ['contractor.dev_title.python']

        with open(path) as fp:
            weights = json.load(fp)
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

    def _load_data(self):
        """
        Load test data.
        """
        with open(os.path.join(BASEDIR, 'trainer', 'trainer.data.json')) as fp:
            self._data = json.load(fp)

if __name__ == '__main__':
    unittest.main()
