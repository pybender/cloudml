__author__ = 'ifoukarakis'

"""
Created on Jan 24, 2013

@author: ifoukarakis

"""
import unittest
import os

from core.trainer.config import FeatureModel, SchemaException
from sklearn.feature_extraction.text import TfidfVectorizer
from core.trainer.featuretype import regex_parse, str_to_int
from core.trainer.scalers import ScalerException

BASEDIR = 'testdata'


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = FeatureModel(os.path.join(BASEDIR, 'features.json'))

    def test_load_features(self):
        self.assertEqual(1, len(self.config._named_feature_types))
        self.assertIn('employer.op_timezone',
                      self.config.required_feature_names)
        self.assertNotIn('tsexams',
                         self.config.required_feature_names)

    def test_load_vectorizers(self):
        expected_vectorizers = {
            'country_pair': 'CountVectorizer',
            'tsexams': 'DictVectorizer',
            'contractor.dev_blurb': 'TfidfVectorizer',
            'contractor.dev_profile_title': 'TfidfVectorizer'
        }

        for name, expected_vect in expected_vectorizers.items():
            feature = self.config.features[name]
            self.assertEqual(expected_vect,
                             feature['transformer'].__class__.__name__,
                             'Invalid vectorizer for feature %s' % name)

    def test_process_classifier(self):
        config = {
            'classifier': {
                'type': 'logistic regression',
                'penalty': 'l1',
                'dual': False,
                'C': 1.0,
                'fit_intercept': True,
                'intercept_scaling': 1.0,
                'class_weight': None,
                'tol': None,
                'should': 'ignore'
            }
        }
        self.config._process_classifier(config)
        self.assertEqual('l1', self.config.classifier['penalty'])
        self.assertEqual(False, self.config.classifier['dual'])
        self.assertEqual(None, self.config.classifier['tol'])
        self.assertNotIn('type', self.config.classifier)
        self.assertNotIn('should', self.config.classifier)

    def test_process_classifier_default_penalty(self):
        config = {
            'classifier': {
                'type': 'logistic regression'
            }
        }
        self.config._process_classifier(config)
        self.assertEqual('l2', self.config.classifier['penalty'])

    def test_process_classifier_invalid_type(self):
        config = {
            'classifier': {
                'type': 'magic guess',
                'penalty': 'l1'
            }
        }
        try:
            self.config._process_classifier(config)
            self.fail('Should not be able to create classifier config for '
                      'invalid classifier type')
        except SchemaException:
            # Should happen
            pass

    def test_process_classifier_no_type(self):
        config = {
            'classifier': {
                'penalty': 'l1'
            }
        }
        try:
            self.config._process_classifier(config)
            self.fail('Should not be able to create classifier with no '
                      'classifier type defined')
        except SchemaException:
            # Should happen
            pass

    def test_process_named_feature_type(self):
        named_ft = {
            'name': 'floating_point',
            'type': 'regex',
            'params': {'pattern': '(\d\.\d+)'}
        }
        self.config._process_named_feature_type(named_ft)
        self.assertEqual(2, len(self.config._named_feature_types))
        ft_instance = self.config._named_feature_types['floating_point']
        self.assertEqual(ft_instance._strategy, regex_parse)
        self.assertEqual(ft_instance._params, named_ft['params'])

    def test_process_named_feature_type_unknown_type(self):
        named_ft = {
            'name': 'floating_point',
            'type': 'zavarakatranemia',
            'params': {'pattern': '(\d\.\d+)'}
        }
        try:
            self.config._process_named_feature_type(named_ft)
            self.fail('Shouldn\'t be able to create unknown type')
        except SchemaException:
            pass

        self.assertEqual(1, len(self.config._named_feature_types))

    def test_process_named_feature_type_missing_params(self):
        named_ft = {
            'name': 'floating_point',
            'type': 'regex'
        }
        try:
            self.config._process_named_feature_type(named_ft)
            self.fail('Shouldn\'t be able to create type with required params')
        except SchemaException:
            pass

        self.assertEqual(1, len(self.config._named_feature_types))

    def test_process_named_feature_type_no_name(self):
        named_ft = {
            'type': 'regex',
            'params': {'pattern': '(\d\.\d+)'}
        }
        try:
            self.config._process_named_feature_type(named_ft)
            self.fail('Shouldn\'t be able to create named type without name')
        except SchemaException:
            pass

        self.assertEqual(1, len(self.config._named_feature_types))

    def test_process_anonymous_feature_type(self):
        ft = {
            'name': 'floating_point',
            'type': 'regex',
            'params': {'pattern': '(\d\.\d+)'}
        }
        ft_instance = self.config._process_feature_type(ft)
        # Make sure that it is not added in named feature types
        self.assertEqual(1, len(self.config._named_feature_types))
        self.assertEqual(ft_instance._strategy, regex_parse)
        self.assertEqual(ft_instance._params, ft['params'])

    def test_process_anonymous_feature_type_unknown_type(self):
        ft = {
            'type': 'zavarakatranemia',
            'params': {'pattern': '(\d\.\d+)'}
        }
        try:
            self.config._process_named_feature_type(ft)
            self.fail('Shouldn\'t be able to create unknown type')
        except SchemaException:
            pass

        self.assertEqual(1, len(self.config._named_feature_types))

    def test_process_anonymous_feature_type_missing_params(self):
        ft = {
            'name': 'floating_point',
            'type': 'regex'
        }
        try:
            self.config._process_named_feature_type(ft)
            self.fail('Shouldn\'t be able to create type with required params')
        except SchemaException:
            pass

        self.assertEqual(1, len(self.config._named_feature_types))

    def test_process_feature_with_named_type(self):
        feature = {
            'name': 'another_test_feature',
            'type': 'str_to_timezone'
        }
        self.config._process_feature(feature)
        self.assertIn('another_test_feature', self.config.features)
        result = self.config.features['another_test_feature']
        self.assertEqual('another_test_feature', result['name'])
        self.assertEqual(self.config._named_feature_types['str_to_timezone'],
                         result['type'])
        self.assertIsNone(result['transformer'])
        self.assertTrue(result['required'])

    def test_process_feature_not_required(self):
        feature = {
            'name': 'another_test_feature',
            'type': 'str_to_timezone',
            'is-required': False
        }
        self.config._process_feature(feature)
        self.assertIn('another_test_feature', self.config.features)
        result = self.config.features['another_test_feature']
        self.assertEqual('another_test_feature', result['name'])
        self.assertEqual(self.config._named_feature_types['str_to_timezone'],
                         result['type'])
        self.assertIsNone(result['transformer'])
        self.assertFalse(result['required'])

    def test_process_feature_anonymous_feature_type(self):
        feature = {
            'name': 'another_test_feature',
            'type': 'int',
            'is-required': False
        }
        self.config._process_feature(feature)
        self.assertIn('another_test_feature', self.config.features)
        result = self.config.features['another_test_feature']
        self.assertEqual('another_test_feature', result['name'])
        self.assertEqual(result['type']._strategy, str_to_int)
        self.assertIsNone(result['transformer'])
        self.assertFalse(result['required'])

    def test_process_feature_with_scaler(self):
        from sklearn.preprocessing import MinMaxScaler 
        feature = {
            'name': 'another_test_feature',
            'scaler': {
                'type': 'ee'
            }
        }
        try:
            self.config._process_feature(feature)
        except ScalerException:
            pass

        feature = {
            'name': 'another_test_feature',
            'scaler': {
                'type': 'MinMaxScaler',
                'feature_range_max': 3,
                'copy': False
            }
        }

        self.config._process_feature(feature)
        self.assertIn('another_test_feature', self.config.features)
        result = self.config.features['another_test_feature']
        scaler = result['scaler']
        self.assertIsNotNone(scaler)
        self.assertIsInstance(scaler, MinMaxScaler)

    def test_process_feature_with_transformer(self):
        feature = {
            'name': 'another_test_feature',
            'transformer': {
                'type': 'Tfidf',
                'ngram_range_min': 1,
                'ngram_range_max': 1,
                'min_df': 10
            }
        }
        self.config._process_feature(feature)
        self.assertIn('another_test_feature', self.config.features)
        result = self.config.features['another_test_feature']
        self.assertEqual('another_test_feature', result['name'])
        transformer = result['transformer']
        self.assertIsNotNone(transformer)
        self.assertIsInstance(transformer, TfidfVectorizer)

    def test_process_feature_without_name(self):
        feature = {
            'type': 'int'
        }
        try:
            self.config._process_feature(feature)
            self.fail('Should not be able to create feature without name')
        except SchemaException:
            # Should happen
            pass

if __name__ == '__main__':
    unittest.main()
