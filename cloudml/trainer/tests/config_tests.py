# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>

import unittest
import os

from cloudml.trainer.config import FeatureModel, SchemaException
from sklearn.feature_extraction.text import TfidfVectorizer
from cloudml.trainer.feature_types import RegexFeatureTypeInstance, \
    PrimitiveFeatureTypeInstance
from cloudml.trainer.scalers import ScalerException


BASEDIR = 'testdata'


class ConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = FeatureModel(os.path.join(BASEDIR, 'features.json'))

    def test_process_group_by(self):
        group_by = ['wwww']
        with self.assertRaises(SchemaException):
            self.config._process_group_by(group_by)

        group_by = ['country_pair']
        self.config._process_group_by(group_by)

        self.assertEqual(self.config.group_by, group_by)

    def test_load_features(self):
        self.assertEqual(1, len(self.config._named_feature_types))
        self.assertIn('employer.op_timezone',
                      self.config.required_feature_names)
        self.assertNotIn('tsexams',
                         self.config.required_feature_names)
        self.assertTrue(str(self.config).startswith("Schema name: bestmatch"))

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
                'params': {
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
        }
        self.config._process_classifier(config)
        self.assertEqual('l1', self.config.classifier['penalty'])
        self.assertEqual(False, self.config.classifier['dual'])
        self.assertEqual(None, self.config.classifier['tol'])
        self.assertNotIn('type', self.config.classifier)
        self.assertNotIn('should', self.config.classifier)

    def test_process_classifier_class_weight(self):
        config = {
            'classifier': {
                'type': 'logistic regression',
                'params': {
                    'class_weight': {
                        '0': 1,
                        '1': 2
                    },
                }
            }
        }
        self.config._process_classifier(config)
        self.assertEqual(
            {
                '0': 1,
                '1': 2
            },
            self.config.classifier['class_weight']
        )

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
        with self.assertRaisesRegexp(SchemaException,
                                     "Invalid classifier type"):
            self.config._process_classifier(config)

    def test_process_classifier_no_type(self):
        config = {
            'classifier': {
                'penalty': 'l1'
            }
        }
        with self.assertRaisesRegexp(SchemaException,
                                     "Invalid classifier type"):
            self.config._process_classifier(config)

    def test_process_named_feature_type(self):
        named_ft = {
            'name': 'floating_point',
            'type': 'regex',
            'params': {'pattern': '(\d\.\d+)'}
        }
        self.config._process_named_feature_type(named_ft)
        self.assertEqual(2, len(self.config._named_feature_types))
        ft_instance = self.config._named_feature_types['floating_point']
        self.assertTrue(isinstance(ft_instance, RegexFeatureTypeInstance))
        self.assertEqual(ft_instance._params, named_ft['params'])

    def test_process_named_feature_type_unknown_type(self):
        named_ft = {
            'name': 'floating_point',
            'type': 'zavarakatranemia',
            'params': {'pattern': '(\d\.\d+)'}
        }
        with self.assertRaisesRegexp(SchemaException,
                                     "Unknown type: zavarakatranemia"):
            self.config._process_named_feature_type(named_ft)

        self.assertEqual(1, len(self.config._named_feature_types))

    def test_process_named_feature_type_missing_params(self):
        named_ft = {
            'name': 'floating_point',
            'type': 'regex'
        }
        with self.assertRaisesRegexp(SchemaException,
                                     "Cannot create instance of feature type"):
            self.config._process_named_feature_type(named_ft)

        self.assertEqual(1, len(self.config._named_feature_types))

    def test_process_named_feature_type_no_name(self):
        named_ft = {
            'type': 'regex',
            'params': {'pattern': '(\d\.\d+)'}
        }
        with self.assertRaisesRegexp(SchemaException,
                                     "Feature types should have a name"):
            self.config._process_named_feature_type(named_ft)

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
        self.assertTrue(isinstance(ft_instance, RegexFeatureTypeInstance))
        self.assertEqual(ft_instance._params, ft['params'])

    def test_process_anonymous_feature_type_unknown_type(self):
        ft = {
            'type': 'zavarakatranemia',
            'params': {'pattern': '(\d\.\d+)'}
        }
        with self.assertRaisesRegexp(SchemaException,
                                     "Feature types should have a name"):
            self.config._process_named_feature_type(ft)

        self.assertEqual(1, len(self.config._named_feature_types))

    def test_process_anonymous_feature_type_missing_params(self):
        ft = {
            'name': 'floating_point',
            'type': 'regex'
        }
        with self.assertRaisesRegexp(SchemaException,
                                     "Cannot create instance of feature type"):
            self.config._process_named_feature_type(ft)

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
        self.assertTrue(isinstance(result['type'],
                                   PrimitiveFeatureTypeInstance))
        self.assertIsNone(result['transformer'])
        self.assertFalse(result['required'])

    def test_process_feature_with_scaler(self):
        from sklearn.preprocessing import MinMaxScaler
        feature = {
            'name': 'another_test_feature',
            'type': 'int',
            'scaler': {
                'type': 'ee'
            }
        }
        with self.assertRaisesRegexp(ScalerException,
                                     "Scaler 'ee' isn't supported."):
            self.config._process_feature(feature)

        feature = {
            'name': 'another_test_feature',
            'type': 'float',
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
            'type': 'text',
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
        with self.assertRaisesRegexp(
                SchemaException,
                "Features should have a name: {'type': 'int'}"):
            self.config._process_feature(feature)

    def test_invalid_data(self):
        with self.assertRaisesRegexp(
                SchemaException, "No JSON object could be decoded"):
            FeatureModel(
                os.path.join(BASEDIR, 'trainer', 'invalid-features.json'))

        with self.assertRaisesRegexp(
                SchemaException, "No JSON object could be decoded"):
            FeatureModel("invalid json", is_file=False)

        with self.assertRaisesRegexp(
                SchemaException, "schema-name is missing"):
            FeatureModel('{"key": 1}', is_file=False)

        with self.assertRaisesRegexp(
                SchemaException, "No target variable defined"):
            FeatureModel(os.path.join(
                BASEDIR, 'trainer', 'features-no-target-feature.json'))

        with self.assertRaisesRegexp(
                SchemaException, "No classifier configuration defined"):
            FeatureModel(os.path.join(
                BASEDIR, 'trainer', 'features-no-classifier.json'))

        with self.assertRaisesRegexp(
                SchemaException,
                "Feature type 'str_to_timezone' already defined"):
            FeatureModel(os.path.join(
                BASEDIR, 'trainer', 'features-type-duplicates.json'))

        with self.assertRaisesRegexp(
                SchemaException, "Feature hire_outcome should have a type"):
            FeatureModel(os.path.join(
                BASEDIR, 'trainer', 'features-no-type.json'))

        with self.assertRaisesRegexp(
                SchemaException, "Type not set on individual feature type"):
            FeatureModel(os.path.join(
                BASEDIR, 'trainer', 'features-feature_type_without-type.json'))
