# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>

import unittest
from cloudml.trainer.feature_types import FEATURE_TYPE_FACTORIES, \
    InvalidFeatureTypeException, RegexFeatureTypeInstance


class FeatureTypeTest(unittest.TestCase):
    def test_get_instance_known_factory_no_params(self):
        factory = FEATURE_TYPE_FACTORIES['int']
        ft_instance = factory.get_instance(None)
        self.assertEqual(ft_instance.python_type, 'int')
        self.assertIsNone(ft_instance._params)
        result = ft_instance.transform('42')
        self.assertEqual(42, result)

        # Do the same but provide an empty directory as params
        ft_instance = factory.get_instance({})
        self.assertEqual(ft_instance.python_type, 'int')
        self.assertEqual(ft_instance._params, {})
        result = ft_instance.transform('42')
        self.assertEqual(42, result)

    def test_get_instance_known_factory_with_params(self):
        factory = FEATURE_TYPE_FACTORIES['regex']
        params = {'pattern': '(\d*\.\d+)', 'should': 'ignore'}
        ft_instance = factory.get_instance(params)
        self.assertTrue(isinstance(ft_instance, RegexFeatureTypeInstance))
        self.assertEqual(ft_instance._params, params)
        result = ft_instance.transform('This is a test. Price is 4.99$')
        self.assertEqual('4.99', result)

        result = ft_instance.transform('This is a test.')
        self.assertEqual(None, result)

        factory = FEATURE_TYPE_FACTORIES['int']
        ft_instance = factory.get_instance({'default': 100})
        self.assertEqual(ft_instance.python_type, 'int')
        self.assertEqual(ft_instance._params, {'default': 100})
        result = ft_instance.transform('a')
        self.assertEqual(100, result)

    def test_get_instance_params_validation(self):
        factory = FEATURE_TYPE_FACTORIES['regex']
        with self.assertRaisesRegexp(InvalidFeatureTypeException,
                                     "Not all required parameters set"):
            ft_instance = factory.get_instance(None)

        factory = FEATURE_TYPE_FACTORIES['regex']
        with self.assertRaisesRegexp(InvalidFeatureTypeException,
                                     "Not all required parameters set"):
            ft_instance = factory.get_instance({})

        factory = FEATURE_TYPE_FACTORIES['regex']
        with self.assertRaisesRegexp(InvalidFeatureTypeException,
                                     "Not all required parameters set"):
            ft_instance = factory.get_instance({'invalid': 'param'})

    def test_get_composite_instance(self):
        params = {
            'chain': [
                {'type': 'regex',
                 'params': {'pattern': '(\d*\.\d+)'}},
                {'type': 'float'}
            ],
            'should': 'ignore'
        }
        factory = FEATURE_TYPE_FACTORIES['composite']
        ft_instance = factory.get_instance(params)
        result = ft_instance.transform('This is a test. Price is 4.99$')
        self.assertEqual(4.99, result)

    def test_get_composite_instance_no_chain(self):
        factory = FEATURE_TYPE_FACTORIES['composite']
        with self.assertRaisesRegexp(InvalidFeatureTypeException,
                                     "Composite feature types should "
                                     "define property \"chain\""):
            factory.get_instance({'should': 'ignore'})

    def test_get_composite_instance_invalid_sub_feature(self):
        params = {
            'chain': [
                {'type': 'regex'},
                {'type': 'float'}
            ]
        }
        factory = FEATURE_TYPE_FACTORIES['composite']
        with self.assertRaisesRegexp(InvalidFeatureTypeException,
                                     "Not all required parameters set"):
            factory.get_instance(params)

    def test_get_composite_instance_invalid_ftype(self):
        factory = FEATURE_TYPE_FACTORIES['composite']

        params = {
            'chain': [
                {'type': 'tb',
                 'params': {'pattern': '(\d*\.\d+)'}}
            ]
        }
        with self.assertRaisesRegexp(InvalidFeatureTypeException,
                                     "Unknown type: tb"):
            ft_instance = factory.get_instance(params)

        params = {
            'chain': [
                {'params': {'pattern': '(\d*\.\d+)'}}
            ]
        }
        with self.assertRaisesRegexp(InvalidFeatureTypeException,
                                     'Type not set on individual '
                                     'feature type'):
            ft_instance = factory.get_instance(params)

    def test_date(self):
        factory = FEATURE_TYPE_FACTORIES['date']
        ft_instance = factory.get_instance({'pattern': 'invalid pattern'})
        result = ft_instance.transform('2012/12/12')
        self.assertEqual(946684800, result)  # default date

        factory = FEATURE_TYPE_FACTORIES['date']
        ft_instance = factory.get_instance({'pattern': "%d/%m/%y %H:%M"})
        result = ft_instance.transform("21/11/06 16:30")
        self.assertEqual(1164126600, result)

        result = ft_instance.transform("invalid date")
        self.assertEqual(946684800, result)  # default date

        result = ft_instance.transform(None)
        self.assertEqual(946684800, result)  # default date

        result = ft_instance.transform(50.5)
        self.assertEqual(946684800, result)  # default date

        factory = FEATURE_TYPE_FACTORIES['date']
        with self.assertRaisesRegexp(InvalidFeatureTypeException,
                                     "Not all required parameters set"):
            factory.get_instance({})

    def test_ordinal(self):
        factory = FEATURE_TYPE_FACTORIES['map']
        with self.assertRaisesRegexp(InvalidFeatureTypeException,
                                     "Not all required parameters set"):
            ft_instance = factory.get_instance({'should': 'ignore'})

        ft_instance = factory.get_instance({
            "mappings": {
                "hire": 1,
                "nohire": 0
            }
        })
        result = ft_instance.transform('non exist in mappings value')
        self.assertEqual(None, result)

        result = ft_instance.transform('hire')
        self.assertEqual(1, result)

    def test_categorical(self):
        factory = FEATURE_TYPE_FACTORIES['categorical']
        ft_instance = factory.get_instance({"split_pattern": '\\s*,\\s*'})
        result = ft_instance.preprocessor.fit_transform(['a,b,c', 'a,b'])
        self.assertEqual(
            result.todense().tolist(), [[1L, 1L, 1L], [1L, 1L, 0L]])
        result = ft_instance.transform('a,b,c')
        self.assertEqual(result, 'a,b,c')
        ft_instance = factory.get_instance({})
        result = ft_instance.transform('a,b,c')
        self.assertEqual(result, 'a,b,c')
