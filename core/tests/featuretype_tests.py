__author__ = 'ifoukarakis'

"""
Created on Jan 24, 2013

@author: ifoukarakis

"""
import unittest
from trainer.featuretype import FEATURE_TYPE_FACTORIES, regex_parse,\
    InvalidFeatureTypeException


class FeatureTypeTest(unittest.TestCase):
    def test_get_instance_known_factory_no_params(self):
        factory = FEATURE_TYPE_FACTORIES['int']
        ft_instance = factory.get_instance(None)
        self.assertEqual(ft_instance._strategy, int)
        self.assertIsNone(ft_instance._params)
        result = ft_instance.transform('42')
        self.assertEqual(42, result)

        # Do the same but provide an empty directory as params
        ft_instance = factory.get_instance({})
        self.assertEqual(ft_instance._strategy, int)
        self.assertEqual(ft_instance._params, {})
        result = ft_instance.transform('42')
        self.assertEqual(42, result)

    def test_get_instance_known_factory_with_params(self):
        factory = FEATURE_TYPE_FACTORIES['regex']
        params = {'pattern': '(\d*\.\d+)', 'should': 'ignore'}
        ft_instance = factory.get_instance(params)
        self.assertEqual(ft_instance._strategy, regex_parse)
        self.assertEqual(ft_instance._params, params)
        result = ft_instance.transform('This is a test. Price is 4.99$')
        self.assertEqual('4.99', result)

    def test_get_instance_known_factory_no_params(self):
        factory = FEATURE_TYPE_FACTORIES['regex']
        try:
            ft_instance = factory.get_instance(None)
            self.fail('Shouldn\'t be able to create a factory type instance')
        except InvalidFeatureTypeException:
            # Should happen
            pass

    def test_get_instance_known_factory_empty_params(self):
        factory = FEATURE_TYPE_FACTORIES['regex']
        try:
            ft_instance = factory.get_instance({})
            self.fail('Shouldn\'t be able to create a factory type instance')
        except InvalidFeatureTypeException:
            # Should happen
            pass

    def test_get_instance_known_factory_invalid_params(self):
        factory = FEATURE_TYPE_FACTORIES['regex']
        try:
            ft_instance = factory.get_instance({'invalid': 'param'})
            self.fail('Shouldn\'t be able to create a factory type instance')
        except InvalidFeatureTypeException:
            # Should happen
            pass

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
        params = {
            'should': 'ignore'
        }
        factory = FEATURE_TYPE_FACTORIES['composite']
        try:
            factory.get_instance(params)
        except InvalidFeatureTypeException:
            # Should happen
            pass

    def test_get_composite_instance_invalid_sub_feature(self):
        params = {
            'chain': [
                {'type': 'regex'},
                {'type': 'float'}
            ]
        }
        factory = FEATURE_TYPE_FACTORIES['composite']
        try:
            factory.get_instance(params)
        except InvalidFeatureTypeException:
            # Should happen
            pass

if __name__ == '__main__':
    unittest.main()
