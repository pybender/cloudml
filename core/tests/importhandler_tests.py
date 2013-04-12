__author__ = 'ifoukarakis'

"""
Created on Jan 27, 2013

@author: ifoukarakis

"""

import unittest
import os


from core.importhandler.importhandler import ExtractionPlan,\
    ImportHandlerException, ImportHandler

BASEDIR = 'testdata'


class ExtractionPlanTest(unittest.TestCase):

    def test_load_valid_plan(self):
        plan = ExtractionPlan(os.path.join(BASEDIR,
                                           'extractor',
                                           'train-config.json'))
        self.assertEqual(plan.schema_name, 'bestmatch')
        self.assertEqual(1, len(plan.datasource))
        self.assertEqual('odw', plan.datasource[0]['name'])
        user_params = ['start', 'end']
        self.assertEqual(user_params, plan.input_params)

    def test_load_plan_with_no_schema_name(self):
        try:
            ExtractionPlan(os.path.join(BASEDIR,
                           'extractor',
                           'train-config-no-schema.json'))
            self.fail('Should not be able to create plan with no schema')
        except ImportHandlerException:
            # Should happen
            pass

    def test_load_plan_with_no_datasource(self):
        try:
            ExtractionPlan(os.path.join(BASEDIR,
                                        'extractor',
                                        'train-config-no-datasource.json'))
            self.fail('Should not be able to create plan with no datasource')
        except ImportHandlerException:
            # Should happen
            pass

    def test_load_plan_with_no_schema_name(self):
        try:
            ExtractionPlan(os.path.join(BASEDIR,
                           'extractor',
                           'train-config-no-queries.json'))
            self.fail('Should not be able to create plan with no queries')
        except ImportHandlerException:
            # Should happen
            pass


class ExtractorTest(unittest.TestCase):
    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(BASEDIR,
                                    'extractor',
                                    'train-config.json'))

    def test_validate_input_params(self):
        try:
            self._extractor = ImportHandler(self._plan, {'start': '2013-01-27',
                                                         'end': '2013-01-30'})
            # Test that all required params are provided.
            self._extractor._validate_input_params({'start': '2013-01-27',
                                                    'end': '2013-01-30'})
            # Test that all required params are provided, and more that
            # will be ignored are provided as well.
            self._extractor._validate_input_params({'start': '2013-01-27',
                                                    'end': '2013-01-30',
                                                    'should': 'ignore'})
        except ImportHandlerException:
            #Should not happen
            self.fail('Should not raise exception when all params are given')

        try:
            # Test when missing params.
            self._extractor._validate_input_params({'end': '2013-01-30'})
            self.fail('Should raise exception when param is mising')
        except ImportHandlerException:
            #Should happen
            pass

        try:
            # Test when no params provided.
            self._extractor._validate_input_params({})
            self.fail('Should raise exception when param is mising')
        except ImportHandlerException:
            #Should happen
            pass

        try:
            # Test when no params provided.
            self._extractor._validate_input_params(None)
            self.fail('Should raise exception when param is mising')
        except ImportHandlerException:
            #Should happen
            pass


if __name__ == '__main__':
    unittest.main()
