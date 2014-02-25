import unittest
import os


from core.xmlimporthandler.importhandler import ExtractionPlan, \
    ImportHandlerException, ImportHandler

BASEDIR = '../../testdata'


class ExtractionXMLPlanTest(unittest.TestCase):

    def setUp(self):
        self.importhandler_file = os.path.join(BASEDIR,
                                           'extractorxml',
                                           'train-import-handler.xml')

    def test_load_valid_plan(self):
        plan = ExtractionPlan(self.importhandler_file)
        # self.assertEqual(plan.schema_name, 'bestmatch')
        # self.assertEqual(1, len(plan.datasource))
        # self.assertEqual('odw', plan.datasource[0]['name'])
        # user_params = ['start', 'end']
        # self.assertEqual(user_params, plan.input_params)

    def test_load_plan_with_syntax_error(self):
        with open(self.importhandler_file, 'r') as fp:
            data = fp.read()
        data = '"' + data
        with self.assertRaises(ImportHandlerException):
            ExtractionPlan(data, is_file=False)

    # def test_load_plan_with_no_datasource(self):
    #     try:
    #         ExtractionPlan(os.path.join(BASEDIR,
    #                                     'extractor',
    #                                     'train-config-no-datasource.json'))
    #         self.fail('Should not be able to create plan with no datasource')
    #     except ImportHandlerException:
    #         # Should happen
    #         pass

    # def test_load_plan_with_no_schema_name(self):
    #     try:
    #         ExtractionPlan(os.path.join(BASEDIR,
    #                        'extractor',
    #                        'train-config-no-queries.json'))
    #         self.fail('Should not be able to create plan with no queries')
    #     except ImportHandlerException:
    #         # Should happen
    #         pass

class ExtractorTest(unittest.TestCase):
    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(BASEDIR,
                                    'extractorxml',
                                    'train-import-handler.xml'))

    def test_imports(self):
        self._extractor = ImportHandler(self._plan,
                                            {'start': '2012-12-03',
                                             'end': '2012-12-04' })
        print self._extractor.next()
        raise Exception('raise')

    def test_validate_input_params(self):
        try:
            self._extractor = ImportHandler(self._plan,
                                            {'start': '2013-01-27',
                                             'end': '2013-01-30'})
            # Test that all required params are provided.
            self._extractor.process_input_params({'start': '2013-01-27',
                                             'end': '2013-01-30'})
        except ImportHandlerException:
            #Should not happen
            self.fail('Should not raise exception when all params are given')

        try:
            # Test when missing params.
            self._extractor.process_input_params({'end': '2013-01-30'})
            self.fail('Should raise exception when param is mising')
        except ImportHandlerException:
            #Should happen
            pass

        try:
            # Test when no params provided.
            self._extractor.process_input_params({})
            self.fail('Should raise exception when param is mising')
        except ImportHandlerException:
            #Should happen
            pass

        try:
            # Test when no params provided.
            self._extractor.process_input_params(None)
            self.fail('Should raise exception when param is mising')
        except ImportHandlerException:
            #Should happen
            pass
