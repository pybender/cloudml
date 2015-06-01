import os
import unittest
import json
from datetime import datetime

from mock import patch
from httmock import HTTMock, urlmatch

from core.importhandler.importhandler import ExtractionPlan, \
    ImportHandlerException, ImportHandler
from core.importhandler.entities import Field, FieldException
from core.importhandler.predict import Predict
from constants import ROW, PARAMS

BASEDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../testdata'))


class TestField(unittest.TestCase):
    def test_field_declaration_validation(self):
        with self.assertRaises(ImportHandlerException):
            field = Field({
                'name': 'field_name',
                'type': 'int'}, entity=None)

    def test_field_required(self):
        field_required = Field({
            'name': 'field_name',
            'type': 'string',
            'required': 'true'
        }, entity=None)
        field = Field({
            'name': 'field_name',
            'type': 'string'
        }, entity=None)
        with self.assertRaises(FieldException):
            field_required.process_value(None, None)
        value = field.process_value(None, None)
        self.assertEqual(value, None)


# class PigXMLPlanTest(unittest.TestCase):
#     def setUp(self):
#         self._plan = ExtractionPlan(os.path.join(BASEDIR,
#                                     'extractorxml',
#                                     'pig-train-import-handler.xml'))
#     def test_pig_datasource(self):
#         self._extractor = ImportHandler(self._plan, PARAMS)
#         row = self._extractor.next()


@urlmatch(netloc='test.odesk.com:11000')
def http_mock(url, request):
    if url.path == '/opening/f/something.json':
        return '[{"application": 123456}]'
    elif url.path == '/some/other/path.json':
        return '[{"application": 78910}]'
    return None


class HttpXMLPlanTest(unittest.TestCase):
    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(
                                    BASEDIR,
                                    'extractorxml',
                                    'http-train-import-handler.xml'))

    def test_http_datasource(self):
        with HTTMock(http_mock):
            self._extractor = ImportHandler(self._plan, PARAMS)
            row = self._extractor.next()
            self.assertEqual(row['application_id'], 123456)

    def test_http_query(self):
        with HTTMock(http_mock):
            self._plan.entity.query = '/some/other/path.json'
            self._extractor = ImportHandler(self._plan, PARAMS)
            row = self._extractor.next()
            self.assertEqual(row['application_id'], 78910)

    def test_http_404(self):
        with HTTMock(http_mock):
            self._plan.entity.query = '/does/not/exist.json'
            try:
                self._extractor = ImportHandler(self._plan, PARAMS)
            except ImportHandlerException as exc:
                self.assertEqual(exc.message[:16], 'Cannot reach url')


class CsvXMLPlanTest(unittest.TestCase):
    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(
                                    BASEDIR,
                                    'extractorxml',
                                    'csv-train-import-handler.xml'))

    def test_csv_datasource(self):
        self._extractor = ImportHandler(self._plan, PARAMS)
        row = self._extractor.next()
        self.assertEqual(row['class'], 'hire')
        self.assertEqual(row['money'], 10)


class ExtractionXMLPlanTest(unittest.TestCase):

    def setUp(self):
        self.generic_importhandler_file = os.path.join(
            BASEDIR, 'extractorxml', 'generic-import-handler.xml')
        self.importhandler_file = os.path.join(
            BASEDIR, 'extractorxml', 'train-import-handler.xml')
        self.incorrect_importhandler_file = os.path.join(
            BASEDIR, 'extractorxml', 'incorrect-import-handler.xml')

    def test_load_valid_plan(self):
        ExtractionPlan(self.importhandler_file)

    def test_load_valid_generic_plan(self):
        ExtractionPlan(self.generic_importhandler_file)

    def test_load_plan_with_syntax_error(self):
        with open(self.importhandler_file, 'r') as fp:
            data = fp.read()
        data = '"' + data
        with self.assertRaises(ImportHandlerException):
            ExtractionPlan(data, is_file=False)

    def test_load_plan_with_schema_error(self):
        with self.assertRaises(ImportHandlerException) as cm:
            ExtractionPlan(self.incorrect_importhandler_file)
        the_exception = cm.exception
        self.assertEqual(
            str(the_exception)[:17],
            'There is an error'
        )

    def test_get_ds_config(self):
        conf = ExtractionPlan.get_datasources_config()
        self.assertEqual(set(['db', 'http', 'pig', 'csv']), set(conf.keys()))


def db_iter_mock(*args, **kwargs):
    for r in [ROW, {'title': 'Application Title'}]:
        yield r


class ImportHandlerTest(unittest.TestCase):
    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(
                                    BASEDIR,
                                    'extractorxml',
                                    'train-import-handler.xml'))

    @patch('core.importhandler.datasources.DbDataSource._get_iter',
           return_value=db_iter_mock())
    def test_imports(self, mock_db):
        self._extractor = ImportHandler(self._plan, PARAMS)
        row = self._extractor.next()
        self.assertTrue(mock_db.called)

        # Checking types
        self.assertEqual(row['check_float'], float(ROW["float_field"]))
        self.assertEqual(row['check_string'], ROW["float_field"])
        self.assertEqual(row['check_int'], int(ROW["int_field"]))
        self.assertEqual(row['check_boolean'], True)
        self.assertEqual(row['check_integer_with_float'], None)
        self.assertEqual(row['check_json'], ROW["json_field"])
        self.assertEqual(row['check_json_jsonpath'], "Professional and \
experienced person")

        # Checking subentries as json datasources
        self.assertEqual(row['employer.country'], 'Philippines')

        # Checking jsonpath and join
        self.assertEqual(row['autors'], 'Nigel and Evelyn')

        # Checking regex and split
        self.assertEqual(row['say_hello'], 'hello')
        self.assertEqual(row['words'], ['Words', 'words', 'words'])

        # Checking javascript func
        self.assertEqual(row['test_script'], 99)
        self.assertEqual(row['test_script_tag'], 99)

        # Checking dataFormat
        self.assertEqual(row['date'], datetime(2014, 6, 1, 13, 33))

        # Checking template
        self.assertEqual(
            row['template'],
            "Greatings: hello and hi and pruvit.")

        # Checking global nested datasources
        self.assertEqual(row['application_title'], 'Application Title')
        self.assertEqual(
            mock_db.call_args_list[1][0][0],
            "SELECT title FROM applications where id==%s;" %
            ROW['application'])

    def test_validate_input_params(self):
        self._extractor = ImportHandler(self._plan, PARAMS)
        with self.assertRaisesRegexp(
                ImportHandlerException, "Missing input parameters"):
            self._extractor.process_input_params({'end': '2013-01-30'})

        with self.assertRaisesRegexp(
                ImportHandlerException, "Missing input parameters"):
            self._extractor.process_input_params({})

        with self.assertRaisesRegexp(
                ImportHandlerException, "Missing input parameters"):
            self._extractor.process_input_params(None)


class PredictTest(unittest.TestCase):
    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(
                                    BASEDIR,
                                    'extractorxml',
                                    'generic-import-handler.xml'))

    def test_predict(self):
        self.assertTrue(isinstance(self._plan.predict, Predict))


def db_row_iter_mock(*args, **kwargs):
    path = os.path.join(BASEDIR, 'extractorxml', 'out.json')
    with open(path, 'r') as fp:
        data = json.loads(fp.read())
    for r in [data]:
        yield r


class CompositeTypeTest(unittest.TestCase):
    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(
                                    BASEDIR,
                                    'extractorxml',
                                    'composite-type-import-handler.xml'))

    @patch('core.importhandler.datasources.DbDataSource._get_iter',
           return_value=db_row_iter_mock())
    def composite_test(self, mock_db):
        self._extractor = ImportHandler(self._plan, {
            'start': '2012-12-03',
            'end': '2012-12-04',
        })
        row = self._extractor.next()
        self.assertEqual(row['country_pair'], 'Australia,Philippines')
        self.assertEqual(
            row['tsexams']['English Spelling Test (U.S. Version)'], 5)


class InputDatasourceTest(unittest.TestCase):

    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(
                                    BASEDIR,
                                    'extractorxml',
                                    'input-datasource-handler.xml'))

    def test_json(self):
        self._extractor = ImportHandler(self._plan, {
            'contractor_info': '{ "skills":[{"skl_status":"0","ts_tests_count"\
:"0","skl_name":"microsoft-excel","skl_external_link":"http:\/\/en.wikipedia.\
org\/wiki\/Microsoft_Excel","skl_has_tests":"1","skl_pretty_name":"Microsoft\
 Excel","skill_uid":"475721704063008779","skl_rank":"1","skl_description":\
 "Microsoft Excel is a proprietary commercial spreadsheet application written\
 and distributed by Microsoft for Microsoft Windows and Mac OS X. It features\
 calculation, graphing tools, pivot tables, and a macro programming language\
 called Visual Basic for Applications."},{"skl_status":"0","ts_tests_count":\
 "0","skl_name":"microsoft-word","skl_external_link":"http:\/\/en.wikipedia.\
 org\/wiki\/Microsoft_Word","skl_has_tests":"1","skl_pretty_name":"Microsoft\
  Word","skill_uid":"475721704071397377","skl_rank":"2","skl_description":\
  "Microsoft Office Word is a word processor designed by Microsoft."}]}',
        })
        row = self._extractor.next()
        self.assertEqual(row['contractor.skills'],
                         'microsoft-excel,microsoft-word')
