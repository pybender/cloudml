import os
import unittest
from datetime import datetime

from mock import patch
from httmock import HTTMock, urlmatch

from core.xmlimporthandler.importhandler import ExtractionPlan, \
    ImportHandlerException, ImportHandler
from core.xmlimporthandler.scripts import ScriptManager
from core.xmlimporthandler.entities import Field, FieldException
from core.xmlimporthandler.inputs import Input
from constants import ROW, PARAMS

BASEDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../testdata'))


class ScriptManagerTest(unittest.TestCase):
    def test_script(self):
        manager = ScriptManager()
        self.assertEqual(manager._exec('1+2'), 3)

    def test_manager(self):
        manager = ScriptManager()
        manager.add_python("""def intToBoolean(a):
            return a == 1
        """)
        self.assertEqual(manager._exec('intToBoolean(1)'), True)


class CompositeTypeTest(unittest.TestCase):
    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(BASEDIR,
                                    'extractorxml',
                                    'composite-type-import-handler.xml'))

    def readability_test(self):
        self._extractor = ImportHandler(self._plan, {
            'start': '2012-12-03',
            'end': '2012-12-04',
        })
        row = self._extractor.next()
        print row
        self.assertTrue(False)


class TestField(unittest.TestCase):
    def test_field_declaration_validation(self):
        with self.assertRaises(ImportHandlerException):
            field = Field({
                'name': 'field_name',
                'type': 'int'})

    def test_field_required(self):
        field_required = Field({
            'name': 'field_name',
            'type': 'string',
            'required': 'true'
        })
        field = Field({
            'name': 'field_name',
            'type': 'string'
        })
        with self.assertRaises(FieldException):
            field_required.process_value(None, None)
        value = field.process_value(None, None)
        self.assertEqual(value, None)


class TestInput(unittest.TestCase):
    def test_params_validation(self):
    # <!-- Boolean parameter -->
    # <param name="only_fjp" type="boolean" />
        inp = Input(dict(name="application", type="integer", regex="\d+"))
        self.assertEqual(inp.process_value('1'), 1)
        self.assertRaises(ImportHandlerException, inp.process_value, 'str')
        self.assertRaises(ImportHandlerException, inp.process_value, '-1')

        inp = Input(dict(name="created", type="date", format="%A %d. %B %Y"))
        self.assertEqual(inp.process_value('Monday 11. March 2002'),
                         datetime(2002, 3, 11, 0, 0))
        with self.assertRaisesRegexp(
                ImportHandlerException, "Value of the input parameter created \
invalid date in format %A %d. %B %Y: 11/03/02"):
            inp.process_value('11/03/02')
        with self.assertRaisesRegexp(
                ImportHandlerException, "Input parameter created is required"):
            inp.process_value(None)


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
        self._plan = ExtractionPlan(os.path.join(BASEDIR,
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
        self._plan = ExtractionPlan(os.path.join(BASEDIR,
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
            str(the_exception)[:26],
            'XML file format is invalid'
        )

    def test_get_ds_config(self):
        conf = ExtractionPlan.get_datasources_config()
        self.assertEqual(set(['db', 'http', 'pig', 'csv']), set(conf.keys()))


def db_iter_mock(*args, **kwargs):
    for r in [ROW, {'title': 'Application Title'}]:
        yield r


class ImportHandlerTest(unittest.TestCase):
    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(BASEDIR,
                                    'extractorxml',
                                    'train-import-handler.xml'))

    @patch('core.xmlimporthandler.datasources.DbDataSource._get_iter',
           return_value=db_iter_mock())
    def test_imports(self, mock_db):
        self._extractor = ImportHandler(self._plan, PARAMS)
        row = self._extractor.next()
        self.assertTrue(mock_db.called)

        # Checking types
        print row
        self.assertEqual(row['check_float'], float(ROW["float_field"]))
        self.assertEqual(row['check_string'], ROW["float_field"])
        self.assertEqual(row['check_int'], int(ROW["int_field"]))
        self.assertEqual(row['check_boolean'], True)
        self.assertEqual(row['check_integer_with_float'], None)

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
