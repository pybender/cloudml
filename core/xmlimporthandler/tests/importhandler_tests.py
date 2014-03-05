import unittest
import os
from mock import patch
from datetime import datetime

from core.xmlimporthandler.importhandler import ExtractionPlan, \
    ImportHandlerException, ImportHandler
from core.xmlimporthandler.scripts import ScriptManager
from core.xmlimporthandler.entities import Field
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
        manager.add_js("""function intToBoolean(a) {
            return a == 1;
        }""")
        self.assertEqual(manager._exec('intToBoolean(1)'), True)


class TestField(unittest.TestCase):
    def test_field_declaration_validation(self):
        with self.assertRaises(ImportHandlerException):
            field = Field({
                'name': 'field_name',
                'type': 'int'})


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
        self.assertEqual(row['test_js'], 99)

        # Checking dataFormat
        self.assertEqual(row['date'], datetime(2014, 6, 1, 13, 33))

        # Checking template
        self.assertEqual(
            row['template'],
            "Greatings: hello and hi and pruvit.")

        # Checking global nested datasources
        self.assertEqual(row['application_title'], 'Application Title')
        self.assertEqual(
            mock_db.call_args_list[1][0][0][0],
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
