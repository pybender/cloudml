"""
Unittests for ExtractionPlan and ImportHandler classes.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import os
import csv
import unittest
import json
from datetime import datetime
from mock import patch, Mock, MagicMock
from httmock import HTTMock, urlmatch
from cloudml.tests.test_utils import StreamPill
import boto3

from cloudml.importhandler.importhandler import ExtractionPlan, \
    ImportHandlerException, ImportHandler
from cloudml.importhandler.predict import Predict
from constants import ROW, PARAMS
from cloudml.tests.test_utils import db_row_iter_mock

BASEDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../testdata'))


class PigXMLPlanTest(unittest.TestCase):
    PIG_DS = 'cloudml.importhandler.datasources.PigDataSource'

    def setUp(self):
        super(PigXMLPlanTest, self).setUp()
        self.pill = StreamPill(debug=True)
        self.session = boto3.session.Session()
        boto3.DEFAULT_SESSION = self.session

    @patch('subprocess.Popen')
    @patch('time.sleep', return_value=None)
    def test_pig_datasource(self, sleep_mock, sqoop_mock):
        # Amazon mock
        self.pill.attach(self.session, os.path.abspath(
            os.path.join(os.path.dirname(__file__),
                         'placebo_responses/importhandler/pigxml')))
        self.pill.playback()

        self._plan = ExtractionPlan(os.path.join(
            BASEDIR, 'extractorxml',
            'pig-train-import-handler.xml'))

        # Sqoop import subprocess mock
        process_mock = Mock()
        attrs = {'wait.return_value': 0,
                 'stdout.readlines.return_value': []}
        process_mock.configure_mock(**attrs)
        sqoop_mock.return_value = process_mock

        with patch('psycopg2.extras.DictCursor.execute'):
            with patch('psycopg2.connect'):
                self._extractor = ImportHandler(self._plan, PARAMS)

        pig_ds = self._extractor.plan.datasources['pig']
        # Checking iterator
        row = self._extractor.next()
        self.assertEquals(row['opening_id'], 57)

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

    # def test_http_404(self):
    #     #with HTTMock(http_mock):
    #     self._plan.entity.query = '/does/not/exist.json'
    #     try:
    #         self._extractor = ImportHandler(self._plan, PARAMS)
    #     except ImportHandlerException as exc:
    #         self.assertEqual(exc.message[:16], 'Cannot reach url')


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
        def _check(name, err):
            file_name = os.path.join(BASEDIR, 'extractorxml',
                                     'invalid', name)
            with self.assertRaisesRegexp(ImportHandlerException, err):
                ExtractionPlan(file_name)

        _check("no-entity.xml", "There is an error in the import handler's "
               "XML, line 15.\W+\w+")
        _check("datasource_name.xml",
               "There are few datasources with name odw")

        with self.assertRaisesRegexp(ImportHandlerException,
                                     "import handler file is empty"):
            ExtractionPlan(None, is_file=False)

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
        self._plan_for_script = ExtractionPlan(
            os.path.join(BASEDIR, 'extractorxml',
                         'train-import-handler-script-file.xml'))

    @patch('cloudml.importhandler.datasources.DbDataSource._get_iter',
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

    @patch('cloudml.importhandler.datasources.DbDataSource._get_iter',
           return_value=db_iter_mock())
    def test_imports_script_src(self, mock_db):
        # Checking js functions calls work from <script src=""/>
        self._extractor_script = ImportHandler(self._plan_for_script, PARAMS)
        row = self._extractor_script.next()
        self.assertTrue(mock_db.called)
        self.assertEqual(row['test_script'], 99)
        self.assertEqual(row['test_script_tag'], 99)

    @patch('cloudml.importhandler.datasources.DbDataSource._get_iter',
           return_value=db_iter_mock())
    def test_store_data_json(self, mock_db):
        self._extractor = ImportHandler(self._plan, PARAMS)
        self._extractor.store_data_json("data.json.bak")
        self.assertTrue(os.path.isfile("data.json.bak"))
        with open("data.json.bak") as fp:
            json_data = fp.read()
            data = json.loads(json_data)
            self.assertEquals(data['application_id'], 555)
        os.remove("data.json.bak")

        self._extractor.store_data_json("data.gz.bak", True)
        self.assertTrue(os.path.isfile("data.gz.bak"))
        os.remove("data.gz.bak")

    @patch('cloudml.importhandler.datasources.DbDataSource._get_iter',
           return_value=db_iter_mock())
    def test_store_data_csv(self, mock_db):
        self._extractor = ImportHandler(self._plan, PARAMS)
        self._extractor.store_data_csv("data.csv.bak")
        self.assertTrue(os.path.isfile("data.csv.bak"))
        with open("data.csv.bak") as fp:
            reader = csv.reader(fp)
            rows = [row for row in reader]
            self.assertEquals(len(rows), 2)
        os.remove("data.csv.bak")

    @patch('cloudml.importhandler.datasources.DbDataSource._get_iter',
           return_value=db_iter_mock())
    def test_store_data_csv_compressed(self, mock_db):
        self._extractor = ImportHandler(self._plan, PARAMS)
        self._extractor.store_data_csv("data.gz.bak", True)
        self.assertTrue(os.path.isfile("data.gz.bak"))
        os.remove("data.gz.bak")

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


class CompositeTypeTest(unittest.TestCase):
    def setUp(self):
        self._plan = ExtractionPlan(os.path.join(
                                    BASEDIR,
                                    'extractorxml',
                                    'composite-type-import-handler.xml'))

    @patch('cloudml.importhandler.datasources.DbDataSource._get_iter',
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
