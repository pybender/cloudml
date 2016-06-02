"""
Unittests for datasources classes.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import unittest
import os
from mock import patch, MagicMock, Mock, ANY
from lxml import objectify
from cloudml.tests.test_utils import StreamPill
import boto3
import json


from cloudml.importhandler.datasources import DataSource, BaseDataSource, \
    DbDataSource, HttpDataSource, CsvDataSource, PigDataSource, \
    InputDataSource
from cloudml.importhandler.exceptions import ImportHandlerException, \
    ProcessException

BASEDIR = os.path.abspath(os.path.dirname(__file__))


class DataSourcesTest(unittest.TestCase):
    DB = objectify.fromstring(
        """<db name="odw"
            host="localhost"
            dbname="odw"
            user="postgres"
            password="postgres"
            vendor="postgres" />""")
    HTTP = objectify.fromstring(
        """<http name="jar" method="GET"
        url="http://upwork.com/jar/" />""")
    CSV = objectify.fromstring(
        """<csv name="csvDataSource" src="%s/stats_header.csv">
            <!-- Note that some columns are ignored -->
            <header name="id" index="0" />
            <header name="name" index="2" />
            <header name="score" index="3" />
        </csv>""" % BASEDIR)
    CSV_WITHOUT_HEADER = objectify.fromstring(
        """<csv name="csvDataSource" src="%s/stats.csv"></csv>""" % BASEDIR)
    PIG = objectify.fromstring("""<pig name="jar"
        amazon_access_token="token"
        amazon_token_secret="secret" bucket_name="mybucket" />""")
    INPUT = objectify.fromstring("""<input name="jar" />""")

    def test_base_datasource(self):
        config = objectify.fromstring(
            """<ds name="odw" p1="val1" p2="val2" />""")
        ds = BaseDataSource(config)
        self.assertEquals(ds.name, 'odw')
        self.assertEquals(ds.type, 'ds')
        self.assertEquals(ds.get_params(), {'p1': 'val1', 'p2': 'val2'})
        self.assertRaises(Exception, ds._get_iter)

    def test_db_datasource(self):
        exec_ = MagicMock()
        cur = Mock()
        cur.__iter__ = Mock(return_value=iter(['result 1', 'result 2']))
        con = Mock()
        con.cursor.return_value = cur
        cur.execute = exec_
        conn_ = MagicMock(return_value=con)

        ds = DataSource.factory(self.DB)
        with patch('psycopg2.connect', conn_):
            query = 'select * from tbl;'
            ds._get_iter(query=query).next()
            exec_.assert_called_with(query)

            exec_.reset_mock()

            query = 'select * from tbl'
            ds._get_iter(query=query).next()
            exec_.assert_called_with(query + ';')

            # query is required
            self.assertRaises(
                ImportHandlerException, ds._get_iter, None)
            self.assertRaises(
                ImportHandlerException, ds._get_iter, ' ')

    def test_db_datasource_invalid_definition(self):
        # Vendor is invalid
        config = objectify.fromstring(
            """<db name="odw"
                host="localhost"
                dbname="odw"
                user="postgres"
                password="postgres"
                vendor="invalid" />""")
        ds = DataSource.factory(config)
        self.assertRaises(ImportHandlerException, ds._get_iter, 'query')

        # Host isn't specified
        config = objectify.fromstring(
            """<db name="odw"
                dbname="odw"
                user="postgres"
                password="postgres"
                vendor="postgres" />""")
        ds = DataSource.factory(config)
        self.assertRaises(ImportHandlerException, ds._get_iter, 'query')

    def test_http_data_source(self):
        mock = MagicMock()
        mock.json.return_value = {"key": "val"}
        with patch('requests.request', mock):
            ds = HttpDataSource(self.HTTP)
            ds._get_iter()
            mock.assert_called_with(
                'GET', 'http://upwork.com/jar/', stream=True)

            mock.reset_mock()

            # query_target isn't supported
            self.assertRaises(
                ImportHandlerException, ds._get_iter, '', 'query_target')

        # url is required
        config = objectify.fromstring(
            """<http name="jar" method="GET" url="" />""")
        self.assertRaises(
            ImportHandlerException, HttpDataSource, config)

        config = objectify.fromstring(
            """<http name="jar" method="GET" />""")
        self.assertRaises(
            ImportHandlerException, HttpDataSource, config)

    def test_csv_datasource(self):
        ds = CsvDataSource(self.CSV)
        self.assertItemsEqual(
            ds.headers, [('id', 0), ('name', 2), ('score', 3)])
        res = ds._get_iter().next()
        self.assertEquals(
            res, {'score': 'score', 'id': 'id', 'name': 'name'})

        ds = CsvDataSource(self.CSV_WITHOUT_HEADER)
        iter_ = ds._get_iter()
        res = iter_.next()
        self.assertEquals(
            res, {'3': 'score1',
                  '0': 'id1',
                  '5': [1, 2, 3],
                  '2': 'name1',
                  '4': {u'key': u'val'},
                  '1': 'type1'})

        res = iter_.next()
        self.assertEquals(
            res, {'2': 'name2',
                  '5': '',
                  '3': 'score2',
                  '4': '{{val}}',
                  '1': 'type2',
                  '0': 'id2'})

        # src is missing
        config = objectify.fromstring(
            """<csv name="jar" method="GET" />""")
        self.assertRaises(
            ImportHandlerException, CsvDataSource, config)

        config = objectify.fromstring(
            """<csv name="csvDataSource" src="%s/stats.csv">
                <!-- Note that some columns are ignored -->
                <header name="id" index="0" />
                <header name="name" index="2" />
                <header name="score" index="10" />
            </csv>""" % BASEDIR)
        ds = CsvDataSource(config)
        iter_ = ds._get_iter()
        self.assertRaises(ImportHandlerException, iter_.next)

    def test_input_datasource(self):
        ds = InputDataSource(self.INPUT)
        ds._get_iter('{"key": "val"}')

    def test_factory(self):
        config = objectify.fromstring("""<invalid />""")
        self.assertRaises(
            ImportHandlerException, DataSource.factory, config)

        config = objectify.fromstring("""<db name="" />""")
        self.assertRaises(
            ImportHandlerException, DataSource.factory, config)

        ds = DataSource.factory(self.DB)
        self.assertEquals(type(ds), DbDataSource)
        self.assertEquals(ds.type, 'db')

        ds = DataSource.factory(self.HTTP)
        self.assertEquals(type(ds), HttpDataSource)
        self.assertEquals(ds.type, 'http')

        ds = DataSource.factory(self.CSV)
        self.assertEquals(type(ds), CsvDataSource)
        self.assertEquals(ds.type, 'csv')

        ds = DataSource.factory(self.PIG)
        self.assertEquals(type(ds), PigDataSource)
        self.assertEquals(ds.type, 'pig')

        ds = DataSource.factory(self.INPUT)
        self.assertEquals(type(ds), InputDataSource)
        self.assertEquals(ds.type, 'input')


def conn_exec_print(cursor, query):
    print "Query is", query


class DbDataSourceTests(unittest.TestCase):

    def setUp(self):
        self.datasource = DataSource.factory(DataSourcesTest.DB)
        self.assertEquals(type(self.datasource), DbDataSource)

    @patch('cloudml.importhandler.db.execute', side_effect=conn_exec_print)
    def test_sql_injection_on_query_target(self, exec_mock):
        query = 'SELECT * FROM pg_catalog.pg_tables'

        iter_ = self.datasource._get_iter(
            query, query_target='target_tbl')

        with self.assertRaises(ValueError):
            self.datasource._get_iter(
                query,
                query_target='target_tbl;delete * from tbl3;')


class PigDataSourceTests(unittest.TestCase):
    PLACEBO_RESPONSES_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     'placebo_responses/datasource/'))

    def setUp(self):
        super(PigDataSourceTests, self).setUp()
        self.pill = StreamPill(debug=True)
        self.session = boto3.session.Session()
        boto3.DEFAULT_SESSION = self.session

    @patch('time.sleep', return_value=None)
    def test_get_iter_existing_job(self, sleep_mock):
        # Amazon mock
        self.pill.attach(self.session,
                         os.path.join(self.PLACEBO_RESPONSES_DIR,
                                      'get_iter_existing_job'))
        self.pill.playback()
        pig_import = 'cloudml.importhandler.datasources.PigDataSource'

        ds = PigDataSource(DataSourcesTest.PIG)

        # test correct case with existing job
        # (DescribeJobFlows_1: 2 steps exist before adding a new one)
        # (DescribeJobFlows_2: RUNNING, RUNNING)
        # (DescribeJobFlows_3: WAITING, COMPLETED)
        ds.jobid = "1234"
        with patch("{}.get_result".format(pig_import), MagicMock()):
            with patch("{}._process_running_state".format(pig_import)) \
                    as run_handler:
                with patch("{}._process_waiting_state".format(pig_import)) \
                        as wait_handler:
                    ds._get_iter('query here', 'query target')
                    # step_number is 3
                    run_handler.assert_called_with(ANY, 'RUNNING', 3)
                    wait_handler.assert_called_with(ANY, 'COMPLETED', 3)

    @patch('time.sleep', return_value=None)
    def test_get_iter_create_job(self, sleep_mock):
        # Amazon mock
        self.pill.attach(self.session,
                         os.path.join(self.PLACEBO_RESPONSES_DIR,
                                      'get_iter_create_job'))
        self.pill.playback()
        pig_import = 'cloudml.importhandler.datasources.PigDataSource'

        ds = PigDataSource(DataSourcesTest.PIG)
        ds.jobid = None
        with patch("{}.get_result".format(pig_import), MagicMock()):
            with patch("{}._process_completed_state".format(pig_import)) as \
                    complete_handler:
                ds._get_iter('query here', 'query target')
                self.assertEqual("234", ds.jobid)
                # step_number is 1
                complete_handler.assert_called_with(ANY, 'COMPLETED', 1)

    @patch('time.sleep', return_value=None)
    def test_get_iter_check_statuses(self, sleep_mock):
        # Amazon mock
        self.pill.attach(self.session,
                         os.path.join(self.PLACEBO_RESPONSES_DIR,
                                      'get_iter_statuses'))
        self.pill.playback()
        pig_import = 'cloudml.importhandler.datasources.PigDataSource'

        ds = PigDataSource(DataSourcesTest.PIG)

        self.assertRaises(ProcessException, ds._get_iter, 'query here')

        _store_query_to_s3 = MagicMock(return_value="s3://bucket/script.jar")
        clear_output_folder = MagicMock()
        _run_steps_on_existing_jobflow = MagicMock(return_value=1)
        get_result = MagicMock()
        _get_log = MagicMock(return_value="Some log")

        ds.jobid = "234"
        with patch("{}._store_query_to_s3".format(pig_import),
                   _store_query_to_s3):
            with patch("{}.clear_output_folder".format(pig_import),
                       clear_output_folder):
                with patch("{}.get_result".format(pig_import), get_result):
                    with patch("{}._run_steps_on_existing_jobflow".format(
                            pig_import, _run_steps_on_existing_jobflow)):
                        with patch("{}._get_log".format(pig_import), _get_log):
                            # test failed case with new job
                            # (DescribeJobFlows_1: FAILED, FAILED)
                            self.assertRaises(ImportHandlerException,
                                              ds._get_iter,
                                              "query here", "query target")

                            # test failed case with new job
                            # (DescribeJobFlows_2: COMPLETED, FAILED)
                            self.assertRaises(ImportHandlerException,
                                              ds._get_iter,
                                              "query here", "query target")

                            # test failed case with new job
                            # (DescribeJobFlows_3: WAITING, FAILED)
                            self.assertRaises(ImportHandlerException,
                                              ds._get_iter,
                                              "query here", "query target")

                            # unexpected status check
                            # (DescribeJobFlows_4: COMPLETED, UNEXPECTED)
                            with patch("{}._process_completed_state".format(
                                    pig_import)) as complete_handler:
                                ds._get_iter('query here', 'query target')
                                complete_handler.assert_called_with(
                                    ANY, 'UNEXPECTED', 1)

                            # unexpected and completed status check
                            # (DescribeJobFlows_5: UNEXPECTED, UNEXPECTED)
                            # (DescribeJobFlows_6: WAITING, PENDING)
                            # (DescribeJobFlows_7: COMPLETED, COMPLETED)
                            with patch("{}._process_waiting_state".format(
                                    pig_import)) as waiting_handler:
                                with patch("{}._process_completed_state".
                                           format(pig_import)) as \
                                        complete_handler:
                                    ds._get_iter('query here', 'query target')
                                    waiting_handler.assert_called_with(
                                        ANY, 'PENDING', 1)
                                    complete_handler.assert_called_with(
                                        ANY, 'COMPLETED', 1)

                            # running and completed status check
                            # (DescribeJobFlows_8: RUNNING, RUNNING)
                            # (DescribeJobFlows_9: WAITING, COMPLETED)
                            with patch("{}._process_running_state".format(
                                    pig_import)) as run_handler:
                                with patch("{}._process_waiting_state".format(
                                        pig_import)) as wait_handler:
                                    ds._get_iter('query here', 'query target')
                                    run_handler.assert_called_with(
                                        ANY, 'RUNNING', 1)
                                    wait_handler.assert_called_with(
                                        ANY, 'COMPLETED', 1)

                            # DescribeJobFlows_10 - corrupted response
                            # (no ExecutionStatusDetail)
                            self.assertRaises(ImportHandlerException,
                                              ds._get_iter,
                                              "query here", "query target")

                            # DescribeJobFlows_11 - corrupted response
                            # (no State)
                            self.assertRaises(ImportHandlerException,
                                              ds._get_iter,
                                              "query here", "query target")

    def test_generate_download_url(self):
        # Amazon mock
        self.pill.attach(self.session,
                         os.path.join(self.PLACEBO_RESPONSES_DIR,
                                      'download_url'))
        self.pill.playback()

        ds = PigDataSource(DataSourcesTest.PIG)
        url = ds.generate_download_url(step=0, log_type='stdout')
        self.assertTrue(url)

    def test_get_pig_step(self):
        # Amazon mock
        self.pill.attach(self.session,
                         os.path.join(self.PLACEBO_RESPONSES_DIR,
                                      'get_pig_step'))
        self.pill.playback()

        ds = PigDataSource(DataSourcesTest.PIG)
        pig_step = ds.get_pig_step('query')
        self.assertTrue(pig_step)

    def test_get_result_job(self):
        response = open(os.path.join(
            self.PLACEBO_RESPONSES_DIR,
            'get_iter_existing_job/elasticmapreduce.DescribeJobFlows_1.json'),
                        'r').read()
        res = json.loads(response)

        ds = PigDataSource(DataSourcesTest.PIG)

        # job has been found
        job = ds._get_result_job(res['data'], "1234")
        self.assertTrue(job)
        self.assertEqual("1234", job["JobFlowId"])

        # no job with this id
        self.assertRaises(ImportHandlerException,
                          ds._get_result_job, res['data'], "1235")

        # error response
        self.assertRaises(ImportHandlerException,
                          ds._get_result_job, {"Error": "error"}, "1234")
