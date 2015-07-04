"""
Unittests for datasources classes.
"""

# Author: Nikolay Melnik <nmelnik@upwork.com>

import unittest
import os
from moto import mock_s3, mock_emr
from mock import patch, MagicMock
from lxml import objectify

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
        next_ = MagicMock(return_value=1)
        ds = DataSource.factory(self.DB)
        with patch('psycopg2.extras.DictCursor.execute', exec_):
            with patch('psycopg2.extras.DictCursor.next', next_):
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


class PigDataSourceTests(unittest.TestCase):

    @patch('time.sleep', return_value=None)
    def test_get_iter(self, sleep_mock):
        ds = PigDataSource(DataSourcesTest.PIG)
        self.assertRaises(ProcessException, ds._get_iter, 'query here')

        get_pig_step = MagicMock()
        clear_output_folder = MagicMock()
        _create_jobflow_and_run_steps = MagicMock()

        def _get_status_mock(status_state, state):
            item_mock = MagicMock()
            item_mock.state = status_state
            status_mock = MagicMock()
            status_mock.state = state
            status_mock.steps = [item_mock, item_mock, item_mock]
            return status_mock

        status_mocks = [_get_status_mock('RUNNING', 'RUNNING'),
                        _get_status_mock('WAITING', 'WAITING'),
                        _get_status_mock('COMPLETED', 'COMPLETED')]
        describe_jobflow = MagicMock(side_effect=status_mocks)
        pig_import = 'cloudml.importhandler.datasources.PigDataSource'
        with patch('{}.get_pig_step'.format(pig_import), get_pig_step):
            with patch('{}.clear_output_folder'.format(pig_import),
                       clear_output_folder):
                with patch("boto.emr.connection.EmrConnection."
                           "describe_jobflow", describe_jobflow):
                    # create new job flow.
                    with patch('{}._create_jobflow_'
                               'and_run_steps'.format(pig_import),
                               _create_jobflow_and_run_steps):
                        iter_ = ds._get_iter('query here', 'query target')

    @mock_s3
    def test_generate_download_url(self):
        ds = PigDataSource(DataSourcesTest.PIG)
        bucket = ds.s3_conn.create_bucket('mybucket')

        url = ds.generate_download_url(step=0, log_type='stdout')
        self.assertTrue(url)

    @mock_s3
    @mock_emr
    def test_get_pig_step(self):
        ds = PigDataSource(DataSourcesTest.PIG)
        bucket = ds.s3_conn.create_bucket('mybucket')

        pig_step = ds.get_pig_step('query')
        self.assertTrue(pig_step)
