import logging
import time
import itertools
import json

import boto
from boto.s3.key import Key
from exceptions import ImportHandlerException
from core.importhandler.db import postgres_iter

logging.getLogger('boto').setLevel(logging.INFO)


class BaseDataSource(object):
    """
    Base class for any type of the datasource.
    """
    DB_ITERS = {
        'postgres': postgres_iter
    }

    def __init__(self, config):
        self.config = config
        self.name = config.get('name')  # unique
        self.type = config.tag

    def _get_iter(self, query=None, query_target=None):
        raise Exception('Not implemented')

    def get_params(self):
        return {}


class DbDataSource(BaseDataSource):
    """
    Database connection.
    """
    def get_params(self):
        res = {}
        for key, val in self.config[0].attrib.iteritems():
            if key not in ['name', ]:
                res[key] = val
        return res

    def _get_iter(self, query, query_target=None):
        query = [query]
        if query_target:
            query.append("SELECT * FROM %s;" % query_target)
        db_iter = self.DB_ITERS.get(self.config[0].attrib['vendor'])

        if db_iter is None:
            raise ImportHandlerException(
                'Database type %s not supported' % self.config['db']['vendor'])

        if 'host' not in self.config[0].attrib:
            raise ImportHandlerException(
                'No database connection details defined')

        conn_params = self.config[0].attrib
        conn_params.pop('name')
        conn_params.pop('vendor')
        conn_string = ' '.join(['%s=%s' % (k, v)
                                for k, v in conn_params.iteritems()])
        return db_iter(query, conn_string)


class HttpDataSource(BaseDataSource):
    pass


class PigDataSource(BaseDataSource):
    S3_LOG_URI = '/cloudml/logs'
    AMAZON_ACCESS_TOKEN = 'AKIAJ3WMYTNKB77YZ5KQ'
    AMAZON_TOKEN_SECRET = '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'
    BUCKET_NAME = 'odesk-match-prod'
    PIG_VERSIONS = '0.11.1'

    def __init__(self, config):
        super(PigDataSource, self).__init__(config)
        self.s3_conn = boto.connect_s3(self.AMAZON_ACCESS_TOKEN,
                                       self.AMAZON_TOKEN_SECRET)

    def store_query_to_s3(self, query, query_target=None):
        # substitute your bucket name here
        b = self.s3_conn.get_bucket(self.BUCKET_NAME)
        k = Key(b)
        k.key = 'cloudml/pig/' + self.name + '_script.pig'
        k.set_contents_from_string(query)
        return 's3://%s/%s' % (self.BUCKET_NAME, k.key)

    def get_result(self, name):
        b = self.s3_conn.get_bucket(self.BUCKET_NAME)
        k = Key(b)
        k.key = "cloudml/output/%s/part-r-00000" % name
        return k.get_contents_as_string()

    def get_log(self, log_uri, jobid, log_type='stdout'):
        b = self.s3_conn.get_bucket(self.BUCKET_NAME)
        k = Key(b)
        k.key = "%s/%s/steps/2/%s" % (log_uri, jobid, log_type)
        return k.get_contents_as_string()

    def _get_iter(self, query, query_target=None):
        import boto.emr
        from boto.emr.step import PigStep, InstallPigStep

        conn = boto.emr.connect_to_region(
            'us-west-2',
            aws_access_key_id=self.AMAZON_ACCESS_TOKEN,
            aws_secret_access_key=self.AMAZON_TOKEN_SECRET)
        pig_file = self.store_query_to_s3(query)
        log_path = '%s/%s' % (self.S3_LOG_URI, self.name)
        log_uri = 's3://%s%s' % (self.BUCKET_NAME, log_path)
        result_uri = "s3://%s/cloudml/output/%s" % (self.BUCKET_NAME, self.name)
        install_pig_step = InstallPigStep(pig_versions=self.PIG_VERSIONS)
        pig_step = PigStep(self.name,
                     pig_file=pig_file,
                     pig_versions=self.PIG_VERSIONS,
                     pig_args=['-p output=%s' % result_uri])
        logging.info('Run emr jobflow')
        jobid = conn.run_jobflow(name='Cloudml jobflow',
                          log_uri=log_uri,
                          ami_version='2.2',
                          steps=[install_pig_step, pig_step])
        logging.info('JobFlowid: %s' % jobid)
        previous_state = None
        while True:
            time.sleep(10)
            status = conn.describe_jobflow(jobid)

            if previous_state != status.state:
                logging.info("State of jobflow: %s" % status.state)
            previous_state = status.state
            if status.state in ('FAILED', ''):
                logging.info('Stdout:')
                logging.info(self.get_log(log_path, jobid))
                logging.info('Stderr:')
                logging.info(self.get_log(log_path, jobid, 'stderr'))
                logging.info('Controller:')
                logging.info(self.get_log(log_path, jobid, 'controller'))
                raise ImportHandlerException('Emr jobflow %s failed' % jobid)
            if status.state in ('COMPLETED'):
                break
        result = self.get_result(self.name)
        logging.info(result[0:10000])
        return iter([])
        #return itertools.imap(lambda s: json.loads(s),itertools.imap(lambda s: s.strip('\n'), result))
        #return itertools.imap(lambda s: json.loads(s), result.splitlines()[1:])


class DataSource(object):
    DATASOURCE_DICT = {
        'db': DbDataSource,
        'pig': PigDataSource,
        'http': HttpDataSource,
    }

    @classmethod
    def factory(cls, config):
        return cls.DATASOURCE_DICT[config.tag](config)
