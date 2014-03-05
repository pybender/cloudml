import logging

import boto
from boto.s3.key import Key
from exceptions import ImportHandlerException
from core.importhandler.db import postgres_iter

logging.basicConfig(level=logging.INFO)


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


class DbDataSource(BaseDataSource):
    """
    Database connection.
    """
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

    def get_log(self, log_uri, jobid, type):
        # substitute your bucket name here
        b = self.s3_conn.get_bucket(self.BUCKET_NAME)
        k = Key(b)
        k.key = "%s/%s/steps/1/stderr" % (log_uri, jobid)
        return k.get_contents_as_string()

    def _get_iter(self, query, query_target=None):
        import boto.emr
        from boto.emr.step import PigStep

        conn = boto.emr.connect_to_region(
            'us-west-2',
            aws_access_key_id=self.AMAZON_ACCESS_TOKEN,
            aws_secret_access_key=self.AMAZON_TOKEN_SECRET)
        #INPUT=s3://myawsbucket/input,-p,OUTPUT=s3://myawsbucket/output
        pig_file = self.store_query_to_s3(query)
        log_path = '%s/%s.log' % (self.S3_LOG_URI, self.name)
        log_uri = 's3://%s/%s' % (self.BUCKET_NAME, log_path)
        # step = PigStep(self.name,
        #              pig_file=pig_file,
        #              pig_versions='latest',
        #              pig_args=[])
        logging.info('Run emr jobflow')
        # jobid = conn.run_jobflow(name='Cloudml jobflow',
        #                   log_uri=log_uri,
        #                   steps=[step])
        jobid = 'j-1IUFTB38IEYIV'
        logging.info('JobFlowid: %s' % jobid)
        status = conn.describe_jobflow(jobid)
        logging.info(status.state)
        print self.get_log(log_path, jobid)
        if status.state in ('FAILED'):
            raise ImportHandlerException('Emr jobflow %s failed' % jobid)
        pass


class DataSource(object):
    DATASOURCE_DICT = {
        'db': DbDataSource,
        'pig': PigDataSource,
        'http': HttpDataSource,
    }

    @classmethod
    def factory(cls, config):
        return cls.DATASOURCE_DICT[config.tag](config)
