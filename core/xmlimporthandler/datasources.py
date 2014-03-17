import logging
import time
import itertools
import json

import boto
from boto.s3.key import Key
import boto.emr
from boto.emr.step import PigStep, InstallPigStep

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
        self.emr_conn = boto.emr.connect_to_region(
            'us-west-2',
            aws_access_key_id=self.AMAZON_ACCESS_TOKEN,
            aws_secret_access_key=self.AMAZON_TOKEN_SECRET)
        self.jobid = config.get('jobid', None)
        ts = int(time.time())
        self.result_path = "/cloudml/output/%s/%d/" % (self.name, ts)
        self.result_uri = "s3://%s%s" % (self.BUCKET_NAME, self.result_path)

        self.log_path = '%s/%s' % (self.S3_LOG_URI, self.name)
        self.log_uri = 's3://%s%s' % (self.BUCKET_NAME, self.log_path)

    def store_query_to_s3(self, query, query_target=None):
        # substitute your bucket name here
        b = self.s3_conn.get_bucket(self.BUCKET_NAME)
        k = Key(b)
        k.key = 'cloudml/pig/' + self.name + '_script.pig'
        k.set_contents_from_string(query)
        return 's3://%s/%s' % (self.BUCKET_NAME, k.key)

    def get_result(self):
        b = self.s3_conn.get_bucket(self.BUCKET_NAME)
        k = Key(b)
        k.key = "%s/part-m-00000" % self.result_path
        return k.get_contents_as_string()

    def delete_output(self, name):
        b = self.s3_conn.get_bucket(self.BUCKET_NAME)
        k = Key(b)
        k.key = "cloudml/output/%s" % name
        k.delete()

    def get_log(self, log_uri, jobid, step, log_type='stdout'):
        b = self.s3_conn.get_bucket(self.BUCKET_NAME)
        k = Key(b)
        k.key = "%s/%s/steps/%d/%s" % (log_uri, jobid, step, log_type)
        logging.info('Log uri: %s' % k.key)
        return k.get_contents_as_string()

    def print_logs(self, log_path, step_number):
        logging.info('Step logs:')
        try:
            logging.info('Stdout:')
            logging.info(self.get_log(log_path, self.jobid, step_number))
            logging.info('Controller:')
            logging.info(self.get_log(log_path, self.jobid, step_number, 'controller'))
            logging.info('Stderr:')
            logging.info(self.get_log(log_path, self.jobid, step_number, 'stderr'))
        except:
            logging.info('Logs are anavailable now (updated every 5 mins)')


    def _get_iter(self, query, query_target=None):
        pig_file = self.store_query_to_s3(query)
        steps = []
        if self.jobid is None:
            install_pig_step = InstallPigStep(pig_versions=self.PIG_VERSIONS)
            steps.append(install_pig_step)
        pig_step = PigStep(self.name,
                     pig_file=pig_file,
                     pig_versions=self.PIG_VERSIONS,
                     pig_args=['-p output=%s' % self.result_uri])
        pig_step.action_on_failure = 'CONTINUE'
        steps.append(pig_step)
        self.delete_output(self.name)
        if self.jobid is not None:
            status = self.emr_conn.describe_jobflow(self.jobid)
            step_number = len(status.steps) + 1
            logging.info('Use existing emr jobflow: %s' % self.jobid)
            step_list = self.emr_conn.add_jobflow_steps(self.jobid, steps)
            step_id = step_list.stepids[0].value
        else:
            logging.info('Run emr jobflow')
            step_number = 2
            self.jobid = self.emr_conn.run_jobflow(name='Cloudml jobflow',
                              log_uri=log_uri,
                              ami_version='2.2',
                              keep_alive=True,
                              action_on_failure='CONTINUE',#'CANCEL_AND_WAIT',
                              steps=steps)
            logging.info('JobFlowid: %s' % self.jobid)
        previous_state = None
        logging.info('Step number: %d' % step_number)

        while True:
            time.sleep(10)
            status = self.emr_conn.describe_jobflow(self.jobid)
            laststatechangereason = status.laststatechangereason
            if previous_state != status.state:
                logging.info("State of jobflow: %s" % status.state)
            previous_state = status.state
            if status.state in ('FAILED', '') or \
                (status.state in ('WAITING',) and \
                laststatechangereason == 'Waiting after step failed'):
                self.print_logs(self.log_path, step_number)
                raise ImportHandlerException('Emr jobflow %s failed' % self.jobid)
            if status.state in ('COMPLETED', 'WAITING'):
                break
        result = self.get_result()
        return itertools.imap(lambda s: json.loads(s), result.splitlines())


class DataSource(object):
    DATASOURCE_DICT = {
        'db': DbDataSource,
        'pig': PigDataSource,
        'http': HttpDataSource,
    }

    @classmethod
    def factory(cls, config):
        return cls.DATASOURCE_DICT[config.tag](config)
