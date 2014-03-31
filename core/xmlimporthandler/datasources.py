import logging
import csv
import time
import itertools
import json

import boto
from boto.s3.key import Key
import requests
from requests import ConnectionError
import boto.emr
from boto.emr.step import PigStep, InstallPigStep, JarStep

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
        res = {}
        for key, val in self.config[0].attrib.iteritems():
            if key not in ['name', ]:
                res[key] = val
        return res


class DbDataSource(BaseDataSource):
    """
    Database connection.
    """

    def _get_iter(self, query, query_target=None):
        query = query.strip(' \t\n\r')
        queries = query.split(';')[:-1]
        logging.info(queries)
        if query_target:
            queries.append("SELECT * FROM %s;" % query_target)
        db_iter = self.DB_ITERS.get(self.config[0].attrib['vendor'])

        if db_iter is None:
            raise ImportHandlerException(
                'Database type %s not supported' % self.config['db']['vendor'])

        if 'host' not in self.config[0].attrib:
            raise ImportHandlerException(
                'No database connection details defined')

        from copy import deepcopy
        conn_params = deepcopy(self.config[0].attrib)
        conn_params.pop('name')
        conn_params.pop('vendor')
        conn_string = ' '.join(['%s=%s' % (k, v)
                                for k, v in conn_params.iteritems()])
        return db_iter(queries, conn_string)


class HttpDataSource(BaseDataSource):
    def _get_iter(self, query=None, query_target=None):
        attrs = self.config[0].attrib

        if 'url' not in attrs:
            raise ImportHandlerException('No url given')

        url = attrs['url']
        if query:
            query = query.strip()
            url = '{0}/{1}'.format(url.rstrip('/'), str(query).lstrip('/'))
        method = attrs.get('method', 'GET')

        # TODO: params?
        try:
            resp = requests.request(
                method, url, stream=True)
        except ConnectionError as exc:
            raise ImportHandlerException('Cannot reach url: {}'.format(
                str(exc)))

        return iter(resp.json())


class CsvDataSource(BaseDataSource):
    def _get_iter(self, query=None, query_target=None):
        attrs = self.config[0].attrib

        headers = []
        for t in self.config[0].xpath('header'):
            headers.append((t.attrib['name'], int(t.attrib['index'])))

        if 'src' not in attrs:
            raise ImportHandlerException('No source given')

        def __get_obj(row):
            if len(headers) == 0:
                return row
            obj = {}
            for name, idx in headers:
                obj[name] = row[idx]
            return obj

        with open(attrs['src'], 'r') as stream:
            if len(headers) > 0:
                reader = csv.reader(stream)
            else:
                reader = csv.DictReader(stream)
            for row in reader:
                obj = __get_obj(row)
                for key, value in obj.items():
                    # Try load json field
                    strkey = str(obj[key])
                    if strkey.startswith('{') or strkey.startswith('['):
                        try:
                            obj[key] = json.loads(value)
                        except Exception:
                            pass
                yield obj


class PigDataSource(BaseDataSource):
    S3_LOG_URI = '/cloudml/logs'
    AMAZON_ACCESS_TOKEN = 'fill me'
    AMAZON_TOKEN_SECRET = 'fill me'
    BUCKET_NAME = 'odesk-match-prod'
    PIG_VERSIONS = '0.11.1'

    def __init__(self, config):
        super(PigDataSource, self).__init__(config)
        self.amazon_access_token = self.config.get('amazon_access_token')
        self.amazon_token_secret = self.config.get('amazon_token_secret')
        self.pig_version = self.config.get('pig_version', self.PIG_VERSIONS)
        logging.info('Use pig version %s' % self.pig_version)
        self.bucket_name = self.config.get('bucket_name', self.BUCKET_NAME)

        self.s3_conn = boto.connect_s3(self.amazon_access_token,
                                       self.amazon_token_secret)
        self.emr_conn = boto.emr.connect_to_region(
            'us-west-2',
            aws_access_key_id=self.amazon_access_token,
            aws_secret_access_key=self.amazon_token_secret)
        self.jobid = config.get('jobid', None)
        ts = int(time.time())
        self.result_path = "/cloudml/output/%s/%d/" % (self.name, ts)
        self.result_uri = "s3://%s%s" % (self.bucket_name, self.result_path)

        self.log_path = '%s/%s' % (self.S3_LOG_URI, self.name)
        self.log_uri = 's3://%s%s' % (self.bucket_name, self.log_path)

    def store_query_to_s3(self, query, query_target=None):
        # substitute your bucket name here
        b = self.s3_conn.get_bucket(self.bucket_name)
        k = Key(b)
        k.key = 'cloudml/pig/' + self.name + '_script.pig'
        k.set_contents_from_string(query)
        return 's3://%s/%s' % (self.bucket_name, k.key)

    def store_sqoop_script_to_s3(self, script=None):
        # substitute your bucket name here
        b = self.s3_conn.get_bucket(self.bucket_name)
        k = Key(b)
        k.key = 'cloudml/pig/' + self.name + '_script.sh'
        #k.set_contents_from_filename("./core/xmlimporthandler/install_sqoop.sh")
        k.set_contents_from_string(script)
        return 's3://%s/%s' % (self.bucket_name, k.key)

    def get_result(self):
        b = self.s3_conn.get_bucket(self.bucket_name)
        k = Key(b)
        k.key = "%s/part-m-00000" % self.result_path
        return k.get_contents_as_string()

    def delete_output(self, name):
        b = self.s3_conn.get_bucket(self.bucket_name)
        k = Key(b)
        k.key = "cloudml/output/%s" % name
        k.delete()

    def get_log(self, log_uri, jobid, step, log_type='stdout'):
        b = self.s3_conn.get_bucket(self.bucket_name)
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
            install_pig_step = InstallPigStep(pig_versions=self.pig_version)
            steps.append(install_pig_step)
            # install_sqoop_step = JarStep(name='Install sqoop',
            # jar='s3n://elasticmapreduce/libs/script-runner/script-runner.jar',
            # step_args=['s3n://install_sqoop.sh',],
            # action_on_failure='CONTINUE')
            #steps.append(install_sqoop_step)
#         query = '''#!/bin/bash
# cd 
# ./sqoop-1.4.4.bin__hadoop-1.0.0/bin/sqoop import --verbose --connect "jdbc:postgresql://172.27.13.141:12000/odw1 --username bestmatch --password bestmatch --table dataset -m 1 --direct
# '''
#         sqoop_script = self.store_sqoop_script_to_s3(query)
#         sqoop_step = JarStep(name='Run sqoop import',
#             jar='s3n://elasticmapreduce/libs/script-runner/script-runner.jar',
#             step_args=[sqoop_script,],
#             action_on_failure='CONTINUE')
        pig_step = PigStep(self.name,
                     pig_file=pig_file,
                     pig_versions=self.pig_version,
                     pig_args=['-p output=%s' % self.result_uri])
        pig_step.action_on_failure = 'CONTINUE'
        steps.append(pig_step)
        self.delete_output(self.name)
        if self.jobid is not None:
            status = self.emr_conn.describe_jobflow(self.jobid)
            step_number = len(status.steps) + 1
            logging.info('Use existing emr jobflow: %s' % self.jobid)
            step_list = self.emr_conn.add_jobflow_steps(self.jobid, steps)
            #step_id = step_list.stepids[0].value
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
        'csv': CsvDataSource
    }

    @classmethod
    def factory(cls, config):
        return cls.DATASOURCE_DICT[config.tag](config)
