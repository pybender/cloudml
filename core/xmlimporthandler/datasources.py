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
from core.importhandler.db import postgres_iter, run_queries

logging.getLogger('boto').setLevel(logging.INFO)


class BaseDataSource(object):
    """
    Base class for any type of the datasource.
    """
    DB = {
        'postgres': [postgres_iter, run_queries]
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
        if query_target:
            queries.append("SELECT * FROM %s;" % query_target)
        db_iter = self.DB.get(self.config[0].attrib['vendor'])[0]

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

    def run_queries(self, queries):
        queries = queries.strip(' \t\n\r')
        queries = queries.split(';')[:-1]
        queries = [q + ';' for q in queries]
        run_query = self.DB.get(self.config[0].attrib['vendor'])[1]
        if run_query is None:
            raise ImportHandlerException(
                'Database type %s not supported' % self.config['db']['vendor'])
        from copy import deepcopy
        conn_params = deepcopy(self.config[0].attrib)
        conn_params.pop('name')
        conn_params.pop('vendor')
        conn_string = ' '.join(['%s=%s' % (k, v)
                                for k, v in conn_params.iteritems()])
        run_query(queries, conn_string)


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
    SQOOP_COMMANT = '''sqoop import --verbose --connect "%(connect)s" --username %(user)s --password %(password)s --table %(table)s -m %(mappers)s'''

    def __init__(self, config):
        super(PigDataSource, self).__init__(config)
        self.steps = []
        self.amazon_access_token = self.config.get('amazon_access_token')
        self.amazon_token_secret = self.config.get('amazon_token_secret')
        self.master_instance_type = self.config.get('master_instance_type', 'm1.small')
        self.slave_instance_type = self.config.get('slave_instance_type', 'm1.small')
        self.num_instances = self.config.get('num_instances', 1)
        self.pig_version = self.config.get('pig_version', self.PIG_VERSIONS)
        logging.info('Use pig version %s' % self.pig_version)
        self.bucket_name = self.config.get('bucket_name', self.BUCKET_NAME)

        self.s3_conn = boto.connect_s3(self.amazon_access_token,
                                       self.amazon_token_secret)
        self.emr_conn = boto.emr.connect_to_region(
            'us-west-1',
            aws_access_key_id=self.amazon_access_token,
            aws_secret_access_key=self.amazon_token_secret)
        self.jobid = config.get('jobid', None)
        ts = int(time.time())
        self.result_path = "/cloudml/output/%s/%d/" % (self.name, ts)
        self.result_uri = "s3n://%s%s" % (self.bucket_name, self.result_path)
        self.sqoop_result_uri = "s3n://%s%ssqoop/" % (self.bucket_name, self.result_path)
        self.sqoop_results_uries = {}
        self.log_path = '%s/%s' % (self.S3_LOG_URI, self.name)
        self.log_uri = 's3://%s%s' % (self.bucket_name, self.log_path)

        self.prepare_cluster()

    def store_query_to_s3(self, query, query_target=None):
        if query_target:
            query +="\nSTORE %s INTO '$output' USING JsonStorage();" % query_target
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
        #TODO: Need getting data from all nodes
        k.key = "%s/part-m-00000" % self.result_path
        try:
            return k.get_contents_as_string()
        except:
            k.key = "%s/part-r-00000" % self.result_path
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
            logging.info('''For getting stderr log please use command:
    s3cmd get %s/%s/steps/%d/stderr stderr''' % (log_path, self.jobid, step_number))
            logging.info('''For getting stdout log please use command:
    s3cmd get %s/%s/steps/%d/stdout stdout''' % (log_path, self.jobid, step_number))

    def prepare_cluster(self):
        if self.jobid is None:
            install_pig_step = InstallPigStep(pig_versions=self.pig_version)
            self.steps.append(install_pig_step)
            install_sqoop_step = JarStep(name='Install sqoop',
            jar='s3n://elasticmapreduce/libs/script-runner/script-runner.jar',
            step_args=['s3n://%s/cloudml/pig/install_sqoop.sh' % self.bucket_name,])
            self.steps.append(install_sqoop_step)

    def run_sqoop_imports(self, sqoop_imports=[]):
        for sqoop_import in sqoop_imports:
            db_param = sqoop_import.datasource.config[0].attrib
            connect = "jdbc:postgresql://%s:%s/%s" % (db_param['host'],
                                                      db_param.get('port', '5432'),
                                                      db_param['dbname'])
            sqoop_script = self.SQOOP_COMMANT % {'table' :sqoop_import.table,
                                                        'connect': connect,
                                                        'password': db_param['password'],
                                                        'user': db_param['user'],
                                                        'mappers': sqoop_import.mappers}
            if sqoop_import.where:
                sqoop_script += " --where %s" % sqoop_import.where
            if sqoop_import.direct:
                sqoop_script += " --direct"
            sqoop_result_uri = "%s%s/" % (self.sqoop_result_uri, sqoop_import.target)
            self.sqoop_results_uries[sqoop_import.target] = sqoop_result_uri
            sqoop_script += " --target-dir %s" % sqoop_result_uri
            
            logging.info('Sqoop command: %s' % sqoop_script)
            import subprocess

            p = subprocess.Popen(sqoop_script, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in p.stdout.readlines():
                logging.info(line)
            retval = p.wait()
            if retval != 0:
                raise ImportHandlerException('Sqoop import  failed')
            #sqoop_script_uri = self.store_sqoop_script_to_s3(sqoop_script)
            # sqoop_step = JarStep(name='Run sqoop import',
            #     jar='s3n://elasticmapreduce/libs/script-runner/script-runner.jar',
            #     step_args=[sqoop_script_uri,],
            #     action_on_failure='CONTINUE')
            # self.steps.append(sqoop_step)

    def _get_iter(self, query, query_target=None):
        pig_file = self.store_query_to_s3(query)
        pig_args=['-p output=%s' % self.result_uri]
        for k,v in self.sqoop_results_uries.iteritems():
            pig_args.append("-p %s=%s" % (k, v))
        pig_step = PigStep(self.name,
                     pig_file=pig_file,
                     pig_versions=self.pig_version,
                     pig_args=pig_args)
        pig_step.action_on_failure = 'CONTINUE'
        self.steps.append(pig_step)
        self.delete_output(self.name)
        # if self.jobid is not None:
        #     status = self.emr_conn.describe_jobflow(self.jobid)
        #     step_number = len(status.steps) + 2
        #     logging.info('Use existing emr jobflow: %s' % self.jobid)
        #     step_list = self.emr_conn.add_jobflow_steps(self.jobid, self.steps)
        #     #step_id = step_list.stepids[0].value
        # else:
        #     logging.info('Run emr jobflow')
        #     step_number = 3
        #     self.jobid = self.emr_conn.run_jobflow(name='Cloudml jobflow',
        #                       log_uri=self.log_uri,
        #                       ami_version='2.2',
        #                       ec2_keyname='nmelnik',
        #                       keep_alive=True,
        #                       num_instances=self.num_instances,
        #                       master_instance_type=self.master_instance_type,
        #                       slave_instance_type=self.slave_instance_type,
        #                       #api_params={'Instances.Ec2SubnetId':'subnet-3f5bc256'},
        #                       action_on_failure='CONTINUE',#'CANCEL_AND_WAIT',
        #                       steps=self.steps)
        #     logging.info('JobFlowid: %s' % self.jobid)
        previous_state = None
        logging.info('Step number: %d' % step_number)
        
        while True:
            time.sleep(10)
            status = self.emr_conn.describe_jobflow(self.jobid)

            
            laststatechangereason = None
            if hasattr(status, 'laststatechangereason'):
                laststatechangereason = status.laststatechangereason
            if previous_state != status.state:
                logging.info("State of jobflow: %s" % status.state)
                if status.state == 'RUNNING':
                    if hasattr(status, 'masterpublicdnsname'):
                        masterpublicdnsname = status.masterpublicdnsname
                        logging.info("Master node dns name: %s" % masterpublicdnsname)
                        logging.info('''For access to hadoop web ui please create ssh tunnel:
ssh -L 9100:%(dns)s:9100 hadoop@%(dns)s -i ~/{yourkey}.pem
After creating ssh tunnel web ui will be available on localhost:9100'''  % {'dns': masterpublicdnsname})

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
