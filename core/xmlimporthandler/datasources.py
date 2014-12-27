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
from boto.exception import EmrResponseError
from boto.emr.step import PigStep, InstallPigStep, JarStep
from boto.emr import BootstrapAction

from exceptions import ImportHandlerException, ProcessException
from core.importhandler.db import postgres_iter, run_queries

logging.getLogger('boto').setLevel(logging.INFO)


DATASOURCES_REQUIRE_QUERY = ['db', 'pig']


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
        for key, val in self.config.attrib.iteritems():
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
        db_iter = self.DB.get(self.config.attrib['vendor'])[0]

        if db_iter is None:
            raise ImportHandlerException(
                'Database type %s not supported' % self.config['db']['vendor'])

        if 'host' not in self.config.attrib:
            raise ImportHandlerException(
                'No database connection details defined')

        from copy import deepcopy
        conn_params = deepcopy(self.config.attrib)
        conn_params.pop('name')
        conn_params.pop('vendor')
        conn_string = ' '.join(['%s=%s' % (k, v)
                                for k, v in conn_params.iteritems()])
        return db_iter(queries, conn_string)

    def run_queries(self, queries):
        queries = queries.strip(' \t\n\r')
        queries = queries.split(';')[:-1]
        queries = [q + ';' for q in queries]
        run_query = self.DB.get(self.config.attrib['vendor'])[1]
        if run_query is None:
            raise ImportHandlerException(
                'Database type %s not supported' % self.config['db']['vendor'])
        from copy import deepcopy
        conn_params = deepcopy(self.config.attrib)
        conn_params.pop('name')
        conn_params.pop('vendor')
        conn_string = ' '.join(['%s=%s' % (k, v)
                                for k, v in conn_params.iteritems()])
        run_query(queries, conn_string)


class HttpDataSource(BaseDataSource):
    def _get_iter(self, query=None, query_target=None):
        attrs = self.config.attrib

        if 'url' not in attrs:
            raise ImportHandlerException('No url given')

        url = attrs['url']
        if query:
            query = query.strip()
            url = '{0}/{1}'.format(url.rstrip('/'), str(query).lstrip('/'))
        method = attrs.get('method', 'GET')

        # TODO: params?
        logging.info('Getting data from: %s' % url)
        try:
            resp = requests.request(
                method, url, stream=True)
        except ConnectionError as exc:
            raise ImportHandlerException('Cannot reach url: {}'.format(
                str(exc)))
        try:
            result = resp.json()
        except Exception as exc:
            raise ImportHandlerException('Cannot parse json: {}'.format(
                str(exc)))
        if isinstance(result, dict):
            return iter([resp.json()])
        return iter(resp.json())


class CsvDataSource(BaseDataSource):
    def _get_iter(self, query=None, query_target=None):
        attrs = self.config.attrib

        headers = []
        for t in self.config.xpath('header'):
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
    AMAZON_ACCESS_TOKEN = 'AKIAJYNNIPFWF2ABW42Q'
    AMAZON_TOKEN_SECRET = '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'
    BUCKET_NAME = 'odesk-match-prod'
    DEFAILT_AMI_VERSION = '3.1.0'
    SQOOP_COMMANT = '''sqoop import --verbose --connect "%(connect)s" --username %(user)s --password %(password)s --table %(table)s --direct-split-size 4000000000 -m %(mappers)s %(options)s'''

    def __init__(self, config):
        super(PigDataSource, self).__init__(config)
        self.steps = []
        self.amazon_access_token = self.config.get('amazon_access_token', self.AMAZON_ACCESS_TOKEN)
        self.amazon_token_secret = self.config.get('amazon_token_secret', self.AMAZON_TOKEN_SECRET)
        self.master_instance_type = self.config.get('master_instance_type', 'm1.small')
        self.slave_instance_type = self.config.get('slave_instance_type', 'm1.small')
        self.num_instances = self.config.get('num_instances', 1)
        self.keep_alive = self.config.get('keep_alive', False)
        self.ec2_keyname = self.config.get('ec2_keyname', 'cloudml-control')
        self.hadoop_params = self.config.get('hadoop_params', None)
        self.ami_version = self.config.get('ami_version', self.DEFAILT_AMI_VERSION)
        logging.info('Using ami version %s' % self.ami_version)
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
        self.sqoop_result_uri = "s3n://%s/cloudml/output/%s/%d_sqoop/" % (self.bucket_name, self.name, ts)
        self.sqoop_results_uries = {}
        self.log_path = '%s/%s' % (self.S3_LOG_URI, self.name)
        self.log_uri = 's3://%s%s' % (self.bucket_name, self.log_path)

        # bucket = self.s3_conn.lookup(self.bucket_name)
        # if bucket is None:
        #     raise ValueError("invalid bucket name")
        # log_key = Key(bucket)
        # log_key.key = self.log_path
        # self.log_url = key.generate_url(expires_in=0, query_auth=False)

        #self.prepare_cluster()

    def set_ih(self, ih):
        self.ih = ih

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
        import cStringIO
        #TODO: Need getting data from all nodes
        k.key = "%spart-m-00000" % self.result_path
        type_result = 'm'
        if not k.exists():
            type_result = 'r'
        i = 0
        callback_sent = False
        while True:
            sbuffer = cStringIO.StringIO()
            k = Key(b)
            k.key = "%spart-%s-%05d" % (self.result_path, type_result, i)
            if not k.exists():
                break
            logging.info('Getting from s3 file %s' % k.key)
            k.get_contents_to_file(sbuffer)
            i += 1
            sbuffer.seek(0)
            for line in sbuffer:
                pig_row = json.loads(line)
                if not callback_sent and self.ih.callback is not None:
                    callback_params = {
                        'jobflow_id': self.jobid,
                        'pig_row': pig_row
                    }
                    self.ih.callback(**callback_params)
                    callback_sent = True
                yield pig_row
            sbuffer.close()

    def delete_output(self, name):
        b = self.s3_conn.get_bucket(self.bucket_name)
        k = Key(b)
        k.key = "cloudml/output/%s" % name
        k.delete()

    def generate_download_url(self, step, log_type, expires_in=3600):
        b = self.s3_conn.get_bucket(self.bucket_name)
        key = Key(b)
        key.key = '/{1}/{2}/steps/%d/%s'.format(
            self.log_path, self.jobid, step, log_type)
        return key.generate_url(expires_in)

    def get_log(self, log_uri, jobid, step, log_type='stdout'):
        b = self.s3_conn.get_bucket(self.bucket_name)
        k = Key(b)
        k.key = "/%s/%s/steps/%d/%s" % (log_uri, jobid, step, log_type)
        logging.info('Log uri: %s' % k.key)
        return k.get_contents_as_string()

    @property
    def s3_logs_folder(self):
        return 's3://{0}{1}/{2}/steps/'.format(
            self.bucket_name, self.log_path, self.jobid)

    def print_logs(self, log_path, step_number):
        logging.info('Step logs:')
        try:
            logging.info('Stdout:')
            logging.info(self.get_log(log_path, self.jobid, step_number))
            logging.info('Controller:')
            logging.info(self.get_log(log_path, self.jobid, step_number, 'controller'))
            logging.info('Stderr:')
            logging.info(self.get_log(log_path, self.jobid, step_number, 'stderr'))
        except Exception, exc:
            logging.error('Exception occures while loading logs: %s', exc)
            logging.info('Logs are unavailable now (updated every 5 mins)')
            logging.info('''For getting stderr log please use command:
    s3cmd get s3://%s%s/%s/steps/%d/stderr stderr''' % (self.bucket_name, log_path, self.jobid, step_number))
            logging.info('''For getting stdout log please use command:
    s3cmd get s3://%s%s/%s/steps/%d/stdout stdout''' % (self.bucket_name, log_path, self.jobid, step_number))
            #logging.info('Download url: %s', generate_download_url(step_number, 'stderr'))

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
                                                 'mappers': sqoop_import.mappers,
                                                 'options': sqoop_import.options}
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

    #############################
    # get_iter and it's helpers #
    #############################

    def _run_jobflow(self):
        bootstrap_actions = []
        install_ganglia = BootstrapAction(
            'Install ganglia',
            's3://elasticmapreduce/bootstrap-actions/install-ganglia', []
        )
        bootstrap_actions.append(install_ganglia)
        if self.hadoop_params is not None:
            params = self.hadoop_params.split(',')
            config_bootstrapper = BootstrapAction(
                'Configure hadoop',
                's3://elasticmapreduce/bootstrap-actions/configure-hadoop',
                params
            )
            bootstrap_actions.append(config_bootstrapper)
        logging.info('Running emr jobflow')
        self.jobid = self.emr_conn.run_jobflow(name='Cloudml jobflow',
                          log_uri=self.log_uri,
                          ami_version=self.ami_version,
                          visible_to_all_users=True,
                          bootstrap_actions=bootstrap_actions,
                          ec2_keyname=self.ec2_keyname,
                          keep_alive=self.keep_alive,
                          num_instances=self.num_instances,
                          master_instance_type=self.master_instance_type,
                          slave_instance_type=self.slave_instance_type,
                          ##api_params={'Instances.Ec2SubnetId':'subnet-3f5bc256'},
                          action_on_failure='CONTINUE',#'CANCEL_AND_WAIT',
                          steps=self.steps)
        logging.info('New JobFlow id is %s' % self.jobid)

    def _append_pig_step(self, query, query_target=None):
        pig_file = self.store_query_to_s3(query, query_target)
        pig_args=['-p', 'output=%s' % self.result_uri]

        for k,v in self.sqoop_results_uries.iteritems():
            pig_args.append("-p")
            pig_args.append("%s=%s" % (k, v))
        pig_step = PigStep(self.name,
                     pig_file=pig_file,
                     #pig_versions=self.pig_version,
                     pig_args=pig_args)
        pig_step.action_on_failure = 'CONTINUE'
        self.steps.append(pig_step)

    def _process_running_state(self, status, step_number):
        if hasattr(status, 'masterpublicdnsname'):
            masterpublicdnsname = status.masterpublicdnsname
            if self.ih.callback is not None:
                callback_params = {
                    'jobflow_id': self.jobid,
                    's3_logs_folder': self.s3_logs_folder,
                    'master_dns': masterpublicdnsname,
                    'step_number': step_number
                }
                self.ih.callback(**callback_params)
            logging.info("Master node dns name: %s" % masterpublicdnsname)
            logging.info('''For access to hadoop web ui please create ssh tunnel:
ssh -D localhost:12345 hadoop@%(dns)s -i ~/.ssh/cloudml-control.pem
After creating ssh tunnel web ui will be available on localhost:9026 using
socks proxy localhost:12345'''  % {'dns': masterpublicdnsname})

    def _fail_jobflow(self, step_number):
        logging.error('Jobflow failed, shutting down.')
        self.print_logs(self.log_path, step_number)
        raise ImportHandlerException('Emr jobflow %s failed' % self.jobid)

    def _get_iter(self, query, query_target=None):
        logging.info('Processing pig datasource...')
        try:
            self._append_pig_step(query, query_target)
        except Exception, exc:
            msg = "Can't initialize pig datasource: {0}".format(exc)
            logging.error(msg)
            raise ProcessException(msg)

        self.delete_output(self.name)
        if self.jobid is not None:
            status = self.emr_conn.describe_jobflow(self.jobid)
            step_number = len(status.steps) + 1
            logging.info('Using existing emr jobflow: %s' % self.jobid)
            step_list = self.emr_conn.add_jobflow_steps(self.jobid, self.steps)
            #step_id = step_list.stepids[0].value
        else:
            step_number = 1
            self._run_jobflow()

        logging.info('Step number: %d' % step_number)

        # Checking jobflow state...
        previous_state = None
        while True:
            time.sleep(10)
            try:
                status = self.emr_conn.describe_jobflow(self.jobid)
            except EmrResponseError:
                logging.info("Getting throttled. Sleeping for 10 secs.")
                time.sleep(10)
                continue

            # laststatechangereason = None
            # if hasattr(status, 'laststatechangereason'):
            #     laststatechangereason = status.laststatechangereason
            
            if previous_state != status.state:
                step_state = status.steps[step_number - 1].state
                logging.info(
                    "State of jobflow changed: %s. Step %s state is: %s",
                    status.state, step_number, step_state)

                if status.state == 'RUNNING':
                    self._process_running_state(status, step_number)

                if status.state == 'COMPLETED':
                    if step_state == 'FAILED':
                        self._fail_jobflow(step_number)
                    elif step_state == 'COMPLETED':
                        logging.info('Step is completed')
                        break
                    else:
                        logging.info(
                            'Unexpected job state for status %s: %s', status.state, step_state)
                elif status.state == 'RUNNING':
                    pass  # processing the task
                elif status.state == 'WAITING':
                    if step_state == 'PENDING':
                        # we reusing cluster and have waiting status of jobflow from previous job
                        pass
                    elif step_state == 'FAILED':
                        self._fail_jobflow(step_number)
                    elif step_state == 'COMPLETED':
                        logging.info('Step is completed')
                        break
                    else:
                        logging.info(
                            'Unexpected job state for status %s: %s', status.state, step_state)
                elif status.state == 'FAILED':
                    self._fail_jobflow(step_number)

                previous_state = status.state

        logging.info("Pig results stored to: s3://%s%s" % (self.bucket_name, self.result_path))
        # for test
        #self.result_path =  '/cloudml/output/pig-script/1403671176/'
        return self.get_result()


class InputDataSource(BaseDataSource):

    def __init__(self):
        self.config = {}
        self.name = 'input'
        self.type = 'input'

    def get_params(self):
        return {}

    def _get_iter(self, query=None, query_target=None):
        try:
            result = json.loads(query)
        except Exception as exc:
            raise ImportHandlerException('Cannot parse json: {}'.format(
                str(exc)))
        if isinstance(result, dict):
            return iter([result])
        return iter(result)


class DataSource(object):
    DATASOURCE_DICT = {
        'db': DbDataSource,
        'pig': PigDataSource,
        'http': HttpDataSource,
        'csv': CsvDataSource,
        'input': InputDataSource
    }

    @classmethod
    def factory(cls, config):
        return cls.DATASOURCE_DICT[config.tag](config)
