"""
This module gathers DataSource classes.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import sys
import logging
import csv
import time
import itertools
import json
import urllib
import contextlib
from botocore.exceptions import ClientError
import boto3
import requests
from requests import ConnectionError
import cStringIO

from exceptions import ImportHandlerException, ProcessException
from db import postgres_iter, run_queries, check_table_name

logging.getLogger('boto').setLevel(logging.INFO)


DATASOURCES_REQUIRE_QUERY = ['db', 'pig']


__all__ = ['DATASOURCES_REQUIRE_QUERY', 'DataSource']


class BaseDataSource(object):
    """
    Base class for any type of the datasource.

    config: lxml.etree._Element
        parsed by lxml.objectify datasource definition tag.
    """
    def __init__(self, config):
        self.config = config
        self.name = config.get('name')  # unique
        if not self.name:
            raise ImportHandlerException('name is required')
        self.type = config.tag

    def _get_iter(self, query=None, query_target=None, params=None):
        """
        Gets datasource iterator.

        query: string
            query string could be different dependly of datasource type.
        query_target: string
            for some datasources it's a name of the entity where results
            would be stored.
        """
        raise Exception('Not implemented')

    def get_params(self):
        return dict([
            (key, val) for key, val in self.config.attrib.iteritems()
            if key != 'name'])


# TODO: implement named connections like:
# <db name="namedDBConnection" name-ref="myODWConnection" />
class DbDataSource(BaseDataSource):
    """
    Database connection.

    config: lxml.etree._Element
        parsed by lxml.objectify datasource definition tag.

    Config should contains attributes:
        name: string
            unique name for this datasource
        host: string
    """
    DB = {
        'postgres': [postgres_iter, run_queries]
    }

    def _get_iter(self, query, query_target=None, params=None):
        """
        Gets datasource iterator for specified select query.

        query: string
            PostgreSQL query string
        query_target: string
            Name of the table, from with we need to select data.
            If `query_target` specified, at the end of the queries
            statement would be added select all from table, named
            query_target expression.
        """
        return self._run(query, query_target)

    def run_queries(self, query):
        """
        Runs query on this datasource.

        query: string
            PostgreSql query string with expressions separated by ';'.
            ';' is required at the end of the query.

        Note:
            It using in the PigDataSource before run Sqoop import
            for creating the table with intermediate results.
        """
        self._run(query, query_target=None, run=True)

    def _run(self, query, query_target=None, run=False):
        queries = self._get_queries_list(query, query_target)
        method = 1 if run else 0
        vendor = self.config.attrib['vendor']
        db_iter = self.DB.get(vendor)[method] if vendor in self.DB else None
        if db_iter is None:
            raise ImportHandlerException(
                'Database type %s not supported' % vendor)

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

    def _get_queries_list(self, query, query_target=None):
        if query is None:
            raise ImportHandlerException(
                "Query is required in the DB datasource")

        query = query.strip(' \t\n\r')
        if not query:
            raise ImportHandlerException(
                "Query is required in the DB datasource")

        if not query.endswith(';'):
            query += ';'

        queries = query.split(';')[:-1]
        queries = [q + ';' for q in queries]

        if query_target:
            check_table_name(query_target)
            queries.append("SELECT * FROM %s;" % query_target)

        return queries


class HttpDataSource(BaseDataSource):
    """
    Datasource for importing JSON data from remote HTTP services.

    config: lxml.etree._Element
        parsed by lxml.objectify datasource definition tag.

    Config contains attributes:
        name: string
            unique name for this datasource
        url: string
            remote service url
        method: string (optional), default='GET'
            http method
    """
    def __init__(self, config):
        super(HttpDataSource, self).__init__(config)

        attrs = self.config.attrib
        if 'url' not in attrs:
            raise ImportHandlerException('No url given')

        self.url = attrs['url']
        if not self.url:
            raise ImportHandlerException('No url given')

        self.method = attrs.get('method', 'GET')

    def _get_iter(self, query=None, query_target=None, params=None):
        """
        Returns datasource iterator.

        query: string
            query string of the url.
        query_target: string
            it isn't used for this datasource.
        """
        if query_target is not None:
            raise ImportHandlerException(
                "Http datasource doesn't support query_target")
        if query:
            query = query.strip()
            url = '{0}/{1}'.format(
                self.url.rstrip('/'), str(query).lstrip('/'))
        else:
            url = self.url

        # TODO: params?
        logging.info('Getting data from: %s' % url)
        try:
            resp = requests.request(self.method, url, stream=True)
        except ConnectionError as exc:
            raise ImportHandlerException(
                'Cannot reach url: {}'.format(str(exc)))
        try:
            result = resp.json()
        except Exception as exc:
            raise ImportHandlerException(
                'Cannot parse json: {}'.format(str(exc)))
        if isinstance(result, dict):
            return iter([resp.json()])
        return iter(resp.json())


class CsvDataSource(BaseDataSource):
    """
    Datasource which can use CSV file for getting data from local files.

    config: lxml.etree._Element
        parsed by lxml.objectify datasource definition tag.

    Config contains:
        name: string
            unique name for this datasource
        src: string
            the path to the CSV file
        headers: string (optional), default='GET'
            Header information can be defined by adding child <header>
            elements to the <csv> element. Each <header> element must
            contain exactly two fields:
                * name - the name of the column
                * index - the column's index (columns are zero-indexed).
    """
    def __init__(self, config):
        super(CsvDataSource, self).__init__(config)

        attrs = self.config.attrib
        self.headers = []
        for t in self.config.xpath('header'):
            self.headers.append(
                (t.attrib['name'], int(t.attrib['index'])))

        if 'src' not in attrs:
            raise ImportHandlerException('No source given')
        self.src = attrs['src']
        if 'delimiter' in attrs:
            self.delimiter = attrs['delimiter']
            if self.delimiter == '\\t':
                self.delimiter = '\t'
        else:
            self.delimiter = ','
        try:
            self.offset = int(attrs.get('offset', 0))
            self.count = int(attrs.get('count', sys.maxint))
        except (ValueError, TypeError):
            raise ImportHandlerException('offset and count should be integers')
        logging.info('In csv datasource {0} there are '
                     'offset {1} and count {2}'.format(
                         self.name, self.offset, self.count))

    def _get_iter(self, query=None, query_target=None, params=None):
        def __get_obj(row):
            if len(self.headers) == 0:
                return {str(i): row[i] for i in range(0, len(row))}
            obj = {}
            for name, idx in self.headers:
                if len(row) <= idx:
                    raise ImportHandlerException(
                        "csv file {0} doesn't contains column "
                        "{1}, named {2}".format(self.src, idx, name))
                obj[name] = row[idx]
            return obj

        i = 0
        with contextlib.closing(urllib.urlopen(self.src)) as stream:
            reader = csv.reader(stream, delimiter=self.delimiter)
            # TODO: not is the best way. Refactore
            for row in reader:
                i += 1
                if i < self.offset:
                    continue
                if i > (self.offset + self.count):
                    break

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
    """
    A connection to a remote Hadoop/Pig cluster

    config: lxml.etree._Element
        parsed by lxml.objectify datasource definition tag.

    Config contains:
        name: string
            a unique name for this datasource
        jobid: string (optional)
            id of hadoop pig jobflow
        amazon_access_token: string
        amazon_token_secret: string
        bucket_name: string (optional)
            Amazon S3 bucket name to store input data and results
        master_instance_type: string (optional)
            Instance type of the master node of EMR
        slave_instance_type: string (optional)
        num_instances: string (optional)
        hadoop_params: string (optional)
        keep_alive: bool (optional)
        ec2_keyname: string (optional)
        ami_version: string (optional)
    """
    SQOOP_COMMAND = '''sqoop import --verbose --connect "%(connect)s" \
--username %(user)s --password %(password)s --table %(table)s \
--direct-split-size 4000000000 -m %(mappers)s %(options)s'''

    def __init__(self, config):
        super(PigDataSource, self).__init__(config)
        self.import_handler = None
        from config import AMAZON_ACCESS_TOKEN, AMAZON_TOKEN_SECRET, \
            S3_LOG_URI, BUCKET_NAME, DEFAILT_AMI_VERSION, \
            DEFAULT_INSTANCE_TYPE, DEFAULT_NUM_INSTANCES
        self.amazon_access_token = self.config.get(
            'amazon_access_token', AMAZON_ACCESS_TOKEN)
        self.amazon_token_secret = self.config.get(
            'amazon_token_secret', AMAZON_TOKEN_SECRET)
        self.master_instance_type = self.config.get(
            'master_instance_type', DEFAULT_INSTANCE_TYPE)
        self.slave_instance_type = self.config.get(
            'slave_instance_type', DEFAULT_INSTANCE_TYPE)
        self.num_instances = self.config.get(
            'num_instances', DEFAULT_NUM_INSTANCES)
        self.keep_alive = self.config.get('keep_alive', False)
        self.ec2_keyname = self.config.get('ec2_keyname', 'cloudml-control')
        self.hadoop_params = self.config.get('hadoop_params', None)
        self.ami_version = self.config.get('ami_version', DEFAILT_AMI_VERSION)
        self.bucket_name = self.config.get('bucket_name', BUCKET_NAME)

        self.s3 = boto3.resource(
            's3',
            aws_access_key_id=self.amazon_access_token,
            aws_secret_access_key=self.amazon_token_secret)
        self.emr = boto3.client(
            'emr',
            region_name='us-west-1',
            aws_access_key_id=self.amazon_access_token,
            aws_secret_access_key=self.amazon_token_secret)

        self.jobid = config.get('jobid', None)
        ts = int(time.time())
        self.result_path = "/cloudml/output/%s/%d/" % (self.name, ts)
        self.result_uri = "s3n://%s%s" % (self.bucket_name, self.result_path)
        self.sqoop_result_uri = "s3n://%s/cloudml/output/%s/%d_sqoop/" % (
            self.bucket_name, self.name, ts)
        self.sqoop_results_uries = {}
        self.log_path = '%s/%s' % (S3_LOG_URI, self.name)
        self.log_uri = 's3://%s%s' % (self.bucket_name, self.log_path)
        logging.info('Using ami version %s' % self.ami_version)

    @property
    def cluster_is_exist(self):
        return self.jobid is not None

    @property
    def s3_logs_folder(self):
        return 's3://{0}{1}/{2}/steps/'.format(
            self.bucket_name, self.log_path, self.jobid)

    def set_import_handler(self, import_handler):
        self.import_handler = import_handler

    def run_sqoop_imports(self, sqoop_imports=[]):
        """
        Runs sqoop imports and saves results to Amazon S3 on
        `self.sqoop_result_uri`.
        """
        for sqoop_import in sqoop_imports:
            db_param = sqoop_import.datasource.config[0].attrib
            connect = "jdbc:postgresql://%s:%s/%s" % (
                db_param['host'],
                db_param.get('port', '5432'),
                db_param['dbname'])
            sqoop_script = self.SQOOP_COMMAND % {
                'table': sqoop_import.table,
                'connect': connect,
                'password': db_param['password'],
                'user': db_param['user'],
                'mappers': sqoop_import.mappers,
                'options': sqoop_import.options}
            if sqoop_import.where:
                sqoop_script += " --where %s" % sqoop_import.where
            if sqoop_import.direct:
                sqoop_script += " --direct"
            sqoop_result_uri = "%s%s/" % (
                self.sqoop_result_uri, sqoop_import.target)
            self.sqoop_results_uries[sqoop_import.target] = sqoop_result_uri
            sqoop_script += " --target-dir %s" % sqoop_result_uri

            logging.info('Sqoop command: %s' % sqoop_script)
            import subprocess

            p = subprocess.Popen(
                sqoop_script, shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT)
            for line in p.stdout.readlines():
                logging.info(line)
            retval = p.wait()
            if retval != 0:
                raise ImportHandlerException('Sqoop import  failed')

    def _get_iter(self, query, query_target=None, params=None):
        """
        Returns results iterator.
        """
        self.process_pig_script(query, query_target)
        return self.get_result()

    def generate_download_url(self, step, log_type, expires_in=3600):
        """
        Generates url to download running pig script logs.
        """
        s3_client = self.s3.meta.client
        key_name = '/{1}/{2}/steps/%d/%s'.format(
            self.log_path, self.jobid, step, log_type)
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket_name, 'Key': key_name},
            ExpiresIn=expires_in)
        return url

    def _get_result_job(self, response, j_id):
        job_flows = response.get('JobFlows', None)
        if not job_flows:
            raise ImportHandlerException("Unexpected EMR result: {}"
                                         .format(response))
        job = None
        for j in job_flows:
            job_id = j.get('JobFlowId', None)
            if job_id and job_id == j_id:
                job = j  # job with requested id found
                break
        if not job:
            raise ImportHandlerException("Job with id #%s not found "
                                         "in response" % self.jobid)
        return job

    def process_pig_script(self, query, query_target=None):
        logging.info('Start processing pig datasource...')
        try:
            pig_step = self.get_pig_step(query, query_target)
        except Exception, exc:
            raise ProcessException(
                "Can't initialize pig datasource: {0}".format(exc))

        self.clear_output_folder(self.name)

        if self.cluster_is_exist:
            step_number = self._run_steps_on_existing_jobflow(pig_step)
        else:
            step_number = self._create_jobflow_and_run_steps(
                bootstrap_actions=self._get_bootstrap_actions(),
                pig_step=pig_step)
        logging.info('Step number is %d' % step_number)

        previous_state = None
        while True:
            time.sleep(10)
            try:
                res = self.emr.describe_job_flows(JobFlowIds=[self.jobid])
            except ClientError:
                logging.info("Getting throttled. Sleeping for 10 secs.")
                time.sleep(10)
                continue

            # getting required job from returned result
            job = self._get_result_job(res, self.jobid)

            # getting job state
            status_detail = job.get('ExecutionStatusDetail', None)
            if not status_detail:
                raise ImportHandlerException("Status Details are not "
                                             "returned for job: {}".
                                             format(job))
            state = status_detail.get('State', None)
            if not state:
                raise ImportHandlerException("State is not present in "
                                             "job status details: {}".
                                             format(status_detail))

            if previous_state != state:
                steps = job.get('Steps', None)
                step_state = None
                if steps:
                    step = steps[int(step_number) - 1]
                    step_detail = step.get('ExecutionStatusDetail', None)
                    if step_detail:
                        step_state = step_detail.get('State', None)
                logging.info(
                    "State of jobflow changed: %s. Step %d state is: %s",
                    state, step_number, step_state)

                fn_name = '_process_{0}_state'.format(state.lower())
                if hasattr(self, fn_name):
                    process_fn = getattr(self, fn_name)
                    process_fn(job, step_state, int(step_number))
                else:
                    logging.warning(
                        'Jobflow status is unexpected: %s', state)

                if state == 'COMPLETED' or \
                        (state == 'WAITING'
                            and step_state == 'COMPLETED'):
                    break  # job is completed -> no need check status

                previous_state = state

        logging.info(
            "Pig results stored to: s3://%s%s" %
            (self.bucket_name, self.result_path))

    def get_result(self):
        """
        Returns running pig script results.
        """
        def key_exists(s3, bucket, key_name):
            """
            Checks if key exists in the bucket
            """
            try:
                s3.Object(bucket, key_name).load()
                return True
            except ClientError as e:
                if e.response['Error']['Code'] == "404":
                    return False
                else:
                    raise ImportHandlerException(e.message)

        # TODO: Need getting data from all nodes
        type_result = 'm'
        if not key_exists(self.s3, self.bucket_name,
                          "%spart-m-00000" % self.result_path):
            type_result = 'r'
        i = 0
        first_result = False
        while True:
            sbuffer = cStringIO.StringIO()
            k_name = "%spart-%s-%05d" % (self.result_path, type_result, i)
            if not key_exists(self.s3, self.bucket_name, k_name):
                break
            logging.info('Getting from s3 file %s' % k_name)
            key = self.s3.Object(self.bucket_name, k_name).get()
            for chunk in iter(lambda: key['Body'].read(4096), b''):
                sbuffer.write(chunk)

            i += 1
            sbuffer.seek(0)
            for line in sbuffer:
                pig_row = json.loads(line)
                if not first_result:
                    if self.import_handler is not None and \
                            self.import_handler.callback is not None:
                        callback_params = {
                            'jobflow_id': self.jobid,
                            'pig_row': pig_row
                        }
                        self.import_handler.callback(**callback_params)
                    first_result = True
                yield pig_row
            sbuffer.close()

    # Creating pig step related methods

    def get_pig_step(self, query, query_target=None):
        """
        Returns pig step
        """
        logging.info('Appending pig step.')
        pig_file = self._store_query_to_s3(query, query_target)
        pig_args = ['s3n://us-east-1.elasticmapreduce/libs/pig/pig-script',
                    '--base-path',
                    's3n://us-east-1.elasticmapreduce/libs/pig/']
        pig_args.extend(['--pig-versions', 'latest'])
        pig_args.extend(['--run-pig-script', '--args', '-f', pig_file])
        pig_args.extend(['-p', 'output=%s' % self.result_uri])
        for k, v in self.sqoop_results_uries.iteritems():
            pig_args.append("-p")
            pig_args.append("%s=%s" % (k, v))
        pig_step = {
            'Name': self.name,
            'ActionOnFailure': 'CONTINUE',
            'HadoopJarStep': {
                'Jar':
                's3n://us-east-1.elasticmapreduce/libs/script-runner/'
                'script-runner.jar',
                'Args': pig_args}
        }
        return pig_step

    def _store_query_to_s3(self, query, query_target=None):
        """
        Stores pig sctipt to Amazon S3 and returns uri to it.
        """
        if query_target:
            query += \
                "\nSTORE %s INTO '$output' USING JsonStorage();" % query_target
        logging.info("Store pig script to s3: %s" % query)
        k_name = 'cloudml/pig/' + self.name + '_script.pig'
        self.s3.Object(self.bucket_name, k_name).put(Body=query)
        return 's3://%s/%s' % (self.bucket_name, k_name)

    def clear_output_folder(self, name):
        """
        Deletes results of previous execution scripts on this datasource.
        """
        logging.info('Clearing output folder on Amazon S3')
        self.s3.Object(self.bucket_name, "cloudml/output/%s" % name).delete()

    # Run steps on jobflow related methods

    def _run_steps_on_existing_jobflow(self, pig_step):
        status = self.emr.describe_job_flows(JobFlowIds=[self.jobid])
        job = self._get_result_job(status, self.jobid)
        steps = job.get('Steps', None)
        step_number = len(steps) + 1 if steps is not None else 1
        logging.info('Using existing emr jobflow: %s' % self.jobid)
        self.emr.add_job_flow_steps(JobFlowId=self.jobid, Steps=[pig_step, ])
        return step_number

    def _create_jobflow_and_run_steps(self, bootstrap_actions, pig_step):
        """
        Runs hadoop jobflow and saves ID to jobid field.
        """
        logging.info('Running emr jobflow')
        logging.info(self.log_uri)
        res = self.emr.run_job_flow(
            Name='CloudML jobflow',
            LogUri=self.log_uri,
            AmiVersion=self.ami_version,
            VisibleToAllUsers=True,
            BootstrapActions=bootstrap_actions,
            Instances={
                'Ec2KeyName': self.ec2_keyname,
                #'KeepJobFlowAliveWhenNoSteps': self.keep_alive,
                'MasterInstanceType': self.master_instance_type,
                'SlaveInstanceType': self.slave_instance_type,
                'InstanceCount': int(self.num_instances),
                #'Ec2SubnetId': 'subnet-3f5bc256',
            },
            JobFlowRole='EMR_EC2_DefaultRole',
            ServiceRole='EMR_DefaultRole',
            Steps=[pig_step, ])
        self.jobid = res.get('JobFlowId', None)
        logging.info('New JobFlow id is %s' % self.jobid)
        return 1

    def _get_bootstrap_actions(self):
        """
        Returns bootstrap actions for starting new cluster:
            * install ganglia
            * configure hadoop
        """
        bootstrap_actions = []
        install_ganglia = {
            'Name': 'Install ganglia',
            'ScriptBootstrapAction': {
                'Path':
                    's3://elasticmapreduce/bootstrap-actions/install-ganglia'}
        }
        bootstrap_actions.append(install_ganglia)
        if self.hadoop_params is not None:
            params = self.hadoop_params.split(',')
            config_bootstrapper = {
                'Name': 'Configure hadoop',
                'ScriptBootstrapAction': {
                    'Path':
                    's3://elasticmapreduce/bootstrap-actions/configure-hadoop',
                    'Args': params}
            }
            bootstrap_actions.append(config_bootstrapper)
        return bootstrap_actions

    # Different jobflow states processing methods.

    def _process_running_state(self, status, step_state, step_number):
        """
        Processes `RUNNING` job status.
        """
        masterpublicdnsname = None
        instance = status.get('Instances', None)
        if instance:
            masterpublicdnsname = instance.get('MasterPublicDnsName', None)

        if masterpublicdnsname:
            if self.import_handler is not None and \
                    self.import_handler.callback is not None:
                callback_params = {
                    'jobflow_id': self.jobid,
                    's3_logs_folder': self.s3_logs_folder,
                    'master_dns': masterpublicdnsname,
                    'step_number': step_number
                }
                self.import_handler.callback(**callback_params)
            logging.info("Master node dns name: %s" % masterpublicdnsname)
            logging.info(
                '''For access to hadoop web ui please create ssh tunnel:
ssh -D localhost:12345 hadoop@%(dns)s -i ~/.ssh/cloudml-control.pem
After creating ssh tunnel web ui will be available on localhost:9026 using
socks proxy localhost:12345''' % {'dns': masterpublicdnsname})

    def _process_completed_state(self, status, step_state, step_number):
        """
        Processes `COMPLETED` job status.
        """
        if step_state == 'FAILED':
            self._fail_jobflow(step_number)
        elif step_state == 'COMPLETED':
            logging.info('Step is completed')
        else:
            state = status.get('ExecutionStatusDetail').get('State', None)
            logging.info(
                'Unexpected job state for status %s: %s',
                state, step_state)

    def _process_failed_state(self, status, step_state, step_number):
        """
        Processes `FAILED` job status.
        """
        self._fail_jobflow(step_number)

    def _process_waiting_state(self, status, step_state, step_number):
        """
        Processes `WAITING` job status.
        """
        if step_state == 'PENDING':
            # we reusing cluster and have waiting status
            # of jobflow from previous job
            pass
        elif step_state == 'FAILED':
            self._fail_jobflow(step_number)
        elif step_state == 'COMPLETED':
            logging.info('Step is completed')
        else:
            state = status.get('ExecutionStatusDetail').get('State', None)
            logging.info(
                'Unexpected job state for status %s: %s',
                state, step_state)

    def _fail_jobflow(self, step_number):
        logging.error('Jobflow failed, shutting down.')
        self._print_logs(self.log_path, step_number)
        raise ImportHandlerException('Emr jobflow %s failed' % self.jobid)

    def _print_logs(self, log_path, step_number):
        """
        Outputs pig script stdout, stderr, controller logs using logging.
        """
        logging.info('Step logs:')
        try:
            logging.info('Stdout:')
            logging.info(self._get_log(log_path, self.jobid, step_number))
            logging.info('Controller:')
            logging.info(self._get_log(
                log_path, self.jobid, step_number, 'controller'))
            logging.info('Stderr:')
            logging.info(self._get_log(
                log_path, self.jobid, step_number, 'stderr'))
        except Exception, exc:
            logging.error('Exception occures while loading logs: %s', exc)
            logging.info('Logs are unavailable now (updated every 5 mins)')
            logging.info('''For getting stderr log please use command:
    s3cmd get s3://%s%s/%s/steps/%d/stderr stderr''' % (
                self.bucket_name, log_path, self.jobid, step_number))
            logging.info('''For getting stdout log please use command:
    s3cmd get s3://%s%s/%s/steps/%d/stdout stdout''' % (
                self.bucket_name, log_path, self.jobid, step_number))

    def _get_log(self, log_uri, jobid, step, log_type='stdout'):
        """
        Gets running pig script logs.
        """
        k_name = "/%s/%s/steps/%d/%s" % (log_uri, jobid, step, log_type)
        logging.info('Log uri: %s' % k_name)
        res = self.s3.Object(self.bucket_name, k_name).get()
        return res['Body'].read(res['ContentLength'])


class InputDataSource(BaseDataSource):
    def __init__(self, config=None):
        self.config = {}
        self.name = 'input'
        self.type = 'input'

    def get_params(self):
        return {}

    def _get_iter(self, query=None, query_target=None, params=None):
        if query == 'any':
            return iter([params])
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
        if config.tag not in cls.DATASOURCE_DICT:
            raise ImportHandlerException(
                '{0} datasource type is not supported'.format(config.tag))
        return cls.DATASOURCE_DICT[config.tag](config)
