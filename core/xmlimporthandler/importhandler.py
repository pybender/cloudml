#! /usr/bin/env python
# encoding: utf-8

import os
import logging
from lxml import etree
from lxml import objectify

from datasources import DataSource
from inputs import Input
from entities import Entity, EntityProcessor
from utils import iterchildren
from exceptions import ImportHandlerException, ProcessException
from scripts import ScriptManager


BASEDIR = os.path.abspath(os.path.dirname(__file__))


class ExtractionPlan(object):
    """
    Reads and validates extraction plan configuration from a XML file.
    """
    def __init__(self, config, is_file=True):
        if is_file:
            with open(config, 'r') as fp:
                config = fp.read()

        try:
            self._data = objectify.fromstring(config)
        except etree.XMLSyntaxError as e:
            raise ImportHandlerException(message='%s %s ' % (config, e))

        if not self.is_valid():
            raise ImportHandlerException(
                'XML file format is invalid: %s' % self.errors)

    @property
    def data(self):
        if hasattr(self, '_data'):
            return self._data
        return None

    @property
    def import_(self):
        return self._data['import']

    @property
    def datasources(self):
        return self._data['datasources']

    @property
    def errors(self):
        if not hasattr(self, '_errors'):
            self._validate_schema()
        return self._errors

    def is_valid(self):
        return not self.errors

    def _validate_schema(self):
        self._errors = []
        with open(os.path.join(BASEDIR, 'schema.xsd'), 'r') as schema_fp:
            xmlschema_doc = etree.parse(schema_fp)
            xmlschema = etree.XMLSchema(xmlschema_doc)
            is_valid = xmlschema(self.data)
            if not is_valid:
                log = xmlschema.error_log
                error = log.last_error
                self._errors.append(error)

        # print self._data.get('type')
        # print self._data.datasources.getchildren()
        # for datasource in self._data.datasources.getchildren():
        #      print datasource.tag
        # print 'rrr',dir(self._data)
        # print g


        # if len(self.plan.get('datasource', [])) == 0:
        #     raise ImportHandlerException('No datasource defined in config')
        # self.datasource = data['datasource']

        # if len(data.get('queries', [])) == 0:
        #     raise ImportHandlerException('No queries defined in config')
        # self.queries = data['queries']

        # for query in self.queries:
        #     self._validate_items(query)

        # self._find_required_input()

    # def _find_required_input(self):
    #     """
    #     Iterates over the plan's queries in order to find which parameters
    #     should be provided by the user.

    #     """
    #     self.input_params = []

    #     for query in self.queries:
    #         user_params = extract_parameters(query.get('sql', ''))
    #         self.input_params.extend(user_params)


class ImportHandler(object):

    def __init__(self, plan, params={}):
        super(ImportHandler, self).__init__()
        self.count = 0
        self.ignored = 0

        config = plan._data

        self.load_inputs(config)
        self.process_input_params(params)

        self.load_datasources(config)
        self.load_scripts(config)

        # Loading import section
        self.entity = Entity(config['import'].entity)
        self.entity_processor = EntityProcessor(
            self.entity, import_handler=self)

    def load_inputs(self, config):
        """
        Loads dictionary of the input parameters
        from import handler configuration.
        """
        self.inputs = {}
        inputs_conf = config.inputs
        if inputs_conf is not None:
            for param_conf in inputs_conf.xpath("param"):
                inp = Input(param_conf)
                self.inputs[inp.name] = inp

    def load_datasources(self, config):
        """
        Loads global datasources from configuration.
        """
        self.datasources = {}
        for ds_config in iterchildren(config.datasources):
            ds = DataSource.factory(ds_config)
            self.datasources[ds.name] = ds

    def load_scripts(self, config):
        """
        Loads and executes javascript from import handler configuration.
        """
        self.script_manager = ScriptManager()
        for script in config.xpath("script"):
            self.script_manager.add_js(script.text)

    def next(self):
        if self.count % 10 == 0:
            logging.info('Processed %s rows so far' % (self.count, ))

        try:
            result = self.entity_processor.process_next()
            self.count += 1
            return result
        except ProcessException, e:
            logging.debug('Ignored line #%d: %s' % (self.count, str(e)))
            self.ignored += 1

    def process_input_params(self, params):
        """
        Validates that all required input params of the current extraction plan
        exist in the given param dictionary.

        Keyword arguments
        params -- the parameters to check
        """
        logging.info('Validate input parameters.')
        self.params = {}
        param_set = set(params.keys() if params else ())
        required_set = set(self.inputs.keys())
        missing = required_set.difference(param_set)
        if len(missing) > 0:
            raise ImportHandlerException('Missing input parameters: %s'
                                         % ', '.join(missing))

        for name, inp in self.inputs.iteritems():
            self.params[name] = inp.process_value(params[name])
