#! /usr/bin/env python
# encoding: utf-8

import os
import logging
import json
from lxml import etree
from lxml import objectify
from  decimal import Decimal

from datasources import DataSource
from inputs import Input
from entities import Entity, EntityProcessor
from utils import iterchildren
from exceptions import ImportHandlerException, ProcessException
from scripts import ScriptManager


BASEDIR = os.path.abspath(os.path.dirname(__file__))


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return "%.2f" % obj
        return json.JSONEncoder.default(self, obj)


class ExtractionPlan(object):
    """
    Reads and validates extraction plan configuration from a XML file.
    """
    def __init__(self, config, is_file=True):
        if is_file:
            with open(config, 'r') as fp:
                config = fp.read()

        try:
            self.data = objectify.fromstring(config)
        except etree.XMLSyntaxError as e:
            raise ImportHandlerException(message='%s %s ' % (config, e))

        if not self.is_valid():
            raise ImportHandlerException(
                'XML file format is invalid: %s' % self.errors)

        self.inputs = {}
        self.load_inputs(self.data)

        self.datasources = {}
        self.load_datasources(self.data)

        self.script_manager = ScriptManager()
        self.load_scripts(self.data)

        # Loading import section
        self.entity = Entity(self.data['import'].entity)

        # TODO: predict section

    # Loading sections methods

    def load_inputs(self, config):
        """
        Loads dictionary of the input parameters
        from import handler configuration.
        """
        inputs_conf = config.inputs
        if inputs_conf is not None:
            for param_conf in inputs_conf.xpath("param"):
                inp = Input(param_conf)
                self.inputs[inp.name] = inp

    def load_datasources(self, config):
        """
        Loads global datasources from configuration.
        """
        for ds_config in iterchildren(config.datasources):
            ds = DataSource.factory(ds_config)
            self.datasources[ds.name] = ds

    def load_scripts(self, config):
        """
        Loads and executes javascript from import handler configuration.
        """
        for script in config.xpath("script"):
            if script.text:
                self.script_manager.add_js(script.text)

    # Schema Validation specific methods

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


class ImportHandler(object):

    def __init__(self, plan, params={}):
        super(ImportHandler, self).__init__()
        self.count = 0
        self.ignored = 0
        self.params = {}
        self.plan = plan

        self.process_input_params(params)
        self.entity_processor = EntityProcessor(
            self.plan.entity, import_handler=self)

    def __iter__(self):
        return self

    def store_data_json(self, output, compress=False):
        """
        Stores the given data to file output using JSON format. The output file
        contains multiple JSON objects, each one containing the data of an
        individual row.

        Keyword arguments:
        output -- the file to store the data to.
        compress -- whether we need to archive data using gzip.

        """
        open_mthd = gzip.open if compress else open
        with open_mthd(output, 'w') as fp:
            for row_data in self:
                fp.write('%s\n' % json.dumps(row_data, cls=DecimalEncoder))
        fp.close()

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
        param_set = set(params.keys() if params else ())
        required_set = set(self.plan.inputs.keys())
        missing = required_set.difference(param_set)
        if len(missing) > 0:
            raise ImportHandlerException('Missing input parameters: %s'
                                         % ', '.join(missing))

        for name, inp in self.plan.inputs.iteritems():
            self.params[name] = inp.process_value(params[name])
