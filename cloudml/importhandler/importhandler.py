#! /usr/bin/env python
# encoding: utf-8

"""
Classes for extracting values from various datasources according\
to extraction plan.
It defines classes ExtractionPlan and ImportHandler
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import os
import logging
import json
import gzip
from lxml import etree
from lxml import objectify
from decimal import Decimal
from collections import defaultdict

from datasources import DataSource
from inputs import Input
from entities import Entity, EntityProcessor
from utils import iterchildren
from exceptions import ImportHandlerException, ProcessException
from scripts import ScriptManager, Script
from predict import Predict
from base import AmazonSettingsMixin

BASEDIR = os.path.abspath(os.path.dirname(__file__))

__all__ = ['ExtractionPlan', 'ImportHandler']


class DecimalEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Decimal):
            return "%.2f" % obj
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


class ExtractionPlan(AmazonSettingsMixin):
    """
    Reads and validates extraction plan configuration from a XML file.
    """
    def __init__(self, config, is_file=True):
        if is_file:
            with open(config, 'r') as fp:
                config = fp.read()

        if not config:
            raise ImportHandlerException('import handler file is empty')

        try:
            self.data = objectify.fromstring(config)
        except etree.XMLSyntaxError as e:
            raise ImportHandlerException(
                "Valid XML is expected for import handler. "
                "Parse error: {0}".format(e)
            )

        if not self.is_valid():
            raise ImportHandlerException(
                "There is an error in the import handler's XML, "
                "line {0}. {1}".format(self.error.line, self.error.message))

        self.inputs = {}
        self.load_inputs(self.data)

        self.datasources = {}
        self.load_datasources(self.data)

        self.scripts = []
        self.script_manager = ScriptManager()
        self.load_scripts(self.data)

        # Loading import section
        self.entity = Entity(self.data['import'].entity)

        # Predict section
        self.predict = Predict(self.data.predict) if \
            hasattr(self.data, 'predict') else None

    # Loading sections methods

    def load_inputs(self, config):
        """
        Loads dictionary of the input parameters
        from import handler configuration.
        """
        if not hasattr(config, "inputs"):
            logging.debug("No input parameters declared")
            return

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
            if ds.name in self.datasources:
                raise ImportHandlerException(
                    'There are few datasources with name {0}'.format(ds.name))
            self.datasources[ds.name] = ds

        ds = DataSource.DATASOURCE_DICT['input']()
        self.datasources[ds.name] = ds

    def load_scripts(self, config):
        """
        Loads and executes javascript from import handler configuration.
        """
        for script in config.xpath("script"):
            s = Script(script)
            s.amazon_settings = self.amazon_settings
            self.scripts.append(s)
            self.script_manager.add_python(s.get_script_str())

    # Schema Validation specific methods

    @property
    def error(self):
        if not hasattr(self, '_error'):
            self._validate_schema()
        return self._error

    def is_valid(self):
        return not self.error

    @classmethod
    def get_schema(cls):
        with open(os.path.join(BASEDIR, 'schema.xsd'), 'r') as schema_fp:
            return etree.parse(schema_fp)

    @classmethod
    def get_datasources_config(cls):
        schema = cls.get_schema()
        conf = defaultdict(list)
        namespaces = {'xs': 'http://www.w3.org/2001/XMLSchema'}
        datasources = schema.xpath(
            '//xs:element[@name = "datasources"]/'
            'xs:complexType/xs:sequence/xs:element', namespaces=namespaces
        )
        for d in datasources:
            params = []
            for a in d.xpath('xs:complexType/xs:attribute[@name != "name"]',
                             namespaces=namespaces):
                params.append({
                    'name': a.attrib['name'],
                    'type': str(a.attrib.get('type') or 'string')
                    .replace('xs:', ''),
                    'required': a.attrib.get('use') == 'required'
                })
            conf[d.attrib['name']] = {'type': d.attrib['name'],
                                      'parameters': params}

        return dict(conf)

    def _validate_schema(self):
        logging.debug('Validating schema...')
        self._error = None
        with open(os.path.join(BASEDIR, 'schema.xsd'), 'r') as schema_fp:
            xmlschema_doc = etree.parse(schema_fp)
            xmlschema = etree.XMLSchema(xmlschema_doc)
            is_valid = xmlschema(self.data)
            if not is_valid:
                log = xmlschema.error_log
                error = log.last_error
                self._error = error


class ImportHandler(object):

    def __init__(self, plan, params={}, callback=None):
        super(ImportHandler, self).__init__()
        self.count = 0
        self.ignored = 0
        self.params = {}
        self.plan = plan
        self.callback = callback

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

    def store_data_csv(self, output, compress=False):
        """
        Stores the given data to file output using CSV format. The output file
        contains multiple comma-separated rows and header containing names of
        the columns.

        Keyword arguments:
        output -- the file to store the data to.
        compress -- whether we need to archive data using gzip.

        """
        import csv

        open_mthd = gzip.open if compress else open

        def _encode(data):
            for key, value in data.iteritems():
                if isinstance(value, basestring):
                    data[key] = value.replace('\r', '')
                elif isinstance(value, (dict, list)):
                    data[key] = json.dumps(value)
            return data

        with open_mthd(output, 'w') as fp:
            first_row = next(self)
            if not first_row:
                return
            fieldnames = first_row.keys()

            writer = csv.DictWriter(
                fp,
                fieldnames=fieldnames,
                quotechar="'",
                quoting=csv.QUOTE_ALL
            )

            writer.writeheader()
            writer.writerow(_encode(first_row))
            for row_data in self:
                writer.writerow(_encode(row_data))

    def next(self):
        if self.count % 1000 == 0:
            logging.info('Processed %s rows so far' % (self.count, ))
        result = None
        while result is None:
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
