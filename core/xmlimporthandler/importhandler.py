#! /usr/bin/env python
# encoding: utf-8

import logging
import json
from jsonpath import jsonpath
from string import Template
from lxml import etree
from lxml import objectify

from core.importhandler.db import postgres_iter


class SqlTemplate(Template):
    delimiter = '#'


def process_primitive(constructor):
    def process(value, field):
        """
        Function to invoke when processing a feature that simply returns the
        value of a column.

        Keyword arguments:
        value -- the value to process for the given feature
        field --
        """
        name = field.get('name')

        result = None
        if value is not None:
            result = constructor(value)

        return {field.get('name'): result}

    return process


PROCESS_STRATEGIES = {
    'string': process_primitive(str),
    'float': process_primitive(float),
    'boolean': process_primitive(bool),
    'integer': process_primitive(int)
}


class ImportHandlerException(Exception):
    def __init__(self, message, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        self.Errors = Errors


class ProcessException(Exception):
    """
    Exception to be raised in case there's a problem processing a feature.

    """
    def __init__(self, message, column=None, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        self._column = column
        self.Errors = Errors


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
            raise ImportHandlerException('XML file format is invalid: %s' % self.errors)

    @property
    def data(self):
        if hasattr(self, '_data'):
            return self._data
        return None

    @property
    def errors(self):
        if not hasattr(self, '_errors'):
            self._validate_schema()
        return self._errors

    def is_valid(self):
        return not self.errors

    def _validate_schema(self):
        self._errors = []
        with open('schema.xsd', 'r') as schema_fp:
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

    # def _validate_items(self, query):
    #     """
    #     Validates that all necessary fields of the query are there.

    #     Keyword arguments:
    #     query -- a dictionary containing the query's configuration

    #     """
    #     for item in query.get('items'):
    #         target_features = item.get('target_features', [])
    #         if len(target_features) == 0:
    #             raise ImportHandlerException('Query item must define at least '
    #                                          'one target feature')
    #         for feature in target_features:
    #             if 'name' not in feature:
    #                 raise ImportHandlerException('Target features must have '
    #                                              'a name')


class BaseImportHandler(object):

    def __init__(self, plan):
        self._plan = plan

    def _process_row(self, row):
        return row


class ImportHandler(BaseImportHandler):

    DB_ITERS = {
        'postgres': postgres_iter
    }

    def __init__(self, plan, params=None):
        super(ImportHandler, self).__init__(plan)
        self.count = 0
        self.ignored = 0
        if params is None:
            params = {}

        logging.info('Validate inputs')
        self._validate_input_params(params)

        self.main_entity = self._plan._data['import'].getchildren()[0]

        query = self.prepare_query(self.main_entity, params)
        datasource_name = self.main_entity.attrib['datasource']
        datasource = self._plan._data.xpath("//*[@name='%s']" % datasource_name)[0]
        self._iterator = self._get_iter(datasource, query)

    def prepare_query(self, main_entity, params):
        query = SqlTemplate(main_entity.query.text).safe_substitute(params)
        target = main_entity.query.get('target')
        query = [query]
        if target:
            query.append("SELECT * FROM %s;" % target)
        return query

    def _get_iter(self, datasource, query): 
        if datasource.tag == 'db':
            db_iter = self.DB_ITERS.get(datasource[0].attrib['vendor'])
            if db_iter is None:
                raise ImportHandlerException('Database type %s not supported'
                                         % datasource['db']['vendor'])

            if 'host' not in datasource[0].attrib:
                raise ImportHandlerException('No database connection details '
                                         'defined')
            conn_params = datasource[0].attrib
            conn_params.pop('name')
            conn_params.pop('vendor')
            conn_string = ' '.join(['%s=%s' % (k, v)for k, v in conn_params.iteritems()])
            print 'query',query
            iterator = db_iter(query, conn_string)
        elif datasource.tag == 'pig':
            raise ImportHandlerException('Not implement')
        else:
            raise ImportHandlerException('Datasource type %s not supported' % datasource.tag)
        

        return iterator

    def _process_field(self, field, value):
        result = {}
        field_type = field.get('type')
        if field_type is not None:
            strategy = PROCESS_STRATEGIES.get(field_type, None)
            if strategy is None:
                raise ImportHandlerException('Unknown strategy %s'
                                                  % field_type)
            return strategy(value, field)
        result[field.get('name')] = value
        return result

    def _process_row(self, row, entity):
        """
        Processes a single row from DB.
        """
        # Hold data of current row processed so far
        row_data = {} 
        for field in entity.xpath("field"):
            name = field.get('name')
            # # Get value from query for this item
            source = field.get('column', None)
            item_value = row.get(source, None)
            transform = field.get('transform', None)
            if transform == 'json':
                # Parse JSON string
                result = {}
                try:
                    if isinstance(item_value, basestring):
                        data = json.loads(item_value)
                    else:
                        data = item_value
                except:
                    raise ProcessException('Couldn\'t parse JSON message')

                for field in entity.xpath("entity[@datasource='%s']/field" % name):
                    name = field.get('name')
                    path_result = jsonpath(data, field.get('jsonpath'))
                    row_data.update(self._process_field(field, path_result[0]))
            else:
                result = self._process_field(field, item_value)
                row_data.update(result)
        return row_data

    def next(self):
        if self.count % 10 == 0:
            logging.info('Processed %s rows so far' % (self.count, ))
        result = None
        #while result is None:
        try:
            row = self._process_row(self._iterator.next(), self.main_entity)
            self.count += 1
            return row
        except ProcessException, e:
            logging.debug('Ignored line #%d: %s' % (self.count, str(e)))
            self.ignored += 1

    def _validate_input_params(self, params):
        """
        Validates that all required input params of the current extraction plan
        exist in the given param dictionary.

        Keyword arguments
        params -- the parameters to check

        """
        param_set = set()
        if params is not None:
            param_set = set(params.keys())

        required_set = set([param.get('name') for param in self._plan._data.inputs.iterchildren()])
        missing = required_set.difference(param_set)
        if len(missing) > 0:
            raise ImportHandlerException('Missing input parameters: %s'
                                         % ', '.join(missing))
