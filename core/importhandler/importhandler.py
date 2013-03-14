#! /usr/bin/env python
# encoding: utf-8
"""
importhandler-- extract values from DB according to extraction plan.

It defines classes ImportHandlerException, ExtractionPlan and ImportHandler

@author:     ifoukarakis

@copyright: 2013 oDesk. All rights reserved.

"""

__author__ = 'ifouk'

import os
import sys
import json
import logging

from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter
from db import postgres_iter
from processors import extract_parameters
from processors import PROCESS_STRATEGIES, ProcessException


class ImportHandlerException(Exception):
    def __init__(self, message, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        self.Errors = Errors


class ExtractionPlan(object):
    """
    Reads extraction plan configuration from a file containing a JSON object.

    """
    def __init__(self, config, is_file=True):
        if is_file:
            with open(config, 'r') as fp:
                data = json.load(fp)
        else:
            data = json.loads(config)

        if 'target-schema' not in data:
            raise ImportHandlerException('No target schema defined in config')
        self.schema_name = data['target-schema']

        if len(data.get('datasource', [])) == 0:
            raise ImportHandlerException('No datasource defined in config')
        self.datasource = data['datasource']

        if len(data.get('queries', [])) == 0:
            raise ImportHandlerException('No queries defined in config')
        self.queries = data['queries']

        for query in self.queries:
            self._validate_items(query)

        self._find_required_input()

    def _find_required_input(self):
        """
        Iterates over the plan's queries in order to find which parameters
        should be provided by the user.

        """
        self.input_params = []

        for query in self.queries:
            user_params = extract_parameters(query.get('sql', ''))
            self.input_params.extend(user_params)

    def _validate_items(self, query):
        """
        Validates that all necessary fields of the query are there.

        Keyword arguments:
        query -- a dictionary containing the query's configuration

        """
        for item in query.get('items'):
            if 'process-as' not in item:
                raise ImportHandlerException('Query item must define '
                                             'process-as field')
            target_features = item.get('target-features', [])
            if len(target_features) == 0:
                raise ImportHandlerException('Query item must define at least '
                                             'one target feature')
            for feature in target_features:
                if 'name' not in feature:
                    raise ImportHandlerException('Target features must have '
                                                 'a name')


class ImportHandler(object):
    DB_ITERS = {
        'postgres': postgres_iter
    }

    def __init__(self, plan, params=None):
        self._plan = plan
        if params is None:
            params = {}
        # Currently we support a single query. This might change in the future.
        self._query = self._plan.queries[0]

        logging.info('Running query %s' % self._query['name'])
        datasource = self._plan.datasource[0]
        self.count = 0
        self.ignored = 0
        self._validate_input_params(params)
        sql = self._query['sql'] % params
        iter_func = self._get_db_iter(datasource)
        self._iterator = iter_func(sql, datasource['db']['conn'])

    def __iter__(self):
        return self

    def next(self):
        if self.count % 1000 == 0:
            logging.info('Processed %s rows so far' % (self.count, ))
        result = None
        while result is None:
            try:
                row = self._process_row(self._iterator.next(), self._query)
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

        required_set = set(self._plan.input_params)
        missing = required_set.difference(param_set)
        if len(missing) > 0:
            raise ImportHandlerException('Missing input parameters: %s'
                                         % ', '.join(missing))

    def _process_row(self, row, query):
        """
        Processes a single row from DB.

        """
        # Hold data of current row processed so far
        row_data = {}
        for item in query['items']:
            strategy = PROCESS_STRATEGIES.get(item['process-as'])
            if strategy is None:
                raise ImportHandlerException('Unknown strategy %s'
                                             % item['process-as'])
            # Get value from query for this dtata
            source = item.get('source', None)
            item_value = row.get(source, None)
            result = strategy(item_value, item, row_data)
            row_data.update(result)

        return row_data

    def _get_db_iter(self, datasource):

        if 'conn' not in datasource['db']:
            raise ImportHandlerException('No database connection details '
                                         'defined')

        db_iter = self.DB_ITERS.get(datasource['db']['vendor'])

        if db_iter is None:
            raise ImportHandlerException('Database type %s not supported'
                                         % datasource['db']['vendor'])

        return db_iter

    def store_data_json(self, output):
        """
        Stores the given data to file output using JSON format. The output file
        contains multiple JSON objects, each one containing the data of an
        individual row.

        Keyword arguments:
        output -- the file to store the data to.

        """
        with open(output, 'w') as fp:
            for row_data in self:
                fp.write('%s\n' % (json.dumps(row_data), ))
