__author__ = 'ifouk'

import logging
import json
import re

from jsonpath import jsonpath

EXPRESSION_RE = re.compile('%\(([^\(\)]+)\)s')


class ProcessException(Exception):
    """
    Exception to be raised in case there's a problem processing a feature.

    """
    def __init__(self, message, column=None, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        self._column = column
        self.Errors = Errors

###
### Functions to use for implementing each strategy
###

###############################################################################


def process_primitive(constructor):
    def process(value, query_item, row_data):
        """
        Function to invoke when processing a feature that simply returns the
        value of a column.

        Keyword arguments:
        value -- the value to process for the given feature
        query_item -- the query item currently processing. Must contain a
                      'target-features' list, with one item. This item must 
                      be a dictionary with the property 'name' set.
        row_data -- a map containing the values of the current row processed 
                    so far.

        """
        target_features = query_item.get('target-features', [])

        result = None
        if value is not None:
            result = constructor(value)

        return {target_features[0]['name']: result}
    
    return process

def process_composite(value, query_item, row_data):
    """
    Function to invoke when processing a feature value that is created from
    other features.

    Keyword arguments:
    value -- the value to process for the given feature
    query_item -- the query item currently processing. Must contain a
                  'target-features' list, with one item. This item must be a
                  dictionary with the property 'name' set.
    row_data -- a map containing the values of the current row processed so
                far.

    """
    result = {}
    target_features = query_item.get('target-features', [])

    for feature in target_features:
        if 'expression' not in feature or \
                'type' not in feature['expression'] or \
                'value' not in feature ['expression']:
            raise ProcessException(
                    'Must define an expression with "type" and "value" for '
                    'target feature %s' % (feature['name']))
        expression_type = feature['expression']['type']
        expression_value = feature['expression']['value']

        required_params = extract_parameters(expression_value)
        missing_params = filter(lambda x: x not in row_data, required_params)
        if len(missing_params) > 0:
            logging.debug('Missing values %s for target feature %s'
                          % (', '.join(missing_params), feature['name']))
            result[feature['name']] = None
        else:
            try:
                if expression_type == 'string':
                    result[feature['name']] = expression_value % row_data
                elif expression_type == 'python':
                    result[feature['name']] = eval(expression_value % row_data)
            except NameError as e:
                raise ProcessException('%s (expression: %s)' %
                                       (e, expression_value % row_data))

    return result


def process_json(value, query_item, row_data):
    """
    Function to invoke when processing a feature value that is a JSON message.

    Keyword arguments:
    value -- the value to process for the given feature
    query_item -- the query item currently processing. Must contain a
                  'target-features' list, with one item. This item must be a
                  dictionary with the property 'name' set.
    row_data -- a map containing the values of the current row processed so
                far.

    """
    # Parse JSON string
    try:
        if isinstance(value, basestring):
            data = json.loads(value)
        else:
            data = value
    except:
        raise ProcessException('Couldn\'t parse JSON message')

    result = {}

    target_features = query_item.get('target-features', [])
    for feature in target_features:
        if 'jsonpath' not in feature:
            raise ProcessException('Target feature %s has no JSONPath'
                                   % feature['path'])
        path_result = jsonpath(data, feature['jsonpath'])

        if path_result is not False:
            # Found result in given path
            if 'key-path' in feature and 'value-path' in feature:
                # Treat as a dictionary
                keys = jsonpath(path_result[0], feature['key-path'])
                values = map(float, jsonpath(path_result[0], feature['value-path']))
                if keys is not False and values is not False:
                    result[feature['name']] = dict(zip(keys, values))
                else:
                    result[feature['name']] = None
            elif len(path_result) > 1:
                # Multiple results from JSONPath
                result_list = filter(None, path_result)
                if feature.get('to-csv', False) is True:
                    result[feature['name']] = ','.join(result_list)
                else:
                    result[feature['name']] = result_list
            else:
                result[feature['name']] = path_result[0]
        else:
            # No data in given path
            result[feature['name']] = None

    return result


def extract_parameters(expression):
    """
    Returns a list with all parameter names for the given expression. Checks
    for all parameters in the format %(name)s
    """
    return EXPRESSION_RE.findall(expression)


PROCESS_STRATEGIES = {
    'identity': process_primitive(lambda x: x),
    'string': process_primitive(str),
    'float': process_primitive(float),
    'boolean': process_primitive(bool),
    'integer': process_primitive(int),
    'json': process_json,
    'composite': process_composite
}
