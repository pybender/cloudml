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


def process_string(value, query_item, row_data):
    """
    Function to invoke when processing a feature that simply returns the
    value of a column.

    Keyword arguments:
    value -- the value to process for the given feature
    query_item -- the query item currently processing. Must contain a
                  'target-features' list, with one item. This item must be a
                  dictionary with the property 'name' set.
    row_data -- a map containing the values of the current row processed so
                far.

    """
    target_features = query_item.get('target-features', [])

    result = None
    if value is not None:
        result = str(value)

    return {target_features[0]['name']: result}


def process_expression(value, query_item, row_data):
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
        if 'expression' not in feature:
            raise ProcessException('Must define an expression for  '
                                   'target feature %s' % (feature['name']))

        required_params = set(extract_parameters(feature['expression']))
        current_params = [k for k, v in row_data.items() if v is not None]
        missing = required_params.difference(set(current_params))
        if len(missing) > 0:
            logging.debug('Missing values %s for target feature %s'
                          % (', '.join(missing), feature['name']))
            result[feature['name']] = None
        else:
            result[feature['name']] = feature['expression'] % row_data

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
        data = json.loads(value)
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
                values = jsonpath(path_result[0], feature['value-path'])
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
    'string': process_string,
    'json': process_json,
    'expression': process_expression
}
