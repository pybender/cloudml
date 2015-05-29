# -*- coding: utf-8 -*-
__author__ = 'ifouk'

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import logging
import json
import re
import math
import datetime


from jsonpath import jsonpath
from sklearn.feature_extraction.readability import Readability

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


def extract_skill_weights(data, opening_skills):
    weight_per_tier = {
        '1': 1,
        '2': 0.5,
        '3': -0.5,
        '4': -1
    }
    if data is None:
        return {}
    if opening_skills is None or len(opening_skills) == 0:
        return {}
    sum_weights = {}
    for assignment in data:
        feedback = assignment.get('feedback_given', {}).get('private_feedback_hire_again')
        skills = assignment.get('as_skills')
        if skills is not None and len(skills) > 0 and 'as_total_charge' in assignment:
            total_charge = float(assignment.get('as_total_charge', 0))
            if total_charge < 0: total_charge = 0
            for s in skills.split(','):
                if s not in sum_weights:
                    sum_weights[s] = 0.0
                weight = weight_per_tier.get(feedback, 1)
                if feedback is not None:
                    print('Found feedback != none')
                sum_weights[s] += weight * math.log(1 + total_charge)

    result = {}
    for skill in opening_skills.split(','):
        if skill in sum_weights:
            result[skill] = sum_weights[skill]

    return result


###
### Functions to use for implementing each strategy
###
###############################################################################
def process_primitive(constructor):
    def process(value, query_item, row_data, script_manager=None):
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
        target_features = query_item.get('target_features', [])

        result = None
        if value is not None:
            result = constructor(value)

        return {target_features[0]['name']: result}

    return process


def process_composite(value, query_item, row_data, script_manager=None):
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
    target_features = query_item.get('target_features', [])

    for feature in target_features:
        if 'expression' not in feature or \
                'type' not in feature['expression'] or \
                'value' not in feature['expression']:
            raise ProcessException('Must define an expression with "type"\
and "value" for target feature %s' % (feature['name']))
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
                #for k, v in row_data.iteritems():
                    # if isinstance(v, basestring):
                    #     row_data[k] = v.encode('utf-8', errors='ignore')
                try:
                    value = expression_value % row_data
                    value = value.decode('utf8', 'ignore')
                except Exception, e:
                    logging.exception('Error when evaluate feature %s, expression_value: %s, expression_type: %s' % 
                            (feature['name'], expression_value, expression_type))
                    raise ProcessException('%s (feature: %s, expression: %s)' %
                                       (e, feature['name'], expression_value))
                if expression_type == 'string':
                    result[feature['name']] = value
                elif expression_type == 'newpython':
                    if script_manager:
                        try:
                            result[feature['name']] = script_manager.execute_function(expression_value, None, row_data)
                        except Exception, e:
                            logging.exception('Error when evaluate feature %s, value: %s, expression_type: %s' % 
                                (feature['name'], value, expression_type))
                            raise ProcessException('%s (expression: %s)' %
                                           (e, value))
                elif expression_type == 'python':
                    try:
                        result[feature['name']] = eval(value)
                    except Exception, e:
                        logging.exception('Error when evaluate feature %s, value: %s, expression_type: %s' %
                            (feature['name'], value, expression_type))
                        raise ProcessException('%s (expression: %s)' %
                                       (e, value))

                elif expression_type == 'readability':
                    if 'readability_type' not in feature['expression']:
                        raise ProcessException('''Must define an expression
with "readability_type" for target feature %s''' % (feature['name']))
                    r_type = feature['expression']['readability_type']
                    if r_type not in READABILITY_METHODS:
                        raise ProcessException('''Readability_type "%s" is
not defined for target feature %s''' % (r_type, feature['name']))
                    r_func = READABILITY_METHODS[r_type]
                    readability = Readability(value)
                    result[feature['name']] = getattr(readability, r_func)()
            except NameError as e:
                raise ProcessException('%s (expression: %s)' %
                                       (e, expression_value % row_data))

    return result


def process_json(value, query_item, row_data, script_manager=None):
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

    target_features = query_item.get('target_features', [])
    for feature in target_features:
        if 'jsonpath' not in feature:
            raise ProcessException('Target feature %s has no JSONPath'
                                   % feature['path'])
        path_result = jsonpath(data, feature['jsonpath'])

        if path_result is not False:
            # Found result in given path
            if 'key_path' in feature and 'value_path' in feature:
                # Treat as a dictionary
                keys = jsonpath(path_result[0], feature['key_path'])
                try:
                    values = map(float,
                                 jsonpath(path_result[0],
                                 feature['value_path']))
                except ValueError as e:
                    raise ProcessException(e)
                except TypeError as e:
                    raise ProcessException(e)
                if keys is not False and values is not False:
                    result[feature['name']] = dict(zip(keys, values))
                else:
                    result[feature['name']] = None
            elif len(path_result) > 1:
                # Multiple results from JSONPath
                result_list = filter(None, path_result)
                if feature.get('to_csv', False) is True:
                    result[feature['name']] = feature.get('delimiter',
                        feature.get('join', ',')).join(result_list)
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
    if isinstance(expression, str) or isinstance(expression, unicode):
        return EXPRESSION_RE.findall(expression)
    else:
        params = set()
        for subexpression in expression:
            params |= set(EXPRESSION_RE.findall(subexpression))
        return list(params)


PROCESS_STRATEGIES = {
    'identity': process_primitive(lambda x: x),
    'string': process_primitive(str),
    'float': process_primitive(float),
    'boolean': process_primitive(bool),
    'integer': process_primitive(int),
    'json': process_json,
    'composite': process_composite
}

READABILITY_METHODS = {
    'ari': 'ARI',
    'flesch_reading_ease': 'FleschReadingEase',
    'flesch_kincaid_grade_level': 'FleschKincaidGradeLevel',
    'gunning_fog_index': 'GunningFogIndex',
    'smog_index': 'SMOGIndex',
    'coleman_liau_index': 'ColemanLiauIndex',
    'lix': 'LIX',
    'rix': 'RIX',
}
