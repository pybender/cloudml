"""
Methods that would be avaiable in python scripts.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import sys
import json
import re
import math
import datetime
import logging
from cloudml.jsonpath import jsonpath
from sklearn.feature_extraction.readability import Readability

from exceptions import ProcessException

reload(sys)
sys.setdefaultencoding('utf-8')


def process_key_value(key_path, value_path, value):  # pragma: no cover
    """
    Method is obsolete, use jsonpath, key_path, value_path parameters
    of the field.
    """
    # Treat as a dictionary
    keys = jsonpath(value, key_path)
    try:
        values = map(float, jsonpath(value, value_path))
    except ValueError as e:
        raise ProcessException(e)
    except TypeError as e:
        raise ProcessException(e)
    if keys is not False and values is not False:
        result = dict(zip(keys, values))
    else:
        result = None
    return result


def composite_string(expression_value, value, row_data):  # pragma: no cover
    res = expression_value % dict(row_data)
    return res.decode('utf8', 'ignore')


def composite_python(expression_value, value, row_data):  # pragma: no cover
    res = composite_string(expression_value, value, row_data)
    try:
        return eval(res)
    except Exception:
        logging.exception(
            'Expression template %s, expression %s' % (expression_value, res))
        raise


def composite_readability(expression_value, value,
                          r_type, row_data):  # pragma: no cover
    res = composite_string(expression_value, value, row_data)
    if r_type not in READABILITY_METHODS:
        raise Exception('Readability_type "%s" is not defined' % r_type)
    r_func = READABILITY_METHODS[r_type]
    readability = Readability(res)
    return getattr(readability, r_func)()


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
