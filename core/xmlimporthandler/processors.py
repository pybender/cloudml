from jsonpath import jsonpath
from sklearn.feature_extraction.readability import Readability

from exceptions import ProcessException


def process_key_value(key_path, value_path, value):
    # Treat as a dictionary
    keys = jsonpath(value, key_path)
    try:
        values = map(float,
                     jsonpath(value,
                     value_path))
    except ValueError as e:
        raise ProcessException(e)
    except TypeError as e:
        raise ProcessException(e)
    if keys is not False and values is not False:
        result = dict(zip(keys, values))
    else:
        result = None
    return result


def composite_string(expression_value, value, row_data):
    res = expression_value % dict(row_data)
    return res.decode('utf8', 'ignore')


def composite_python(expression_value, value, row_data):
    res = composite_string(expression_value, value, row_data)
    return eval(res)


def composite_readability(expression_value, value, r_type, row_data):
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
