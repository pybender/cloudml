"""
This module gathers utility functions to for model training, evaluation, etc.
"""

# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>


def copy_expected(source_params, expected):
    """
    Filters dictionary of source_params to include only keys in expected.

    source_params: dict
        initial dictionary to be filtered
    expected: list
        keyword arguments expected to be present in the resulting dictionary
    """
    result = {}
    for param in expected:
        if param in source_params:
            result[param] = source_params[param]

    return result


def parse_parameters(config, settings, process_range_params=True):
    """
    Parse config parameters according to settings.
    """
    defaults = settings.get('defaults', {})
    source_params = defaults.copy()
    expected = settings.get('parameters', {})
    source_params.update(config.get('params', {}))
    expected = [item['name'] for item in expected]
    params = copy_expected(source_params, expected)

    if process_range_params:
        # process range params
        for param_name, param in params.copy().iteritems():
            if param_name.endswith('_min'):
                params.pop(param_name)
                param_name = param_name.replace('_min', '')
                param_max = params.pop(param_name + '_max')
                params[param_name] = (param, param_max)
    return params


def set_defaults(original, params):
    """Set defaults dict from default values in parameters"""
    for p in params:
        if 'default' in p and not p['name'] in original:
            original[p['name']] = p['default']
    return original


def set_params_defaults(params, defaults):
    """Updated parameters config with default values from defaults dict"""
    updated_defaults = []
    for p in params:
        cp = p.copy()
        if cp['name'] in defaults:
            cp['default'] = defaults[cp['name']]
        updated_defaults.append(cp)
    return updated_defaults


def is_empty(var):
    """
    Returns true if item is None or has a length of zero (if this item has
    a length).

    var: string
        the item to check if is empty.
    """

    if var is None:
        return True

    try:
        if len(var) == 0:
            return True
    except TypeError:
        # Item has no length (i.e. integer, float)
        pass

    return False


def float_or_int(value):
    """
    >>> float_or_int(1)
    1
    >>> float_or_int(1.5)
    1.5
    >>> float_or_int("1")
    1
    >>> float_or_int('1.5')
    1.5
    >>> float_or_int("s")
    Traceback (most recent call last):
    ...
    ValueError: could not convert string to float: s
    """
    if isinstance(value, (int, float)):
        return value
    value = float(value)
    if int(value) == value:
        value = int(value)
    return value
