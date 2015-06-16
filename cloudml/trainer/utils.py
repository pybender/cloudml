"""
This module gathers utility functions to for model training, evaluation, etc.
"""

# Authors: Ioannis Foukarakis <ifoukarakis@upwork.com>
#          Nikolay Melnik <nmelnik@upwork.com>


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


def parse_parameters(config, settings, brief=True):
    """
    Parse config parameters according to settings.
    """
    defaults = settings.get('defaults', {})
    source_params = defaults.copy()
    expected = settings.get('parameters', {})
    if not brief:
        expected = [p['name'] for p in expected]
    source_params.update(config.get('params', {}))
    return copy_expected(source_params, expected)


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
