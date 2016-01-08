""" Uitility methods for processing data """

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import logging
import os

import coloredlogs


def process_bool(val=None):
    if val is None:
        return False
    if isinstance(val, basestring):
        try:
            from distutils.util import strtobool
            return bool(strtobool(val))
        except ValueError:
            return False
    else:
        return bool(val)


def init_logging(debug):
    logging_level = logging.INFO
    if debug is True:
        logging_level = logging.DEBUG
    logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
                        level=logging_level)
    logger = logging.getLogger()
    logger.setLevel(logging_level)
    coloredlogs.install(level=logging.DEBUG)


def determine_data_format(filepath):
    try:
        file_format = os.path.splitext(filepath)[1][1:]
    except:
        logging.warning("Could not determine input data file format."
                        "'json' would be used.")
        return 'json'
    if file_format not in ('json', 'csv'):
        logging.warning("Input data file format is invalid {0}. "
                        "Trying to parse it as 'json'".format(file_format))
        return 'json'
    return file_format


def isfloat(value):
    """
    >>> isfloat('1.5')
    True
    >>> isfloat('5')
    True
    >>> isfloat('5,0')
    False
    """
    try:
        float(value)
    except:
        return False
    else:
        return True


def isint(value):
    """
    >>> isint('1.5')
    False
    >>> isint('5')
    True
    >>> isint('other')
    False
    """
    try:
        float_value = float(value)
        int_value = int(float_value)
    except:
        return False
    else:
        return float_value == int_value
