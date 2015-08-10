# Author: Nikolay Melnik <nmelnik@upwork.com>
import logging
import os


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


def determine_data_format(filepath):
    try:
        format = os.path.splitext(filepath)[1][1:]
    except Exception:
        raise
        logging.warning("Could not determine input data file format."
                        "'json' would be used.")
        return 'json'
    if format not in ('json', 'csv'):
        logging.warning("Input data file format is invalid {0}. "
                        "Trying to parse it as 'json'")
        return 'json'
    return format


def isfloat(x):
    """
    >>> isfloat('1.5')
    True
    >>> isfloat('5')
    True
    >>> isfloat('5,0')
    False
    """
    try:
        a = float(x)
    except:
        return False
    else:
        return True


def isint(x):
    """
    >>> isint('1.5')
    False
    >>> isint('5')
    True
    >>> isint('other')
    False
    """
    try:
        a = float(x)
        b = int(a)
    except:
        return False
    else:
        return a == b
