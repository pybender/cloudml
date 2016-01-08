"""
Classes for processing user input data.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import re

from exceptions import ImportHandlerException
from db import postgres_iter
from utils import process_primitive, process_date
from cloudml.utils import process_bool


__all__ = ['Input']


class Input(object):
    """
    Input parameter configuration.

    config: lxml.etree._Element
        parsed by lxml.objectify input definition tag.
    """
# TODO: int in documentation.
# We need use int or integer everywhere
    PROCESS_STRATEGIES = {
        'string': process_primitive(str),
        'float': process_primitive(float),
        'boolean': process_primitive(process_bool),
        'integer': process_primitive(int),
        'date': process_date
    }

    def __init__(self, config):
        self.name = config.get('name')
        self.type = config.get('type', 'string')
        # a regular expression that can be used to validate
        # input parameter value
        self.regex = config.get('regex')
        # formating instrunctions for the parameter
        self.format = config.get('format')
        if self.type == 'date' and not self.format:
            self.format = "%Y-%m-%d"

    def process_value(self, value):
        """
        Validates and converts input parameter to corresponding type.
        """
        if value is None:
            raise ImportHandlerException(
                'Input parameter %s is required' % self.name)

        strategy = self.PROCESS_STRATEGIES.get(self.type)
        if strategy is None:
            types = ", ".join(self.PROCESS_STRATEGIES.keys())
            raise ImportHandlerException('Type of the input parameter %s is \
invalid: %s. Choose one of %s' % (self.name, self.type, types))

        if self.regex:
            match = re.match(self.regex, value)
            if not match:
                raise ImportHandlerException("Value of the input parameter %s \
doesn't match to regular expression %s: %s" % (self.name, self.regex, value))
        try:
            return strategy(value, format=self.format)
        except ValueError:
            raise ImportHandlerException(
                'Value of the input parameter %s is invalid %s%s: %s' %
                (self.name, self.type,
                 " in format %s" % self.format if self.format else "",
                 value))
