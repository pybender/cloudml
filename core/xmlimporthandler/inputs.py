from exceptions import ImportHandlerException
from core.importhandler.db import postgres_iter
from utils import process_primitive


def process_date(value, format):
    return value


class Input(object):
    PROCESS_STRATEGIES = {
        'string': process_primitive(str),
        'float': process_primitive(float),
        'boolean': process_primitive(bool),
        'integer': process_primitive(int),
        'date': process_date
    }

    def __init__(self, config):
        self.name = config.get('name')
        self.type = config.get('type', 'string')
        # a regular expression that can be used to validate
        # input parameter value
        self.regexp = config.get('regexp')
        # formating instrunctions for the parameter
        self.format = config.get('format')

    def process_value(self, value):
        strategy = self.PROCESS_STRATEGIES.get(self.type)
        if strategy is None:
            types = ", ".join(self.PROCESS_STRATEGIES.keys())
            raise ImportHandlerException(
                'Type of the input %s is invalid: %s. Choose one of %s' %
                (self.name, self.type, types))

        return strategy(value, format=self.format)
