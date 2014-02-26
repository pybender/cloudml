from exceptions import ImportHandlerException
from core.importhandler.db import postgres_iter


class BaseInput(object):
    """
    Base class for any type of the Input.
    """
    def __init__(self, config):
        self.name = config.attrib['name']
        self.type = config.get('type')
        self.regexp = config.get('regexp')
        self.format = config.get('format')

    def process_value(self, value):
        return value


class StringInput(BaseInput):
    pass


class Input(object):
    INPUT_DICT = {
        'string': StringInput,
    }

    @classmethod
    def factory(cls, config):
        name = config.attrib.get('type')
        if name in cls.INPUT_DICT:
            return cls.INPUT_DICT[name](config)
        return StringInput(config)
