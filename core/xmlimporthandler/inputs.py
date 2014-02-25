from exceptions import ImportHandlerException
from core.importhandler.db import postgres_iter


class BaseInput(object):
    """
    Base class for any type of the Input.
    """
    def __init__(self, config):
        self.config = config
        self.name = config.attrib['name']
        self.type = config.attrib.get('type')
        self.regexp = config.attrib.get('regexp')
        self.format = config.attrib.get('format')

    def set_value(self, value):
        self.validate(value)
        self.value = value

    def validate(self, value):
        # TODO: if self.regexp: add regexp validation
        return True


class StringInput(BaseInput):
    pass


class Input(object):
    INPUT_DICT = {
        'string': StringInput,
    }

    @classmethod
    def factory(cls, config):
        name = config.attrib.get('type')
        print "name", name
        if name in cls.INPUT_DICT:
            return cls.INPUT_DICT[name](config)
        return StringInput(config)
