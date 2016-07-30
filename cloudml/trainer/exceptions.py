# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>


class BaseTrainerException(Exception):
    def __init__(self, message, errors=None):
        super(BaseTrainerException, self).__init__(message)
        self.errors = errors


class SchemaException(BaseTrainerException):
    """
    Exception to be raised if there is an error parsing or using the
    configuration.
    """


class ItemParseException(BaseTrainerException):
    """
    Exception to be raised if there is an error parsing an item according to
    its feature type
    """


class ItemParseIgnore(BaseTrainerException):
    """
    Exception to be raised if item is ignored
    """


class InvalidTrainerFile(BaseTrainerException):
    """
    Exception to be raised if trainer could not be unpickled from file.
    """


class TransformerNotFound(BaseTrainerException):
    """
    Exception to be raised if predefined transormer could not be found.
    """


class EmptyDataException(BaseTrainerException):
    pass
