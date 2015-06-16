# Authors: Ioannis Foukarakis <ifoukarakis@upwork.com>
#          Nikolay Melnik <nmelnik@upwork.com>


class BaseException(Exception):
    def __init__(self, message, Errors=None):
        super(BaseException, self).__init__(message)
        self.Errors = Errors


class SchemaException(BaseException):
    """
    Exception to be raised if there is an error parsing or using the
    configuration.
    """


class ItemParseException(BaseException):
    """
    Exception to be raised if there is an error parsing an item according to
    its feature type
    """


class InvalidTrainerFile(BaseException):
    """
    Exception to be raised if trainer could not be unpickled from file.
    """


class TransformerNotFound(BaseException):
    """
    Exception to be raised if predefined transormer could not be found.
    """


class EmptyDataException(BaseException):
    pass
