class ItemParseException(Exception):
    """
    Exception to be raised if there is an error parsing an item according to
    its feature type

    """

    def __init__(self, message, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        # Now for your custom code...
        self.Errors = Errors


class InvalidTrainerFile(Exception):
    """
    Exception to be raised if trainer could not be unpickled from file.
    """
    pass


class TransformerNotFound(Exception):
    """
    Exception to be raised if predefined transormer could not be found.
    """
    pass
