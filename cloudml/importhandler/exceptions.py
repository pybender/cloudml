"""
Custom XML Import Handler exceptions.
"""

# Author: Nikolay Melnik <nmelnik@upwork.com>


class ProcessException(Exception):
    """
    Exception to be raised in case there's a problem processing a feature.

    """
    def __init__(self, message, column=None, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        self._column = column
        self.Errors = Errors


class ImportHandlerException(Exception):
    def __init__(self, message, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        self.Errors = Errors


class LocalScriptNotFoundException(Exception):
    def __init__(self, message, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        self.Errors = Errors
