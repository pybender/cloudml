"""
Custom XML Import Handler exceptions.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>


class ProcessException(Exception):
    """
    Exception to be raised in case there's a problem processing a feature.

    """
    def __init__(self, message, column=None, errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        self._column = column
        self.errors = errors


class ImportHandlerException(Exception):
    def __init__(self, message, errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        self.errors = errors


class LocalScriptNotFoundException(Exception):
    def __init__(self, message, errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        self.errors = errors
