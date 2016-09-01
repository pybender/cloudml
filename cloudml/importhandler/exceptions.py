"""
Custom XML Import Handler exceptions.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>
from cloudml import ChainedException


class ProcessException(ChainedException):
    """
    Exception to be raised in case there's a problem processing a feature.

    """
    def __init__(self, message, chain=None, column=None, errors=None):
        # Call the base class constructor with the parameters it needs
        super(ProcessException, self).__init__(message, chain)
        self._column = column
        self.errors = errors


class ImportHandlerException(ChainedException):
    def __init__(self, message, chain=None, errors=None):
        # Call the base class constructor with the parameters it needs
        super(ImportHandlerException, self).__init__(message, chain)
        self.errors = errors


class LocalScriptNotFoundException(ChainedException):
    def __init__(self, message, chain=None, errors=None):
        # Call the base class constructor with the parameters it needs
        super(LocalScriptNotFoundException, self).__init__(message, chain)
        self.errors = errors
