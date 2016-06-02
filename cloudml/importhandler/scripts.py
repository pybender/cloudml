"""
Python script manager.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import logging

from utils import ParametrizedTemplate

# Context:
from processors import composite_string, composite_python, \
    composite_readability, process_key_value  # pylint: disable=W0611
from exceptions import ImportHandlerException, LocalScriptNotFoundException
from base import AmazonSettingsMixin
import boto3
import os.path

__all__ = ['ScriptManager', 'Script']


class Script(AmazonSettingsMixin):
    """
    Manages script entity in XML Import Handler
    """
    def __init__(self, config):
        self.text = config.text
        self.src = config.attrib.get("src", None)
        self.out_string = ''

    def _process_amazon_file(self):
        AMAZON_ACCESS_TOKEN, AMAZON_TOKEN_SECRET, \
            BUCKET_NAME = self.amazon_settings
        try:
            s3 = boto3.resource(
                's3',
                aws_access_key_id=AMAZON_ACCESS_TOKEN,
                aws_secret_access_key=AMAZON_TOKEN_SECRET)
            res = s3.Object(BUCKET_NAME, self.src).get()
            self.out_string = res["Body"].read(res["ContentLength"])

        except Exception as exc:
            raise ImportHandlerException("Error accessing file '{0}' on Amazon"
                                         ": {1}".format(self.src, exc.message))

    def _process_local_file(self):
        if os.path.isfile(self.src):
            with open(self.src, 'r') as fp:
                fs = fp.read()
                self.out_string = fs or ''
                fp.close()
        else:
            raise LocalScriptNotFoundException("Local file '{0}' not "
                                               "found".format(self.src))

    def get_script_str(self):
        if self.src:
            try:
                self._process_local_file()
            except LocalScriptNotFoundException as e:
                try:
                    self._process_amazon_file()
                except Exception as exc:
                    raise ImportHandlerException(
                        "{0}. Searching on Amazon: {1} ".format(
                            e.message, exc.message))
            except Exception as ex:
                raise ImportHandlerException("Error while accessing script "
                                             "'{0}': {1}".format(self.src,
                                                                 ex.message))
        elif self.text:
            self.out_string = self.text

        return self.out_string


class ScriptManager(object):
    """
    Manages and executes python scripts.
    """
    def __init__(self):
        self.data = ''
        self.context = {}

    def add_python(self, script):
        """
        Adds python methods to the script manager.
        """
        try:
            if script:
                eval(compile(script, "<str>", "exec"), self.context,
                     self.context)
        except Exception,  exc:
            raise ImportHandlerException(
                "Exception occurs while adding python script: {0}. {1}".format(
                    script[:250], exc))

    def execute_function(self, script, value,
                         row_data=None, local_vars={}):
        """
        Executes function and returns it's result.

        script: string
            python code to execute.
        value: any
            would be passed as #{value} parameter to the script
        row_data: dict
            script template parameters
        local_vars: dict
            execution context
        """
        def update_strings(val):
            if isinstance(val, basestring):
                return "'%s'" % val
            return val

        row_data = row_data or {}
        params = {'value': update_strings(value)}
        params.update(row_data)
        params.update(local_vars)
        text = ParametrizedTemplate(script).safe_substitute(params)
        try:
            return self._exec(text, local_vars)
        except Exception,  exc:
            raise ImportHandlerException(
                "Exception occurs while executing script: {0}. {1}".format(
                    text[:100], exc))

    def _exec(self, text, row_data=None):
        row_data = row_data or {}
        context = globals().copy()
        context.update(locals())
        context.update(prepare_context(row_data))
        context.update(self.context)

        try:
            return eval(text, context, context)
        except Exception,  exc:
            raise ImportHandlerException(
                "Exception occurs while executing script: {0}. {1}".format(
                    text[:100], exc))


def prepare_context(data):
    """
    Prepares context dictionary.
    Convertes defenitions like
        data['el.field1'] = val1
        data['el.field2'] = val2
    to object el in the context, where field1 = val1, field2 = val2

    data: dict
        dictionary of the context data

    >>> data = {'data.x1': 10, 'data.result.sum': 21}
    >>> data['data.result.metrics.extra'] = [1, 2, 3]
    >>> res = prepare_context(data)
    >>> el = res['data']
    >>> el.x1
    10
    >>> el.result.sum
    21
    >>> el.result.metrics.extra[0]
    1

    >>> data = {'data': 10, 'data.x': 3}
    >>> prepare_context(data)
    Traceback (most recent call last):
        ...
    ImportHandlerException: Can't set variable 'data' in \
the context twice. Keys are: data.x, data.

    >>> prepare_context({'data': 10, 'data.x': 3})
    Traceback (most recent call last):
        ...
    ImportHandlerException: Can't set variable 'data' in the \
context twice. Keys are: data.x, data.

    >>> prepare_context({'data.x': 10, 'data.x.y': 3})
    Traceback (most recent call last):
        ...
    ImportHandlerException: Can't create the x variable for \
data.x: element x already exist and equals The item (<class \
'cloudml.importhandler.scripts.ContextItem'>). Keys are: data.x.y, data.x.

    >>> prepare_context({'data.x': 10, 'data': 3})
    Traceback (most recent call last):
        ...
    ImportHandlerException: Can't create the 'data' variable \
in the context: element 'data' already exist and equals 3 \
(<type 'int'>). Keys are: data, data.x.

    >>> prepare_context({'data.x.y.a': 10, 'data.x.y': 3})
    Traceback (most recent call last):
        ...
    ImportHandlerException: Can't create the y variable for \
data.x.y.a: element y already exist and equals 3 (<type 'int'>). \
Keys are: data.x.y, data.x.y.a.

    >>> prepare_context({'': 10})
    Traceback (most recent call last):
        ...
    ImportHandlerException: Variable name couldn't be empty.

    >>> prepare_context({None: 10})
    Traceback (most recent call last):
        ...
    ImportHandlerException: Variable name couldn't be empty.

    >>> prepare_context({'x. ': 10})
    Traceback (most recent call last):
        ...
    ImportHandlerException: Field name couldn't be empty. Key is 'x. '.
    """
    class ContextItem(object):
        def __str__(self):
            return "The item"

    context = {}
    for key, val in data.iteritems():
        _check_name(key)

        splited = key.split('.')
        splitted_count = len(splited)
        if splitted_count == 1:  # simply key value here
            if key in context:
                raise ImportHandlerException(
                    "Can't set variable '{0}' in the context"
                    " twice. Keys are: {1}.".format(
                        key, ', '.join(data.keys())))
            context[key] = val
        else:
            # build obj using recursion
            for i in xrange(0, splitted_count):
                name = splited[i]
                _check_name(name, key)
                if i == 0:
                    if name in context:
                        obj = context[name]
                        if not isinstance(obj, ContextItem):
                            raise ImportHandlerException(
                                "Can't create the '{0}' variable in the "
                                "context: element '{0}' already exist and "
                                "equals {1} ({2}). Keys "
                                "are: {3}.".format(
                                    name, str(obj)[:20], type(obj),
                                    ', '.join(data.keys())))
                    else:
                        context[name] = ContextItem()
                        obj = context[name]
                elif i == splitted_count - 1:
                    # creating the class field
                    if hasattr(obj, name):
                        raise ImportHandlerException(
                            "Can't create the {0} variable for {3}:"
                            " element {0} already exist and equals "
                            "{1} ({2}). Keys are: {4}.".format(
                                name, str(obj)[:20], type(obj), key,
                                ', '.join(data.keys())))
                    setattr(obj, name, val)
                else:
                    if hasattr(obj, name):
                        obj = getattr(obj, name)
                        if not isinstance(obj, ContextItem):
                            raise ImportHandlerException(
                                "Can't create the {0} variable for {3}:"
                                " element {0} already exist and equals "
                                "{1} ({2}). Keys are: {4}.".format(
                                    name, str(obj)[:20], type(obj), key,
                                    ', '.join(data.keys())))
                    else:
                        item = ContextItem()
                        setattr(obj, name, item)
                        obj = item
    return context


def _check_name(name, key=None):
    def _raise():
        if key:
            raise ImportHandlerException(
                "Field name couldn't be "
                "empty. Key is '{}'.".format(key))
        else:
            raise ImportHandlerException(
                "Variable name couldn't be empty.")

    if name is None:
        _raise()

    name = name.strip(' \t\n\r')
    if not name:
        _raise()
