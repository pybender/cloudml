from datetime import timedelta
from functools import update_wrapper, wraps
import json
import pickle
import logging
from time import time

from sqlalchemy.types import TypeDecorator, VARCHAR

from flask import make_response, request, current_app, jsonify
from api import app


ERR_INVALID_CONTENT_TYPE = 1000
ERR_NO_SUCH_MODEL = 1001
ERR_NO_MODELS = 1002
ERR_STORING_MODEL = 1003
ERR_LOADING_MODEL = 1004
ERR_INVALID_DATA = 1005


class JSONEncodedDict(TypeDecorator):
    """Represents an immutable structure as a json-encoded string.

    Usage::

        JSONEncodedDict(255)

    """

    impl = VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            try:
                value = json.loads(value)
            except ValueError, exc:
                logging.error('Error when loads %s: %s', value, exc)
                raise

        return value


class PickledValue(TypeDecorator):
    """Represents an structure as a pickled string.

    Usage::

        PickledValue(255)

    """

    impl = VARCHAR

    def process_bind_param(self, value, dialect):
        if value is not None:
            value = pickle.dumps(value)

        return value

    def process_result_value(self, value, dialect):
        if value is not None:
            value = pickle.loads(value)
        return value



def consumes(content_type=None):
    """
    Annotation that filters requests according to Content-type. If the
    request's content-type doesn't match the content type defined in the
    annotation, 405 Method not allowed is returned.

    Keyword arguments:
    content_type -- a string describing the content type that the decorated
                   function awaits for parsing.

    """
    def _consumes_decorator(view_func):
        def _decorator(*args, **kwargs):
            req_type = request.headers['Content-Type']
            if req_type != content_type:
                # If content type doesn't match.
                return odesk_error_response(405, ERR_INVALID_CONTENT_TYPE,
                                            'Invalid content-type',
                                            'Accepts only %s' % content_type)

            # Content type matches, return
            return view_func(*args, **kwargs)
        return wraps(view_func)(_decorator)
    return _consumes_decorator


def odesk_error_response(status, code, message, debug=None):
    """
    Creates a JSON error response that is compliant with https://sites.google.com/a/odesk.com/eng/Home/FunctionalSpecifications/webservices-error-handling-enhancements-frd

    Keyword arguments
    status -- The HTTP status code to return.
    code -- Internal application's error code.
    message -- A text describing the application's error.
    debug -- Additional debug information, to be added only if server is
             running on debug mode.
    """
    result = {}

    result = {'response': {
              'server_time': time(),
              'error': {'status': status, 'code': code,
                        'message': message}}}
    if app.debug:
        result['response']['error']['debug'] = debug

    response = jsonify(result)
    response.status_code = status
    response.headers.add('Content-type', 'application/json')
    response.headers.add('X-Odesk-Error-Code', code)
    response.headers.add('X-Odesk-Error-Message', message)
    return response


def crossdomain(origin=None, methods=None, headers=None,
                max_age=21600, attach_to_all=True,
                automatic_options=True):
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, basestring):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, basestring):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        def wrapped_function(*args, **kwargs):
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers

            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator
