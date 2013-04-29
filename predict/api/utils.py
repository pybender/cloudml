from time import time

from flask import make_response, request, current_app, jsonify


ERR_NO_SUCH_MODEL = 1001
ERR_NO_SUCH_IMPORT_HANDLER = 1002
ERR_PREDICT = 1003
ERR_NO_MODELS = 1004

def odesk_error_response(status, code, message, debug=None, traceback=None):
    """
    Creates a JSON error response that is compliant with
    https://sites.google.com/a/odesk.com/eng/Home/FunctionalSpecifications/webservices-error-handling-enhancements-frd

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
                        'message': message, 'traceback': traceback}}}
    if current_app.debug:
        result['response']['error']['debug'] = current_app.debug

    response = jsonify(result)
    response.status_code = status
    response.headers.add('Content-type', 'application/json')
    response.headers.add('X-Odesk-Error-Code', code)
    response.headers.add('X-Odesk-Error-Message', message)
    return response
