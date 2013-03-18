import json
from flask import request
from sqlalchemy.orm.exc import NoResultFound

from api import app
from api.utils import ERR_NO_SUCH_MODEL, odesk_error_response
from api.serialization import BriefDetailsEncoder, FullDetailsEncoder


def render(brief=True, code=200):
    def wrap(func, *args, **kwargs):
        def wrapper(self, *args, **kwargs):
            try:
                context = func(self, *args, **kwargs)
                encoder = BriefDetailsEncoder if brief else FullDetailsEncoder
                resp = json.dumps(context, cls=encoder,
                                  indent=None if request.is_xhr else 2)
                return app.response_class(resp, mimetype='application/json'), code
            except NoResultFound:
                return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                            'Model doesn\'t exist')
        return wrapper
    return wrap
