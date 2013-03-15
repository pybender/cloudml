import json
from flask import request

from api import app
from api.serialization import BriefDetailsEncoder, FullDetailsEncoder


def render(brief=True, code=200):
    def wrap(func, *args, **kwargs):
        def wrapper(self, *args, **kwargs):
            context = func(self, *args, **kwargs)
            encoder = BriefDetailsEncoder if brief else FullDetailsEncoder
            resp = json.dumps(context, cls=encoder,
                              indent=None if request.is_xhr else 2)
            return app.response_class(resp, mimetype='application/json'), code
        return wrapper
    return wrap
