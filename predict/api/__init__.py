import logging

from flask import Flask
from werkzeug.routing import BaseConverter

from load_utils import load_models, load_import_handlers


app = Flask(__name__)
app.config.from_object('api.config')


class RegExConverter(BaseConverter):
    """
    Converter that allows routing to specific functions according to given
    regular expression.

    """
    def __init__(self, url_map, *items):
        super(RegExConverter, self).__init__(url_map)
        self.regex = items[0]

app.url_map.converters['regex'] = RegExConverter

app.models = {}
app.import_handlers = {}

logging_level = logging.INFO
logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
        level=logging_level)

load_models(app)
load_import_handlers(app)

import views
