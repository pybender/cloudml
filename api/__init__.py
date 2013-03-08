from flask import Flask
from werkzeug.routing import BaseConverter
from flask.ext.sqlalchemy import SQLAlchemy
from flask.ext import restful

app = Flask(__name__)
app.config.from_object('api.config')

db = SQLAlchemy(app)

api = restful.Api(app)


class RegExConverter(BaseConverter):
    """
    Converter that allows routing to specific functions according to given
    regular expression.

    """
    def __init__(self, url_map, *items):
        super(RegExConverter, self).__init__(url_map)
        self.regex = items[0]

app.url_map.converters['regex'] = RegExConverter


import models
import views
