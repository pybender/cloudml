from flask import Flask
from werkzeug.routing import BaseConverter
from flask.ext.mongokit import MongoKit
from flask.ext import restful
from celery import Celery

from flask.ext.sqlalchemy import SQLAlchemy





app = Flask(__name__)
app.config.from_object('api.config')

db = MongoKit(app)
db1 = SQLAlchemy(app)


celery = Celery('cloudml')
celery.add_defaults(lambda: app.config)
# celery = Celery(, broker=app.config['BROKER_URL'],
#                 backend=app.config['CELERY_RESULT_BACKEND'])
#celery.config_from_object('celeryconfig')
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
