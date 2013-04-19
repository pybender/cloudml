import logging
import json
from datetime import datetime
import pickle

from bson import Binary
from flask.ext.mongokit import Document

from api import connection


@connection.register
class Model(Document):
    """
    Represents Model details and it's Tests.
    """
    STATUS_NEW = 'New'
    STATUS_QUEUED = 'Queued'
    STATUS_TRAINING = 'Training'
    STATUS_TRAINED = 'Trained'
    STATUS_ERROR = 'Error'

    __collection__ = 'models'
    structure = {
        'name': basestring,
        'status': basestring,
        'created_on': datetime,
        'updated_on': datetime,
        'error': basestring,

        'features': dict,
        # {
        #     "schema-name": unicode,
        #     "classifier": dict,
        #     "feature-types": list,
        #     "features": list
        # },
        'target_variable': unicode,

        # Import data to train and test options
        'import_params': list,
        'importhandler': dict,
        # {
        #     "target-schema": unicode,
        #     "datasource": list,
        #     "queries": list
        # },
        'train_importhandler': dict,
        #  {
        #     "target-schema": unicode,
        #     "datasource": list,
        #     "queries": list
        # },

        'trainer': None,
        'positive_weights': list,
        'negative_weights': list,
        'comparable': bool,
        #'tests': list,
    }
    required_fields = ['name', 'created_on', ]
    default_values = {'created_on': datetime.utcnow,
                      'updated_on': datetime.utcnow,
                      'status': STATUS_NEW,
                      'comparable': False, }
    use_dot_notation = True

    def get_import_handler(self, parameters=None, is_test=False):
        from core.importhandler.importhandler import ExtractionPlan, \
            ImportHandler
        handler = json.dumps(self.importhandler if is_test
                             else self.train_importhandler)
        plan = ExtractionPlan(handler, is_file=False)
        handler = ImportHandler(plan, parameters)
        return handler

    def run_test(self, parameters=True):
        trainer = pickle.loads(self.trainer)
        test_handler = self.get_import_handler(parameters, is_test=True)
        metrics = trainer.test(test_handler)
        raw_data = trainer._raw_data
        # TODO:
        #trainer.clear_temp_data()
        return metrics, raw_data

    def set_trainer(self, trainer):
        self.trainer = Binary(pickle.dumps(trainer))
        self.target_variable = trainer._feature_model.target_variable

    def set_weights(self, positive, negative):
        from helpers.weights import calc_weights_css
        self.positive_weights = calc_weights_css(positive, 'green')
        self.negative_weights = calc_weights_css(negative, 'red')
        self.negative_weights.reverse()


@connection.register
class Test(Document):
    STATUS_QUEUED = 'Queued'
    STATUS_IN_PROGRESS = 'In Progress'
    STATUS_COMPLETED = 'Completed'
    STATUS_ERROR = 'Error'

    __collection__ = 'tests'
    structure = {
        'name': basestring,
        'model_name': basestring,
        'status': basestring,
        'error': basestring,
        'created_on': datetime,
        'updated_on': datetime,
        'data': dict,
        'examples_count': int,
        'parameters': dict,
        'classes_set': list,
        'accuracy': float,
        'metrics': dict,
        'model': Model,
        # Raw test data
        #'examples': [TestExample ],
    }
    required_fields = ['name', 'created_on', 'updated_on',
                       'status']
    default_values = {'created_on': datetime.utcnow,
                      'updated_on': datetime.utcnow,
                      'status': STATUS_QUEUED}
    use_dot_notation = True
    use_autorefs = True

    @classmethod
    def generate_name(cls, model, base_name='Test'):
        count = model.tests.count()
        return "%s-%s" % (base_name, count + 1)

    @property
    def data_count(self):
        return self.data.count()

    @property
    def model_name(self):
        return self.model.name


@connection.register
class TestExample(Document):
    __collection__ = 'example'

    structure = {
        'created_on': datetime,
        'data_input': dict,
        'weighted_data_input': dict,

        'name': unicode,
        'label': unicode,
        'pred_label': unicode,
        'test': Test,

        'test_name': basestring,
        'model_name': basestring,
    }
    use_autorefs = True
    default_values = {'created_on': datetime.utcnow}
    required_fields = ['created_on', ]


@connection.register
class ImportHandler(Document):
    TYPE_DB = 'Db'
    TYPE_REQUEST = 'Request'
    __collection__ = 'handlers'
    structure = {
        'name': basestring,
        'type': basestring,
        'created_on': datetime,
        'updated_on': datetime,
        'data': dict,
    }
    required_fields = ['name', 'created_on', 'updated_on', ]
    default_values = {'created_on': datetime.utcnow,
                      'updated_on': datetime.utcnow,
                      'type': TYPE_DB}
    use_dot_notation = True

    def __repr__(self):
        return '<Import Handler %r>' % self.name
