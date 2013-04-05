import logging
import json
from datetime import datetime
import pickle

from bson import Binary
from flask.ext.mongokit import Document

from api import db


@db.register
class Model(Document):
    STATUS_NEW = 'New'
    STATUS_QUEUED = 'Queued'
    STATUS_TRAINING = 'Training'
    STATUS_TRAINED = 'Trained'
    STATUS_ERROR = 'Error'

    __collection__ = 'models'
    structure = {
        'name': unicode,
        'status': basestring,
        'created_on': datetime,
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
    }
    required_fields = ['name', 'created_on', ]
    default_values = {'created_on': datetime.utcnow,
                      'status': STATUS_NEW,
                      'comparable': False}
    use_dot_notation = True

    def get_import_handler(self, parameters=None, is_test=False):
        from core.importhandler.importhandler import ExtractionPlan, \
            ImportHandler
        handler = json.dumps(self.importhandler if is_test
                             else self.train_importhandler)
        plan = ExtractionPlan(handler, is_file=False)
        handler = ImportHandler(plan, parameters)
        return handler

    # def run_test(self, parameters=True):
    #     test_handler = self.get_import_handler(parameters, is_test=True)
    #     metrics = self.trainer.test(test_handler)
    #     raw_data = self.trainer._raw_data
    #     self.trainer.clear_temp_data()
    #     return metrics, raw_data

    def set_trainer(self, trainer):
        self.trainer = Binary(pickle.dumps(trainer))
        self.target_variable = trainer._feature_model.target_variable

    def set_weights(self, positive, negative):
        from helpers.weights import calc_weights_css
        self.positive_weights = calc_weights_css(positive, 'green')
        self.negative_weights = calc_weights_css(negative, 'red')
        self.negative_weights.reverse()

    def __repr__(self):
        return '<Model %r>' % self.name


@db.register
class ImportHandler(Document):
    TYPE_DB = 'Db'
    TYPE_REQUEST = 'Request'
    __collection__ = 'handlers'
    structure = {
        'name': unicode,
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


# class Test(db.Document, Serializer):
#     __public__ = ('id', 'name', 'created_on', 'accuracy',
#                   'parameters', 'status', 'error')
#     __all_public__ = ('id', 'name', 'created_on', 'accuracy', 'parameters',
#                       'classes_set', 'metrics',
#                       'status', 'error', 'model_name')
#     STATUS_QUEUED = 'Queued'
#     STATUS_IN_PROGRESS = 'In Progress'
#     STATUS_COMPLETED = 'Completed'
#     STATUS_ERROR = 'Error'

#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(50))
#     status = db.Column(db.String(10), default=STATUS_IN_PROGRESS)
#     error = db.Column(db.Text)
#     created_on = db.Column(db.DateTime)
#     model_id = db.Column(db.Integer, db.ForeignKey('model.id'))

#     # Import params
#     parameters = db.Column(JSONEncodedDict)
#     data = db.relationship('Data', backref='test', lazy='dynamic')
#     classes_set = db.Column(JSONEncodedDict)

#     accuracy = db.Column(db.Float)
#     metrics = deferred(db.Column(JSONEncodedDict))

#     def __init__(self, model):
#         self.model_id = model.id
#         self.name = Test.generate_name(model)
#         self.created_on = datetime.now()

#     def __repr__(self):
#         return '<Test %r>' % self.name

#     @classmethod
#     def generate_name(cls, model, base_name='Test'):
#         count = model.tests.count()
#         return "%s-%s" % (base_name, count + 1)

#     @property
#     def data_count(self):
#         return self.data.count()

#     @property
#     def model_name(self):
#         return self.model.name


# class Data(db.Document, Serializer):
#     __public__ = ('id', 'created_on', 'data_input', 'label', 'pred_label')
#     __all_public__ = ('id', 'created_on', 'data_input',
#                       'weighted_data_input', 'label', 'pred_label')

#     id = db.Column(db.Integer, primary_key=True)
#     created_on = db.Column(db.DateTime)
#     data_input = deferred(db.Column(JSONEncodedDict))
#     weighted_data_input = deferred(db.Column(JSONEncodedDict))
#     label = db.Column(db.String(50))
#     pred_label = db.Column(db.String(50))
#     test_id = db.Column(db.Integer, db.ForeignKey('test.id'))
#     group_by_field = db.Column(db.String(250))

#     def __init__(self, data_input, test_id, weighted_data_input,
#                  label, pred_label):
#         self.data_input = data_input
#         self.weighted_data_input = weighted_data_input
#         self.created_on = datetime.now()
#         self.test_id = test_id
#         self.label = label
#         self.pred_label = pred_label

#     @classmethod
#     def loads_from_raw_data(cls, model, test, raw_data, labels,
# pred, group_by):
#         def decode(row):
#             for key, val in row.iteritems():
#                 try:
#                     if isinstance(val, basestring):
#                         row[key] = val.encode('ascii', 'ignore')
#                 except UnicodeDecodeError, exc:
#                     #logging.error('Error while decoding %s: %s', val, exc)
#                     row[key] = ""
#             return row

#         from helpers.weights import get_weighted_data
#         from itertools import izip
#         print "group_by", group_by
#         for row, label, pred in izip(raw_data, labels, pred):
#             row = decode(row)
#             weighted_data_input = get_weighted_data(model, row)
#             data = cls(row, test.id, weighted_data_input,
#                        str(label), str(pred))
#             data.group_by_field = row[group_by]
#             db.session.add(data)
#         db.session.commit()

#     @property
#     def target_variable(self):
#         return self.test.model.target_variable

#     @property
#     def title(self):
#         # TODO: hack
#         try:
#             return self.data_input['contractor.dev_profile_title']
#         except:
#             return 'No Title'
