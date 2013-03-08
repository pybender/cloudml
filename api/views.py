from datetime import datetime
import json
import pickle

from flask import render_template, url_for, redirect
from flask import request, jsonify
from flask.ext import restful
from flask.views import MethodView
from flask.ext.restful import reqparse

from trainer.store import load_trainer
from importhandler.importhandler import ExtractionPlan, ImportHandler

from api import app, db, api
from api.models import Model, Test, Data
from api.utils import crossdomain, odesk_error_response, consumes, SWJsonify


model_parser = reqparse.RequestParser()
model_parser.add_argument('name', type=str)


ERR_INVALID_CONTENT_TYPE = 1000
ERR_NO_SUCH_MODEL = 1001
ERR_NO_MODELS = 1002
ERR_STORING_MODEL = 1003
ERR_LOADING_MODEL = 1004
ERR_INVALID_DATA = 1005


class Models(restful.Resource):
    decorators = [crossdomain(origin='*')]

    def get(self, model):
        model = Model.query.filter(Model.name == model).first()
        if model is None:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        "Model %s doesn't exist" % model)
        #import pdb; pdb.set_trace()
        tests = model.tests.all()
        return SWJsonify({'model': model, 'tests': tests})

    def delete(self, model):
        model = Model.query.filter(Model.name == model).first()
        if model is None:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        "Model %s doesn't exist" % model)
        model.delete()
        return '', 204

    def post(self, model):
        file = request.files['file']
        import_handler_local = request.files['import_handler_local']
        model = Model(model)
        trainer = load_trainer(file)
        model.trainer = trainer
        model.import_handler = import_handler_local.read()
        plan = ExtractionPlan(model.import_handler, is_file=False)
        model.import_params = plan.input_params
        db.session.add(model)
        db.session.commit()
        return SWJsonify({'model': model}), 201


class ModelList(restful.Resource):
    decorators = [crossdomain(origin='*')]

    def get(self):
        models = Model.query.all()
        return SWJsonify({'models': models})

api.add_resource(ModelList, '/cloudml/b/v1/model')
api.add_resource(Models, '/cloudml/b/v1/model/<regex("[\w\.]+"):model>')


class Tests(restful.Resource):
    decorators = [crossdomain(origin='*')]

    def get(self, model, test_name):
        # TODO:
        model = Model.query.filter(Model.name == model).first()
        test = model.tests.filter(Test.name == test_name).first()
        if test is None:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        'Test %s doesn\'t exist' % test_name)
        return SWJsonify({'test': test, 'model': model}, all_fields=True)

    def delete(self, model, test_name):
        test = Test.query.filter(Test.name == test_name).first()
        if test is None:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        'Test %s doesn\'t exist' % test_name)
        test.delete()
        return '', 204

    def post(self, model, test_name):
        # TODO: save parameters to Test model
        test_parser = reqparse.RequestParser()
        test_parser.add_argument('start', type=str)
        test_parser.add_argument('end', type=str)
        param = test_parser.parse_args()
        model = Model.query.filter(Model.name == model).first()
        count = model.tests.count()
        test = Test("%s-%s" % (test_name, count + 1))
        test.model_id = model.id
        test.parameters = param
        plan = ExtractionPlan(model.import_handler, is_file=False)
        test_handler = ImportHandler(plan, param)

        metrics = model.trainer.test(test_handler)
        raw_data = model.trainer._raw_data
        test.accuracy = metrics.accuracy
        metrics_dict = metrics.get_metrics_dict()

        confusion_matrix = metrics_dict['confusion_matrix']
        confusion_matrix_ex = []
        for i, val in enumerate(metrics.classes_set):
            confusion_matrix_ex.append((val, confusion_matrix[i]))
        metrics_dict['confusion_matrix'] = confusion_matrix_ex

        test.metrics = metrics_dict
        test.classes_set = list(metrics.classes_set)
        db.session.add(test)
        db.session.commit()
        # store test data in db
        Data.loads_from_raw_data(test, raw_data)

        return SWJsonify({'test': test}), 201

api.add_resource(Tests, '/cloudml/b/v1/model/<regex("[\w\.]+"):model>/test/<regex("[\w\.\-]+"):test_name>')


class DatalList(restful.Resource):
    decorators = [crossdomain(origin='*')]

    def get(self, model, test_name):
        test = Test.query.filter(Test.name == test_name).first()
        data = test.data.all()
        return SWJsonify({'data': data})

api.add_resource(DatalList, '/cloudml/b/v1/model/<regex("[\w\.]+"):model>/test/<regex("[\w\.\-]+"):test_name>/data')

# @app.route('/cloudml/b/v1/model/<regex("[\w\.]+"):model>/evaluate',
#            methods=['POST'])
# @consumes('application/json')
# def evaluate(model):
#     """
#     Evaluates the given data and returns the probabilities for possible
#     classes.

#     Keyword arguments:
#     model -- The id of the model to use for evaluating the data.

#     """
#     global models
#     logging.info('Request to evaluate using model %s' % model)
#     if model not in models:
#         return odesk_error_response(404, ERR_NO_SUCH_MODEL,
#                                     'Model %s doesn\'t exist' % model)

#     trainer = models[model]
#     data = request.json
#     result = trainer.predict(data)
#     response = {'probabilities': []}
#     count = 0
#     for item in result['probs']:
#         response['probabilities'].append({'item': count,
#                                           'probs': item.tolist()})
#         count += 1
#     return jsonify(response)
