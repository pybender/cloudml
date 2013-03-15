from flask.ext import restful
from flask.ext.restful import reqparse
from flask import request

from api.decorators import render
from api import db, api
from api.utils import crossdomain, ERR_NO_SUCH_MODEL, odesk_error_response
from api.models import Model, Test, Data

from core.trainer.store import load_trainer
from core.trainer.trainer import Trainer
from core.trainer.config import FeatureModel
from core.importhandler.importhandler import ExtractionPlan, ImportHandler

model_parser = reqparse.RequestParser()
model_parser.add_argument('name', type=str)

page_parser = reqparse.RequestParser()
page_parser.add_argument('page', type=int)


class Models(restful.Resource):
    decorators = [crossdomain(origin='*')]

    def get(self, model=None, action=None):
        if model is None:
            return self._list()
        else:
            model = Model.query.filter(Model.name == model).first()
            if model is None:
                return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                            "Model %s doesn't exist" % model)
            if action == 'tests':
                return self._load_tests(model)
            else:
                return self._details(model)

    @render()
    def _list(self):
        models = Model.query.all()
        found = models.count(Model.id)
        return {'models': models, 'found': found}

    @render(brief=False)
    def _details(self, model):
        return {'model': model}

    @render()
    def _load_tests(self, model):
        tests = model.tests.all()
        return {'model': model, 'tests': tests}

    def delete(self, model):
        model = Model.query.filter(Model.name == model).first()
        if model is None:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        "Model %s doesn't exist" % model)
        model.delete()
        return '', 204

    @render(code=201)
    def post(self, model):
        file = request.files['file']
        import_handler_local = request.files['import_handler_local']
        features = request.files['features']
        model = Model(model)
        trainer = load_trainer(file)
        model.trainer = trainer
        model.set_weights(**trainer.get_weights())
        model.features = features.read()
        model.import_handler = import_handler_local.read()
        plan = ExtractionPlan(model.import_handler, is_file=False)
        model.import_params = plan.input_params
        db.session.add(model)
        db.session.commit()
        return {'model': model}

    def put(self, model):
        model = Model.query.filter(Model.name == model).one()
        return {'model': model}

api.add_resource(Models, '/cloudml/b/v1/model')
api.add_resource(Models, '/cloudml/b/v1/model/<regex("[\w\.]+"):model>')
api.add_resource(Models, '/cloudml/b/v1/model/<regex("[\w\.]+"):model>/<regex("[\w\.]+"):action>')


class Train(restful.Resource):
    decorators = [crossdomain(origin='*')]

    @render(code=201)
    def post(self, model):
        import_handler_local = request.files['import_handler_local']
        features = request.files['features']
        model = Model(model)
        model = FeatureModel(args.path)
        trainer = Trainer(model)
        plan = ExtractionPlan(model.import_handler, is_file=False)
        test_handler = ImportHandler(plan, param)
        trainer.train(train_handler)
        model.trainer = trainer
        model.set_weights(**trainer.get_weights())
        model.features = features.read()
        model.import_handler = import_handler_local.read()
        plan = ExtractionPlan(model.import_handler, is_file=False)
        model.import_params = plan.input_params
        db.session.add(model)
        db.session.commit()
        return {'model': model}

api.add_resource(Train, '/cloudml/b/v1/model/train/<regex("[\w\.]+"):model>')


class Tests(restful.Resource):
    decorators = [crossdomain(origin='*')]

    @render(brief=False)
    def get(self, model, test_name):
        # TODO:
        model = Model.query.filter(Model.name == model).first()
        test = model.tests.filter(Test.name == test_name).first()
        if test is None:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        'Test %s doesn\'t exist' % test_name)
        return {'test': test, 'model': model}

    def delete(self, model, test_name):
        test = Test.query.filter(Test.name == test_name).first()
        if test is None:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        'Test %s doesn\'t exist' % test_name)
        test.delete()
        return '', 204

    @render(code=201)
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
        Data.loads_from_raw_data(model, test, raw_data)

        return {'test': test}

api.add_resource(Tests, '/cloudml/b/v1/model/<regex("[\w\.]+"):model>/test/<regex("[\w\.\-]+"):test_name>')


class Datas(restful.Resource):
    decorators = [crossdomain(origin='*')]

    @render(brief=False)
    def get(self, model, test_name, data_id):
        # TODO:
        model = Model.query.filter(Model.name == model).first()
        test = model.tests.filter(Test.name == test_name).first()
        data = test.data.filter(Data.id == data_id).first()
        if test is None:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        'Test %s doesn\'t exist' % test_name)
        return {'test': test, 'model': model, 'data': data}


class DatalList(restful.Resource):
    decorators = [crossdomain(origin='*')]

    @render()
    def get(self, model, test_name):
        param = page_parser.parse_args()
        model = Model.query.filter(Model.name == model).first()
        test = model.tests.filter(Test.name == test_name).first()
        data_paginated = test.data.paginate(param['page'], 20, False)
        return {'model': model,
                          'data': {'items': data_paginated.items,
                                   'pages': data_paginated.pages,
                                   'total': data_paginated.total,
                                   'page': data_paginated.page,
                                   'has_next': data_paginated.has_next,
                                   'has_prev': data_paginated.has_prev,
                                   'per_page': data_paginated.per_page},
                          'test': test}

api.add_resource(DatalList, '/cloudml/b/v1/model/<regex("[\w\.]+"):model>/test/<regex("[\w\.\-]+"):test_name>/data')
api.add_resource(Datas,
 '/cloudml/b/v1/model/<regex("[\w\.]+"):model>/test/<regex("[\w\.\-]+"):test_name>/data/<regex("[\w\.\-]+"):data_id>')

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
