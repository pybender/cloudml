from flask.ext import restful
from flask.ext.restful import reqparse
from flask import request
from werkzeug.datastructures import FileStorage
from sqlalchemy import and_

from api.decorators import render
from api import db, api
from api.utils import crossdomain, ERR_NO_SUCH_MODEL, odesk_error_response
from api.models import Model, Test, Data

from core.trainer.store import load_trainer
from core.trainer.trainer import Trainer
from core.trainer.config import FeatureModel
from core.importhandler.importhandler import ExtractionPlan, ImportHandler


page_parser = reqparse.RequestParser()
page_parser.add_argument('page', type=int)


model_parser = reqparse.RequestParser()
model_parser.add_argument('importhandler', type=str)
model_parser.add_argument('features', type=str)
model_parser.add_argument('trainer', type=FileStorage, location='files')


class Models(restful.Resource):
    """
    Models API methods
    """
    decorators = [crossdomain(origin='*')]
    methods = ['GET', 'OPTIONS', 'PUT', 'POST']

    def get(self, model=None, action=None):
        if model == "":
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
        """
        Gets list of Trained Models
        """
        models = Model.query.all()
        found = models.count(Model.id)
        return {'models': models, 'found': found}

    @render(brief=False)
    def _details(self, model):
        """
        Gets Trained Model details
        """
        return {'model': model}

    @render()
    def _load_tests(self, model):
        """
        Gets list of Trained Model's tests
        """
        tests = model.tests.all()
        return {'model': model, 'tests': tests}

    def delete(self, model):
        """
        Deletes unused Trained Model
        """
        model = Model.query.filter(Model.name == model).first()
        if model is None:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        "Model %s doesn't exist" % model)
        model.delete()
        return '', 204

    def post(self, model):
        """
        Adds new Trained Model
        """
        param = model_parser.parse_args()
        if not param['features']:
            return self._upload(model, param)
        else:
            return self._add(model, param)

    @render(code=201)
    def _add(self, model, param):
        model = Model(model)
        model.importhandler = param['importhandler']
        model.features = param['features']
        feature_model = FeatureModel(model.features, is_file=False)
        trainer = Trainer(feature_model)
        plan = ExtractionPlan(model.importhandler, is_file=False)
        #train_handler = ImportHandler(plan, param)
        model.import_params = plan.input_params
        #trainer.train(train_handler)
        model.trainer = trainer
        db.session.add(model)
        db.session.commit()
        return {'model': model}

    @render(code=201)
    def _upload(self, model, param):
        """
        Upload new Trained Model
        """
        param = model_parser.parse_args()
        model = Model(model)
        trainer = load_trainer(param['trainer'])
        model.trainer = trainer
        model.set_weights(**trainer.get_weights())
        model.importhandler = param['importhandler']
        plan = ExtractionPlan(model.importhandler, is_file=False)
        model.import_params = plan.input_params
        db.session.add(model)
        db.session.commit()
        return {'model': model}

    @render(code=200)
    def put(self, model):
        param = model_parser.parse_args()
        model = Model.query.filter(Model.name == model).one()
        model.importhandler = param['importhandler']
        db.session.commit()
        return {'model': model}


api.add_resource(Models, '/cloudml/b/v1/model/<regex("[\w\.]*"):model>',
    '/cloudml/b/v1/model/<regex("[\w\.]+"):model>/<regex("[\w\.]+"):action>')


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
        test = db.session.query(Test).join(Model)\
            .filter(Model.name == model,
                    Test.name == test_name).one()
        return {'test': test, 'model': test.model}

    @render(code=204)
    def delete(self, model, test_name):
        test = db.session.query(Test).join(Model)\
            .filter(Model.name == model,
                    Test.name == test_name).one()
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
        plan = ExtractionPlan(model.importhandler, is_file=False)
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
        data = db.session.query(Data).join(Test).join(Model).\
            filter(and_(Model.name == model, Test.name == test_name,
                        Data.id == data_id)).one()
        return {'data': data}


class DataList(restful.Resource):
    decorators = [crossdomain(origin='*')]

    @render()
    def get(self, model, test_name):
        param = page_parser.parse_args()
        test = db.session.query(Test).join(Model)\
            .filter(Model.name == model,
                    Test.name == test_name).one()
        data_paginated = test.data.paginate(param['page'] or 1, 20, False)
        return {'model': test.model,
                'data': {'items': data_paginated.items,
                         'pages': data_paginated.pages,
                         'total': data_paginated.total,
                         'page': data_paginated.page,
                         'has_next': data_paginated.has_next,
                         'has_prev': data_paginated.has_prev,
                         'per_page': data_paginated.per_page},
                'test': test}

api.add_resource(DataList, '/cloudml/b/v1/model/<regex("[\w\.]+"):model>/test/<regex("[\w\.\-]+"):test_name>/data')
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
