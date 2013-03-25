from flask.ext import restful
from flask.ext.restful import reqparse
from flask import request
from werkzeug.datastructures import FileStorage
from sqlalchemy import and_
from sqlalchemy.sql.expression import asc, desc

from api.decorators import render
from api import db, api
from api.utils import crossdomain, ERR_NO_SUCH_MODEL, odesk_error_response
from api.models import Model, Test, Data
from api.exceptions import MethodException
from api.tasks import train_model, run_test

from core.trainer.store import load_trainer
from core.trainer.trainer import Trainer
from core.trainer.config import FeatureModel
from core.importhandler.importhandler import ExtractionPlan, ImportHandler

get_parser = reqparse.RequestParser()
get_parser.add_argument('show', type=str)

page_parser = reqparse.RequestParser()
page_parser.add_argument('page', type=int)


model_parser = reqparse.RequestParser()
model_parser.add_argument('importhandler', type=str)
model_parser.add_argument('train_importhandler', type=str)
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

        GET parameters:
            * comparable - returns list of models that could be compared
            to each other. (They should be Trained and has successfull
            completed Test)
            * show - list of fields to return
        """
        parser = get_parser
        parser.add_argument('comparable', type=bool)

        param = parser.parse_args()
        comparable = param.get('comparable', False)
        fields = param.get('show', None)
        fields = fields.split(',') if fields else Model.__public__
        if comparable:
            # TODO: Look for models with completed tests
            models = Model.query.all()
        else:
            models = Model.query.all()
        return {'models': qs2list(models, fields)}

    @render(brief=False)
    def _details(self, model):
        """
        Gets Trained Model details
        """
        param = get_parser.parse_args()
        fields = param.get('show', None)
        fields = fields.split(',') if fields else Model.__public__
        return {'model': qs2list(model, fields)}

    @render()
    def _load_tests(self, model):
        """
        Gets list of Trained Model's tests
        """
        param = get_parser.parse_args()
        fields = param.get('show', None)
        fields = fields.split(',') if fields else Test.__public__
        tests = model.tests.all()
        return {'tests': qs2list(tests, fields)}

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

    @render(code=201)
    def post(self, model):
        """
        Adds new Trained Model
        """
        param = model_parser.parse_args()
        try:
            if not param['features']:
                return self._upload(model, param)
            else:
                return self._add(model, param)
        except Exception, exc:
            raise MethodException(exc.message)

    def _add(self, model, param):
        model = Model(model)
        model.train_importhandler = param['train_importhandler']
        model.importhandler = param['importhandler']
        model.features = param['features']

        feature_model = FeatureModel(model.features, is_file=False)
        trainer = Trainer(feature_model)
        plan = ExtractionPlan(model.train_importhandler, is_file=False)
        model.import_params = plan.input_params
        model.trainer = trainer

        db.session.add(model)
        db.session.commit()
        return {'model': model}

    def _upload(self, model, param):
        """
        Upload already trained Model
        """
        param = model_parser.parse_args()
        model = Model(model)
        trainer = load_trainer(param['trainer'])
        model.status = Model.STATUS_TRAINED
        model.set_trainer(trainer)
        model.importhandler = param['importhandler']
        plan = ExtractionPlan(model.importhandler, is_file=False)
        model.import_params = plan.input_params
        db.session.add(model)
        db.session.commit()
        return {'model': model}

    @render(code=200)
    def put(self, model, action=None):
        model = Model.query.filter(Model.name == model).one()
        if action is None:
            return self._edit(model)
        elif action == 'train':
            return self._train(model)

    def _edit(self, model):
        param = model_parser.parse_args()
        model.importhandler = param['importhandler']
        model.train_importhandler = param['train_importhandler']
        db.session.commit()
        return {'model': model}

    def _train(self, model):
        parser = populate_parser(model)
        parameters = parser.parse_args()

        model.status = Model.STATUS_QUEUED
        db.session.commit()

        train_model.delay(model, parameters)
        return {'model': model}


api.add_resource(Models, '/cloudml/b/v1/model/<regex("[\w\.]*"):model>',
                 '/cloudml/b/v1/model/<regex("[\w\.]+"):model>/<regex("[\w\.]+"):action>')


class Tests(restful.Resource):
    decorators = [crossdomain(origin='*')]

    @render()
    def get(self, model, test_name):
        test = db.session.query(Test).join(Model)\
            .filter(Model.name == model,
                    Test.name == test_name).one()
        param = get_parser.parse_args()
        fields = param.get('show', None)
        fields = fields.split(',') if fields else Test.__public__
        return {'test': qs2list(test, fields)}

    @render(code=204)
    def delete(self, model, test_name):
        test = db.session.query(Test).join(Model)\
            .filter(Model.name == model,
                    Test.name == test_name).one()
        test.delete()
        return '', 204

    @render(code=201)
    def post(self, model, test_name):
        model = Model.query.filter(Model.name == model,
                                   Model.status == Model.STATUS_TRAINED)\
            .first()

        parser = populate_parser(model)
        parameters = parser.parse_args()

        test = Test(model)
        test.status = Test.STATUS_QUEUED
        test.parameters = parameters
        db.session.add(test)
        db.session.commit()

        run_test.delay(test, test.model)

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


class CompareReport(restful.Resource):
    decorators = [crossdomain(origin='*')]

    @render(brief=False)
    def get(self):
        parser = reqparse.RequestParser()
        parser.add_argument('test1', type=str)
        parser.add_argument('test2', type=str)
        parser.add_argument('model1', type=str)
        parser.add_argument('model2', type=str)
        parameters = parser.parse_args()
        test1 = db.session.query(Test).join(Model)\
            .filter(Model.name == parameters['model1'],
                    Test.name == parameters['test1']).one()
        test2 = db.session.query(Test).join(Model)\
            .filter(Model.name == parameters['model2'],
                    Test.name == parameters['test2']).one()
        examples1 = db.session.query(Data).join(Test)\
            .filter(Data.test == test1)\
            .order_by(desc(Data.pred_label))[:10]
        examples2 = db.session.query(Data).join(Test)\
            .filter(Data.test == test2)\
            .order_by(desc(Data.pred_label))[:10]
        return {'test1': test1, 'test2': test2,
                'examples1': examples1, 'examples2': examples2}

api.add_resource(CompareReport, '/cloudml/b/v1/reports/compare')


def populate_parser(model):
    parser = reqparse.RequestParser()
    for param in model.import_params:
        parser.add_argument(param, type=str)
    return parser


def qs2list(obj, fields):
    from collections import Iterable

    def model2dict(model):
        data = {}
        for field in fields:
            if hasattr(model, field):
                data[field] = getattr(model, field)
            else:
                subfields = field.split('.')
                count = len(subfields)
                val = model
                el = data
                for i, subfield in enumerate(subfields):
                    if not subfield in el:
                        el[subfield] = {}
                    if hasattr(val, subfield):
                        val = getattr(val, subfield)
                        if i == count - 1:
                            el[subfield] = val
                    el = el[subfield]
        return data

    if isinstance(obj, Iterable):
        model_list = []
        for model in obj:
            model_list.append(model2dict(model))
        return model_list
    else:
        return model2dict(obj)
