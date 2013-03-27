from flask.ext import restful
from flask.ext.restful import reqparse
from werkzeug.datastructures import FileStorage
from sqlalchemy import and_
from sqlalchemy.sql.expression import asc, desc
from sqlalchemy.orm import undefer
from sqlalchemy import orm

from api.decorators import render
from api import db, api
from api.utils import crossdomain, ERR_NO_SUCH_MODEL, odesk_error_response
from api.models import Model, Test, Data
from api.exceptions import MethodException
from api.tasks import train_model, run_test

from core.trainer.store import load_trainer
from core.trainer.trainer import Trainer
from core.trainer.config import FeatureModel
from core.importhandler.importhandler import ExtractionPlan

get_parser = reqparse.RequestParser()
get_parser.add_argument('show', type=str)

page_parser = reqparse.RequestParser()
page_parser.add_argument('page', type=int)


model_parser = reqparse.RequestParser()
model_parser.add_argument('importhandler', type=str)
model_parser.add_argument('train_importhandler', type=str)
model_parser.add_argument('features', type=str)
model_parser.add_argument('trainer', type=FileStorage, location='files')


class BaseResource(restful.Resource):
    MODEL = None
    GET_ACTIONS = ()
    GET_PARAMS = (('show', str), )

    def get(self, model=None, action=None):
        if action:
            if action in self.GET_ACTIONS:
                return getattr(self, "_get_%s_action" % action)(model)
            else:
                return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                            "Invalid action: %s" % action)
        if model:
            return self._details(model)
        else:
            return self._list()

    def delete(self, model):
        """
        Deletes unused model
        """
        model = self.MODEL.query.filter(self.MODEL.name == model).first()
        if model is None:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        "Model %s doesn't exist" % model)
        model.delete()
        return '', 204

    @render()
    def _list(self, extra_params=()):
        """
        Gets list of models

        GET parameters:
            * show - list of fields to return
        """
        params = self._parse_parameters(extra_params + self.GET_PARAMS)
        fields = self._get_fields_to_show(params)
        opts = self._get_undefer_options(fields)
        models = self._get_list_query(params, opts)
        return {'models': qs2list(models, fields)}

    def _get_list_query(self, params, opts):
        return db.session.query(self.MODEL).options(*opts).all()

    @render()
    def _details(self, model, extra_params=()):
        """
        GET model by name
        """
        params = self._parse_parameters(extra_params + self.GET_PARAMS)
        fields = self._get_fields_to_show(params)
        opts = self._get_undefer_options(fields)
        model = db.session.query(self.MODEL).options(*opts)\
            .filter(self.MODEL.name == model)
        return {'model': qs2list(model.one(), fields)}

    def _get_fields_to_show(self, params):
        fields = params.get('show', None)
        return fields.split(',') if fields else Model.__public__

    def _parse_parameters(self, extra_params=()):
        parser = reqparse.RequestParser()
        for name, param_type in extra_params:
            parser.add_argument(name, type=param_type)
        return parser.parse_args()

    def _get_undefer_options(self, fields):
        def is_model_field(prop):
            return hasattr(self.MODEL, prop) and \
                type(getattr(self.MODEL, prop)) == \
                orm.attributes.InstrumentedAttribute
        opts = []
        for field in fields:
            if is_model_field(field):
                opts.append(undefer(field))
        return opts


class Models(BaseResource):
    """
    Models API methods
    """
    MODEL = Model
    GET_ACTIONS = ('tests', )
    decorators = [crossdomain(origin='*')]
    methods = ['GET', 'OPTIONS', 'PUT', 'POST']

    def _get_list_query(self, params, opts):
        comparable = params.get('comparable', False)
        # TODO: Look for models with completed tests if comparable
        return super(Models, self)._get_list_query(params, opts)

    @render()
    def _get_tests_action(self, model):
        """
        Gets list of Trained Model's tests
        """
        param = get_parser.parse_args()
        fields = param.get('show', None)
        fields = fields.split(',') if fields else Test.__public__
        tests = db.session.query(Test).join(Model).filter(Model.name == model).all()
        return {'tests': qs2list(tests, fields)}

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
        model.train_importhandler = param['importhandler']
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
        model.importhandler = param['importhandler'] or model.importhandler
        model.train_importhandler = param['train_importhandler'] \
            or model.train_importhandler
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
        param = get_parser.parse_args()
        fields = param.get('show', None)
        fields = fields.split(',') if fields else Data.__public__
        data = db.session.query(Data).join(Test).join(Model).\
            filter(and_(Model.name == model, Test.name == test_name,
                        Data.id == data_id)).one()
        return {'data': qs2list(data, fields)}


class DataList(restful.Resource):
    decorators = [crossdomain(origin='*')]

    @render()
    def get(self, model, test_name):
        param = page_parser.parse_args()
        test = db.session.query(Test).join(Model)\
            .filter(Model.name == model,
                    Test.name == test_name).one()
        data_paginated = test.data.paginate(param['page'] or 1, 20, False)
        param = get_parser.parse_args()
        fields = param.get('show', None)
        fields = fields.split(',') if fields else Data.__public__
        return {'data': {'items': qs2list(data_paginated.items, fields),
                         'pages': data_paginated.pages,
                         'total': data_paginated.total,
                         'page': data_paginated.page,
                         'has_next': data_paginated.has_next,
                         'has_prev': data_paginated.has_prev,
                         'per_page': data_paginated.per_page}}

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
                    if isinstance(val, Iterable) and subfield in val:
                        val = val[subfield]
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
