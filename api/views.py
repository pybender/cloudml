from flask.ext import restful
from flask import request
from flask.ext.restful import reqparse
from werkzeug.datastructures import FileStorage
from sqlalchemy import and_
from sqlalchemy.sql.expression import asc, desc
from sqlalchemy.orm import undefer
from sqlalchemy import orm

from api.decorators import render
from api import db, api
from api.utils import crossdomain, ERR_NO_SUCH_MODEL, odesk_error_response
from api.models import Model, Test, Data, ImportHandler
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
    OBJECT_NAME = 'model'
    NEED_PAGING = False
    GET_PARAMS = (('show', str), )
    PAGING_PARAMS = (('page', int), )

    def get(self, action=None, **kwargs):
        if action:
            if action in self.GET_ACTIONS:
                return getattr(self, "_get_%s_action" % action)(**kwargs)
            else:
                return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                            "Invalid action: %s" % action)
        if self._is_list_method(**kwargs):
            return self._list(**kwargs)
        else:
            return self._details(**kwargs)

    @render(code=201)
    def post(self, name):
        """
        Adds new model
        """
        parser = self.get_model_parser()
        params = parser.parse_args()

        obj = self.MODEL()
        obj.name = name
        self._fill_post_data(obj, params)

        db.session.add(obj)
        db.session.commit()
        return {self.OBJECT_NAME: obj}

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
    def _list(self, extra_params=(), **kwargs):
        """
        Gets list of models

        GET parameters:
            * show - list of fields to return
        """
        parser_params = extra_params + self.GET_PARAMS
        if self.NEED_PAGING:
            parser_params += self.PAGING_PARAMS

        params = self._parse_parameters(parser_params)
        fields = self._get_fields_to_show(params)
        opts = self._get_undefer_options(fields)
        models = self._get_list_query(params, opts, **kwargs)
        if self.NEED_PAGING:
            data_paginated = models.paginate(params['page'] or 1, 20, False)
            data = {'items': qs2list(data_paginated.items, fields),
                    'pages': data_paginated.pages,
                    'total': data_paginated.total,
                    'page': data_paginated.page,
                    'has_next': data_paginated.has_next,
                    'has_prev': data_paginated.has_prev,
                    'per_page': data_paginated.per_page}
            return {self.list_key: data}
        else:
            return {self.list_key: qs2list(models.all(), fields)}

    @property
    def list_key(self):
        return '%ss' % self.OBJECT_NAME

    def _is_list_method(self, **kwargs):
        name = kwargs.get('name')
        return not name

    def _get_list_query(self, params, opts, **kwargs):
        return self.MODEL.query.options(*opts)

    def _get_details_query(self, params, opts, **kwargs):
        name = kwargs.get('name')
        return db.session.query(self.MODEL).options(*opts)\
            .filter(self.MODEL.name == name)

    def _fill_post_data(self, obj, params):
        raise NotImplemented('Should be implemented in the child class')

    @render()
    def _details(self, extra_params=(), **kwargs):
        """
        GET model by name
        """
        params = self._parse_parameters(extra_params + self.GET_PARAMS)
        fields = self._get_fields_to_show(params)
        opts = self._get_undefer_options(fields)
        model = self._get_details_query(params, opts, **kwargs)
        return {self.OBJECT_NAME: qs2list(model.one(), fields)}

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
            return prop in self.MODEL.__table__._columns.keys()

        opts = []
        for field in fields:
            if is_model_field(field):
                opts.append(undefer(field))
            else:
                subfileds = field.split('.')
                if subfileds:
                    subfield = subfileds[0]
                    if is_model_field(subfield):
                        opts.append(undefer(subfield))
        return opts


class Models(BaseResource):
    """
    Models API methods
    """
    MODEL = Model
    GET_ACTIONS = ('tests', )
    decorators = [crossdomain(origin='*')]
    methods = ['GET', 'OPTIONS', 'PUT', 'POST']

    def _get_list_query(self, params, opts, **kwargs):
        comparable = params.get('comparable', False)
        # TODO: Look for models with completed tests if comparable
        return super(Models, self)._get_list_query(params, opts)

    def _is_list_method(self, **kwargs):
        model = kwargs.get('model')
        return not model

    def _get_details_query(self, params, opts, **kwargs):
        model = kwargs.get('model')
        return db.session.query(self.MODEL).options(*opts)\
            .filter(self.MODEL.name == model)

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


api.add_resource(Models, '/cloudml/model/<regex("[\w\.]*"):model>',
                 '/cloudml/model/<regex("[\w\.]+"):model>/<regex("[\w\.]+"):action>')


class ImportHandlerResource(BaseResource):
    """
    Import handler API methods
    """
    MODEL = ImportHandler
    OBJECT_NAME = 'import_handler'
    decorators = [crossdomain(origin='*')]
    methods = ['GET', 'OPTIONS', 'PUT', 'POST']

    @classmethod
    def get_model_parser(cls):
        if not hasattr(cls, "_model_parser"):
            parser = reqparse.RequestParser()
            parser.add_argument('data', type=str)
            parser.add_argument('type', type=str)
            cls._model_parser = parser
        return cls._model_parser

    def _fill_post_data(self, obj, params):
        obj.type = params.get('type')
        obj.data = params.get('data')

api.add_resource(ImportHandlerResource,
                 '/cloudml/import/handler/<regex("[\w\.]*"):name>')


class Tests(BaseResource):
    MODEL = Test
    OBJECT_NAME = 'test'
    decorators = [crossdomain(origin='*')]

    def _get_details_query(self, params, opts, **kwargs):
        model = kwargs.get('model')
        test_name = kwargs.get('test_name')
        return Test.query.options(*opts).join(Model)\
            .filter(Model.name == model,
                    Test.name == test_name)

    @render(code=201)
    def post(self, model, test_name):
        opts = [undefer(field) for field in ('importhandler', 'trainer',
            'positive_weights', 'negative_weights',
            'positive_weights_tree', 'negative_weights_tree')]
        model = Model.query.options(*opts).filter(
            Model.name == model,
            Model.status == Model.STATUS_TRAINED).first()

        parser = populate_parser(model)
        parameters = parser.parse_args()

        test = Test(model)
        test.status = Test.STATUS_QUEUED
        test.parameters = parameters
        db.session.add(test)
        db.session.commit()

        run_test.delay(test, test.model)

        return {'test': test}

api.add_resource(Tests, '/cloudml/model/<regex("[\w\.]+"):model>/test/<regex("[\w\.\-]+"):test_name>')


class Datas(BaseResource):
    MODEL = Data
    OBJECT_NAME = 'data'
    NEED_PAGING = True
    decorators = [crossdomain(origin='*')]

    def _is_list_method(self, **kwargs):
        data_id = kwargs.get('data_id')
        return not data_id

    def _get_list_query(self, params, opts, **kwargs):
        model = kwargs.get('model')
        test_name = kwargs.get('test_name')
        return Data.query.options(*opts).join(Test).join(Model)\
            .filter(and_(Model.name == model, Test.name == test_name)).group_by(Data.group_by_field)

    def _get_details_query(self, params, opts, **kwargs):
        model = kwargs.get('model')
        test_name = kwargs.get('test_name')
        data_id = kwargs.get('data_id')
        return Data.query.options(*opts)\
            .join(Test).join(Model)\
            .filter(and_(Model.name == model, Test.name == test_name,
                         Data.id == data_id))

api.add_resource(Datas, '/cloudml/model/<regex("[\w\.]+"):model>/test/<regex("[\w\.\-]+"):test_name>/data')
api.add_resource(Datas,
                 '/cloudml/model/<regex("[\w\.]+"):model>/test/<regex("[\w\.\-]+"):test_name>/data/<regex("[\w\.\-]+"):data_id>')


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

api.add_resource(CompareReport, '/cloudml/reports/compare')


class Predict(restful.Resource):
    decorators = [crossdomain(origin='*')]

    @render(code=201)
    def post(self, model):
        from core.importhandler.importhandler import ExtractionPlan, RequestImportHandler
        from itertools import izip
        model = Model.query.filter(Model.name == model).one()
        data = request.json
        plan = ExtractionPlan(model.importhandler, is_file=False)
        import_handler = RequestImportHandler(plan, data)
        result = []
        count = 0
        probabilities = model.trainer.predict(import_handler, ignore_error=False)
        for prob, label in izip(probabilities['probs'], probabilities['labels']):
            prob = prob.tolist() if not (prob is None) else []
            label = label.tolist() if not (label is None)  else []
            result.append({'item': count,
                           'label': label,
                           'probs': prob})
            count += 1
        return result

api.add_resource(Predict, '/cloudml/model/<regex("[\w\.]*"):model>/predict')

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
