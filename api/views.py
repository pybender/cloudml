from flask.ext import restful
from flask import request
from flask.ext.restful import reqparse
from werkzeug.datastructures import FileStorage
from sqlalchemy import and_
from sqlalchemy import func
from sqlalchemy.sql.expression import asc, desc
from sqlalchemy.orm import undefer
from sqlalchemy.orm.exc import NoResultFound

from api.decorators import render
from api import db, api
from api.utils import crossdomain, ERR_NO_SUCH_MODEL, odesk_error_response
from api.models import Model, Test, Data, ImportHandler
from api.tasks import train_model, run_test
from api.resources import BaseResource, qs2list

from core.trainer.store import load_trainer
from core.trainer.trainer import Trainer
from core.trainer.config import FeatureModel
from core.importhandler.importhandler import ExtractionPlan,\
    RequestImportHandler

get_parser = reqparse.RequestParser()
get_parser.add_argument('show', type=str)

page_parser = reqparse.RequestParser()
page_parser.add_argument('page', type=int)


model_parser = reqparse.RequestParser()
model_parser.add_argument('importhandler', type=str)
model_parser.add_argument('train_importhandler', type=str)
model_parser.add_argument('features', type=str)
model_parser.add_argument('trainer', type=FileStorage, location='files')


class Models(BaseResource):
    """
    Models API methods
    """
    MODEL = Model
    GET_ACTIONS = ('tests', 'weights')
    methods = ('GET', 'OPTIONS', 'DELETE', 'PUT', 'POST')

    def _get_list_query(self, params, opts, **kwargs):
        comparable = params.get('comparable', False)
        if comparable:
            return Model.options(*opts).filter(Model.comparable)
        else:
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
        tests = db.session.query(Test).join(Model)\
            .filter(Model.name == model).all()
        return {'tests': qs2list(tests, fields)}

    @render()
    def _get_weights_action(self, per_page=50, **kwargs):
        """
        Gets list with Model's weighted parameters with pagination.
        """
        paging_params = (('ppage', int), ('npage', int),)
        params = self._parse_parameters(self.GET_PARAMS + paging_params)
        fields = self._get_fields_to_show(params)
        wfields = ['positive_weights', 'negative_weights']
        opts = self._get_undefer_options(fields + wfields)
        model = self._get_details_query(params, opts, **kwargs).one()
        ppage = params.get('ppage') or 1
        npage = params.get('npage') or 1
        return {self.OBJECT_NAME: qs2list(model, fields),
                'positive': model.positive_weights
                [(ppage - 1) * per_page:ppage * per_page],
                'negative': model.negative_weights
                [(npage - 1) * per_page:npage * per_page]}

    @render(code=201)
    def post(self, model):
        """
        Adds new Trained Model
        """
        param = model_parser.parse_args()
        if not param['features']:
            return self._upload(model, param)
        else:
            return self._add(model, param)

    def _add(self, model, param):
        model = Model(model)
        model.train_importhandler = param['importhandler']
        model.importhandler = param['importhandler']
        model.features = param['features']

        feature_model = FeatureModel(model.features, is_file=False)
        trainer = Trainer(feature_model)
        plan = ExtractionPlan(model.train_importhandler, is_file=False)
        model.import_params = plan.input_params
        model.import_params.append('group_by')
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
        model.import_params.append('group_by')
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

    def _is_list_method(self, **kwargs):
        test_name = kwargs.get('test_name')
        return not test_name

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

api.add_resource(Tests, '/cloudml/model/<regex("[\w\.]+"):model>/test/\
<regex("[\w\.\-]+"):test_name>')


class Datas(BaseResource):
    MODEL = Data
    OBJECT_NAME = 'data'
    NEED_PAGING = True
    GET_ACTIONS = ('groupped', )
    decorators = [crossdomain(origin='*')]

    def _is_list_method(self, **kwargs):
        data_id = kwargs.get('data_id')
        return not data_id

    def _get_list_query(self, params, opts, **kwargs):
        model = kwargs.get('model')
        test_name = kwargs.get('test_name')
        return Data.query.options(*opts).join(Test).join(Model)\
            .filter(and_(Model.name == model, Test.name == test_name))

    def _get_details_query(self, params, opts, **kwargs):
        model = kwargs.get('model')
        test_name = kwargs.get('test_name')
        data_id = kwargs.get('data_id')
        return Data.query.options(*opts)\
            .join(Test).join(Model)\
            .filter(and_(Model.name == model, Test.name == test_name,
                         Data.id == data_id))

    @render()
    def _get_groupped_action(self, model, test_name, **kwargs):
        """
        Groups data by `group_by_field` field.
        """
        from ml_metrics import apk
        import numpy as np
        test = Test.query.join(Model)\
            .filter(Model.name == model,
                    Test.name == test_name).one()
        datas = db.session.query(Data.group_by_field,
                                 func.count(Data.group_by_field).label('total'))\
            .join(Test).join(Model)\
            .filter(and_(Model.name == model, Test.name == test_name))\
            .group_by(Data.group_by_field).order_by('total DESC').all()[:100]

        res = []
        avps = []
        for d in datas:
            data_set = Data.query.filter(Data.group_by_field == d[0]).all()
            labels = [i.label for i in data_set]
            pred_labels = [i.pred_label for i in data_set]
            avp = apk(labels, pred_labels)
            avps.append(avp)
            res.append({'group_by_field': d[0],
                        'count': d[1], 'avp': avp})
        mavp = np.mean(avps)
        field_name = test.parameters.get('group_by')
        return {self.list_key: {'items': res}, 'field_name': field_name, 'mavp': mavp}


api.add_resource(Datas, '/cloudml/model/<regex("[\w\.]+"):model>/test/\
<regex("[\w\.\-]+"):test_name>/data',
                 '/cloudml/model/<regex("[\w\.]+")\
:model>/test/<regex("[\w\.\-]+"):test_name>/data/<regex("[\w\.\-]+"):data_id>',
                 '/cloudml/model/<regex("[\w\.]+"):model>/test/\
<regex("[\w\.\-]+"):test_name>/action/<regex("[\w\.]+"):action>/data')


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
    def post(self, model, import_handler):
        from itertools import izip
        try:
            importhandler = ImportHandler.query.filter(
                ImportHandler.name == import_handler).one()
        except NoResultFound, exc:
            exc.message = "Import handler %s doesn\'t exist" % import_handler
            raise exc
        try:
            model = Model.query.filter(Model.name == model).one()
        except Exception, exc:
            exc.message = "Model %s doesn\'t exist" % model
            raise exc
        data = [request.form, ]
        plan = ExtractionPlan(importhandler.data, is_file=False)
        request_import_handler = RequestImportHandler(plan, data)
        probabilities = model.trainer.predict(request_import_handler,
                                              ignore_error=False)
        prob = probabilities['probs'][0]
        label = probabilities['labels'][0]
        prob = prob.tolist() if not (prob is None) else []
        label = label.tolist() if not (label is None) else []
        result = {'label': label,
                  'probs': prob}
        return result

api.add_resource(Predict, '/cloudml/model/<regex("[\w\.]*"):model>/\
<regex("[\w\.]*"):import_handler>/predict')


def populate_parser(model):
    parser = reqparse.RequestParser()
    for param in model.import_params:
        parser.add_argument(param, type=str)
    return parser
