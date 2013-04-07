import json

from flask.ext.restful import reqparse
from werkzeug.datastructures import FileStorage

from api import db, api, app
from api.utils import crossdomain, ERR_NO_SUCH_MODEL, odesk_error_response
#, Test, Data, ImportHandler
from api.tasks import train_model, run_test
from api.resources import BaseResource

from core.trainer.store import load_trainer
from core.trainer.trainer import Trainer
from core.trainer.config import FeatureModel
from core.importhandler.importhandler import ExtractionPlan, \
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
    GET_ACTIONS = ('weights', )
    PUT_ACTIONS = ('train', )
    FILTER_PARAMS = (('status', str), )
    # ('comparable', bool)
    methods = ('GET', 'OPTIONS', 'DELETE', 'PUT', 'POST')

    @property
    def Model(self):
        return db.cloudml.Model

    def _get_model_parser(self, **kwargs):
        """
        Returns Model parser that used when POST model.
        """
        return model_parser

    # GET specific methods

    def _get_weights_action(self, per_page=50, **kwargs):
        """
        Gets list with Model's weighted parameters with pagination.
        """
        paging_params = (('ppage', int), ('npage', int),)
        params = self._parse_parameters(self.GET_PARAMS + paging_params)

        # Paginate weights
        ppage = params.get('ppage') or 1
        npage = params.get('npage') or 1
        fields = self._get_fields_to_show(params)
        fields_dict = {'positive_weights': {'$slice': [(ppage - 1) * per_page, per_page]},
                       'negative_weights': {'$slice': [(npage - 1) * per_page, per_page]}}
        for field in fields:
            fields_dict[field] = ""

        model = self._get_details_query(params, fields_dict, **kwargs)
        return self._render({self.OBJECT_NAME: model})

    # POST specific methods

    def _fill_post_data(self, obj, params, **kwargs):
        """
        Fills Model specific fields when uploading trained model or
        creating new model.
        """
        obj.name = kwargs.get('name')
        if 'features' in params and params['features']:
            # Uploading new model
            feature_model = FeatureModel(params['features'], is_file=False)
            trainer = Trainer(feature_model)
            obj.set_trainer(trainer)

            obj.features = json.loads(params['features'])
        else:
            # Uploading trained model
            trainer = load_trainer(params['trainer'])
            obj.status = obj.STATUS_TRAINED
            obj.set_trainer(trainer)
            obj.set_weights(**trainer.get_weights())

        obj.importhandler = json.loads(params['importhandler'])
        obj.train_importhandler = json.loads(params['importhandler'])
        plan = ExtractionPlan(params['importhandler'], is_file=False)
        obj.import_params = plan.input_params
        obj.import_params.append('group_by')
        obj.save()

    # PUT specififc methods

    def _fill_put_data(self, model):
        param = model_parser.parse_args()
        model.importhandler = param['importhandler'] or model.importhandler
        model.train_importhandler = param['train_importhandler'] \
            or model.train_importhandler
        db.session.commit()
        return {'model': model}

    def _put_train_action(self, **kwargs):
        model = self._get_details_query(None, None,
                                        **kwargs)
        parser = populate_parser(model)
        params = parser.parse_args()
        train_model.delay(model.name, params)
        model.status = model.STATUS_QUEUED
        model.save()

        return self._render({self.OBJECT_NAME: model._id})

api.add_resource(Models, '/cloudml/model/<regex("[\w\.]*"):name>',
                 '/cloudml/model/<regex("[\w\.]+"):name>/<regex("[\w\.]+"):action>')


class ImportHandlerResource(BaseResource):
    """
    Import handler API methods
    """
    @property
    def Model(self):
        return db.cloudml.ImportHandler

    OBJECT_NAME = 'import_handler'
    decorators = [crossdomain(origin='*')]
    methods = ['GET', 'OPTIONS', 'PUT', 'POST']

    @classmethod
    def _get_model_parser(cls):
        if not hasattr(cls, "_model_parser"):
            parser = reqparse.RequestParser()
            parser.add_argument('data', type=str)
            parser.add_argument('type', type=str)
            cls._model_parser = parser
        return cls._model_parser

    def _fill_post_data(self, obj, params, name):
        obj.name = name
        obj.type = params.get('type')
        obj.data = json.loads(params.get('data'))
        obj.save()

api.add_resource(ImportHandlerResource,
                 '/cloudml/import/handler/<regex("[\w\.]*"):name>')


class Tests(BaseResource):
    """
    Tests API Resource
    """
    OBJECT_NAME = 'test'

    @property
    def Model(self):
        return db.cloudml.Model

    def _get_list_query(self, params, fields, **kwargs):
        model_name = kwargs.get('model')
        fields = ["tests.%s" % f for f in fields]
        model = self.Model.find_one({'name': model_name}, fields)
        return model.tests

    def _get_details_query(self, params, fields, **kwargs):
        model_name = kwargs.get('model')
        test_num = int(kwargs.get('name'))
        fields = ["tests.%s" % f for f in fields]
        model = self.Model.find_one({'name': model_name}, fields)
        return model.tests[test_num]

    def _fill_post_data(self, obj, params, model, name):
        parser = populate_parser(obj)
        parameters = parser.parse_args()
        test = db.cloudml.Test()
        test.status = test.STATUS_QUEUED
        test.parameters = parameters
        total = len(obj.tests)
        test.name = "Test%s" % (total + 1)
        obj.tests.append(test)
        obj.save(check_keys=False)
        run_test.run(obj.name, total)

    def _get_post_model(self, params, **kwargs):
        model = kwargs.get('model')
        return self.Model.find_one({'name': model})

api.add_resource(Tests, '/cloudml/model/<regex("[\w\.]+"):model>/test/\
<regex("[\w\.\-]+"):name>', '/cloudml/model/<regex("[\w\.]+"):model>/tests')


class Datas(BaseResource):
    @property
    def Model(self):
        return db.cloudml.Model

    OBJECT_NAME = 'data'
    NEED_PAGING = True
    GET_ACTIONS = ('groupped', )
    DETAILS_PARAM = 'example_num'
    decorators = [crossdomain(origin='*')]

    def _get_list_query(self, params, fields, **kwargs):
        model_name = kwargs.get('model')
        test_num = int(kwargs.get('test_num'))
        fields = ["tests.examples.%s" % f for f in fields]
        model = self.Model.find_one({'name': model_name}, fields)
        return model.tests[test_num]['examples']

    def _paginate(self, models, page=1, per_page=20):
        total = len(models)
        offset = (page - 1) * per_page
        return total, models[offset: offset + per_page]

    def _get_details_query(self, params, fields, **kwargs):
        model_name = kwargs.get('model')
        test_num = int(kwargs.get('test_num'))
        example_num = int(kwargs.get('example_num'))
        fields = ["tests.examples.%s" % f for f in fields]
        model = self.Model.find_one({'name': model_name}, fields)
        return model.tests[test_num]['examples'][example_num]

    def _get_groupped_action(self, **kwargs):
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
<regex("[\w\.\-]+"):test_num>/data',
                 '/cloudml/model/<regex("[\w\.]+")\
:model>/test/<regex("[\w\.\-]+"):test_num>/data/<regex("[\w\.\-]+"):example_num>',
                 '/cloudml/model/<regex("[\w\.]+"):model>/test/\
<regex("[\w\.\-]+"):test_num>/action/<regex("[\w\.]+"):action>/data')


# class CompareReport(restful.Resource):
#     decorators = [crossdomain(origin='*')]

#     @render(brief=False)
#     def get(self):
#         parser = reqparse.RequestParser()
#         parser.add_argument('test1', type=str)
#         parser.add_argument('test2', type=str)
#         parser.add_argument('model1', type=str)
#         parser.add_argument('model2', type=str)
#         parameters = parser.parse_args()
#         test1 = db.session.query(Test).join(Model)\
#             .filter(Model.name == parameters['model1'],
#                     Test.name == parameters['test1']).one()
#         test2 = db.session.query(Test).join(Model)\
#             .filter(Model.name == parameters['model2'],
#                     Test.name == parameters['test2']).one()
#         examples1 = db.session.query(Data).join(Test)\
#             .filter(Data.test == test1)\
#             .order_by(desc(Data.pred_label))[:10]
#         examples2 = db.session.query(Data).join(Test)\
#             .filter(Data.test == test2)\
#             .order_by(desc(Data.pred_label))[:10]
#         return {'test1': test1, 'test2': test2,
#                 'examples1': examples1, 'examples2': examples2}

# api.add_resource(CompareReport, '/cloudml/reports/compare')


# class Predict(restful.Resource):
#     decorators = [crossdomain(origin='*')]

#     @render(code=201)
#     def post(self, model, import_handler):
#         from itertools import izip
#         try:
#             importhandler = ImportHandler.query.filter(
#                 ImportHandler.name == import_handler).one()
#         except NoResultFound, exc:
#             exc.message = "Import handler %s doesn\'t exist" % import_handler
#             raise exc
#         try:
#             model = Model.query.filter(Model.name == model).one()
#         except Exception, exc:
#             exc.message = "Model %s doesn\'t exist" % model
#             raise exc
#         data = [request.form, ]
#         plan = ExtractionPlan(importhandler.data, is_file=False)
#         request_import_handler = RequestImportHandler(plan, data)
#         probabilities = model.trainer.predict(request_import_handler,
#                                               ignore_error=False)
#         prob = probabilities['probs'][0]
#         label = probabilities['labels'][0]
#         prob = prob.tolist() if not (prob is None) else []
#         label = label.tolist() if not (label is None) else []
#         result = {'label': label,
#                   'probs': prob}
#         return result

# api.add_resource(Predict, '/cloudml/model/<regex("[\w\.]*"):model>/\
# <regex("[\w\.]*"):import_handler>/predict')


def populate_parser(model):
    parser = reqparse.RequestParser()
    for param in model.import_params:
        parser.add_argument(param, type=str)
    return parser
