import json
from flask.ext.restful import reqparse
from werkzeug.datastructures import FileStorage
from bson.objectid import ObjectId

from api import db, api
from api.utils import crossdomain, ERR_INVALID_DATA, odesk_error_response
from api.resources import BaseResource
from core.trainer.store import load_trainer
from core.trainer.trainer import Trainer
from core.trainer.config import FeatureModel
from core.importhandler.importhandler import ExtractionPlan, \
    RequestImportHandler
from api.models import Model, Test, TestExample, ImportHandler

model_parser = reqparse.RequestParser()
model_parser.add_argument('importhandler', type=str, default=None)
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
        fields_dict = {'positive_weights': {'$slice': [(ppage - 1) * per_page,
                                                       per_page]},
                       'negative_weights': {'$slice': [(npage - 1) * per_page,
                                                       per_page]}}
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
        obj.save()

    # PUT specififc methods

    def _fill_put_data(self, model, param,  **kwargs):
        importhandler = None
        train_importhandler = None
        if param['importhandler']:
            importhandler = json.loads(param['importhandler'])
        if param['train_importhandler']:
            train_importhandler = json.loads(param['train_importhandler'])
        model.importhandler =  importhandler or model.importhandler
        model.train_importhandler =  train_importhandler \
            or model.train_importhandler
        model.save()
        return model

    def _put_train_action(self, **kwargs):
        from api.tasks import train_model
        model = self._get_details_query(None, None,
                                        **kwargs)
        parser = populate_parser(model)
        params = parser.parse_args()
        train_model.delay(model.name, params)
        model.status = model.STATUS_QUEUED
        model.save()

        return self._render({self.OBJECT_NAME: model._id})

api.add_resource(Models, '/cloudml/model/<regex("[\w\.]*"):name>',
                 '/cloudml/model/<regex("[\w\.]+"):name>/\
<regex("[\w\.]+"):action>')


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
        return db.cloudml.Test

    def _get_list_query(self, params, fields, **kwargs):
        model_name = kwargs.get('model')
        return self.Model.find({'model_name': model_name}, fields)

    def _get_details_query(self, params, fields, **kwargs):
        model_name = kwargs.get('model')
        test_name = kwargs.get('name')
        return self.Model.find_one({'model_name': model_name,
                                   'name': test_name}, fields)

    def post(self, action=None, **kwargs):
        from api.tasks import run_test
        model_name = kwargs.get('model')
        model = db.cloudml.Model.find_one({'name': model_name})
        parser = populate_parser(model)
        parameters = parser.parse_args()
        test = db.cloudml.Test()
        test.status = test.STATUS_QUEUED
        test.parameters = parameters

        total = db.cloudml.Test.find({'model_name': model.name}).count()
        test.name = "Test%s" % (total + 1)
        test.model_name = model.name
        test.model = model
        test.save(check_keys=False)
        run_test.delay(str(test._id))
        return self._render({self.OBJECT_NAME: test._id}, code=201)

api.add_resource(Tests, '/cloudml/model/<regex("[\w\.]+"):model>/test/\
<regex("[\w\.\-]+"):name>', '/cloudml/model/<regex("[\w\.]+"):model>/tests')


REDUCE_FUNC = 'function(obj, prev) {\
                            prev.list.push({"label": obj.pred_label,\
                            "pred": obj.label})\
                      }'


class TestExamplesResource(BaseResource):
    @property
    def Model(self):
        return db.cloudml.TestExample

    OBJECT_NAME = 'data'
    NEED_PAGING = True
    GET_ACTIONS = ('groupped', )
    DETAILS_PARAM = 'example_id'
    decorators = [crossdomain(origin='*')]

    def _get_list_query(self, params, fields, **kwargs):
        model_name = kwargs.get('model')
        test_name = kwargs.get('test_name')
        return self.Model.find({'model_name': model_name,
                                'test_name': test_name}, fields)

    def _get_details_query(self, params, fields, **kwargs):
        model_name = kwargs.get('model')
        test_name = kwargs.get('test_name')
        example_id = kwargs.get('example_id')
        return self.Model.find_one({'model_name': model_name,
                                    'test_name': test_name,
                                    '_id': ObjectId(example_id)}, fields)

    def _get_groupped_action(self, **kwargs):
        """
        Groups data by `group_by_field` field and calculates mean average precision.
        Note: `group_by_field` should be specified in request parameters.
        """
        from ml_metrics import apk
        import numpy as np
        from operator import itemgetter

        # getting from request parameters fieldname to group.
        parser = reqparse.RequestParser()
        parser.add_argument('field', type=str)
        params = parser.parse_args()
        group_by_field = params.get('field')
        if not group_by_field:
            return odesk_error_response(400, ERR_INVALID_DATA,
                                        'field parameter is required')
        model_name = kwargs.get('model')
        test_name = kwargs.get('test_name')
        ex_collection = db.cloudml.TestExample.collection
        groups = ex_collection.group([group_by_field, ],
                                     {'model_name': model_name,
                                      'test_name': test_name},
                                     {'list': []}, REDUCE_FUNC)
        
        res = []
        avps = []

        for group in groups:
            group_list = group['list']
            labels = [item['label'] for item in group_list]
            pred_labels = [item['pred'] for item in group_list]
            avp = apk(labels, pred_labels)
            avps.append(avp)
            res.append({'group_by_field': group[group_by_field],
                        'count': len(group_list),
                        'avp': avp})
        res = sorted(res, key=itemgetter("count"), reverse=True)[:100]
        mavp = np.mean(avps)

        context = {self.list_key: {'items': res},
                   'field_name': group_by_field,
                   'mavp': mavp}
        return self._render(context)

api.add_resource(TestExamplesResource, '/cloudml/model/\
<regex("[\w\.]+"):model>/test/<regex("[\w\.\-]+"):test_name>/data',
                 '/cloudml/model/<regex("[\w\.]+")\
:model>/test/<regex("[\w\.\-]+"):test_name>/data/\
<regex("[\w\.\-]+"):example_id>', '/cloudml/model/<regex("[\w\.]+"):model>\
/test/<regex("[\w\.\-]+"):test_name>/action/<regex("[\w\.]+"):action>/data')


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
