import json
import math
from flask.ext import restful
from flask.ext.restful import reqparse

from api.utils import crossdomain, ERR_NO_SUCH_MODEL, odesk_error_response
from api.serialization import encode_model
from api import app


class BaseResource(restful.Resource):
    """
    Base class for any API Resource
    """
    GET_ACTIONS = ()
    POST_ACTIONS = ()
    PUT_ACTIONS = ()

    DETAILS_PARAM = 'name'
    OBJECT_NAME = 'model'
    NEED_PAGING = False
    GET_PARAMS = (('show', str), )
    FILTER_PARAMS = ()
    PAGING_PARAMS = (('page', int), )
    decorators = [crossdomain(origin='*',
                              headers="accept, origin, content-type")]

    @property
    def Model(self):
        """
        Returns base DB model of the Resource.
        """
        raise NotImplemented()

    @property
    def list_key(self):
        """
        Returns a key name, when list of results returned.
        """
        return '%ss' % self.OBJECT_NAME

    # HTTP Methods

    def get(self, action=None, **kwargs):
        """
        GET model/models.
            * action - specific action for GET method.
                Note: action should be in `GET_ACTIONS` list and
                _get_{{ action }}_action method should be implemented.
            * ... - list of url parameters. For example parent_name and name.
        """
        if action:
            return self._apply_action(action, method='GET', **kwargs)

        if self._is_list_method(**kwargs):
            return self._list(**kwargs)
        else:
            return self._details(**kwargs)

    def post(self, action=None, **kwargs):
        """
        Adds new model
        """
        if action:
            return self._apply_action(action, method='POST', **kwargs)

        parser = self._get_model_parser()
        params = parser.parse_args()

        model = self.Model()
        self._fill_post_data(model, params, **kwargs)
        model.save()
        return self._render({self.OBJECT_NAME: model._id}, code=201)

    def put(self, action=None, **kwargs):
        """
        Updates new model
        """
        if action:
            return self._apply_action(action, method='PUT', **kwargs)

        parser = self._get_model_parser(method='PUT')
        params = parser.parse_args()

        model = self.Model()
        model = self._fill_put_data(model, params, **kwargs)
        model.save()
        return self._render({self.OBJECT_NAME: model._id}, code=200)

    def delete(self, action=None, **kwargs):
        """
        Deletes unused model
        """
        model = self._get_details_query(None, ('_id', 'name'), **kwargs)
        model.collection.remove({'_id': model._id})
        return '', 204

    def _list(self, extra_params=(), **kwargs):
        """
        Gets list of models

        GET parameters:
            * show - list of fields to return
        """
        parser_params = extra_params + self.GET_PARAMS + self.FILTER_PARAMS
        if self.NEED_PAGING:
            parser_params += self.PAGING_PARAMS
        params = self._parse_parameters(parser_params)
        fields = self._get_fields_to_show(params)

        # Removing empty values
        kw = dict([(k, v) for k, v in kwargs.iteritems() if v])
        models = self._get_list_query(params, fields, **kw)
        context = {}
        if self.NEED_PAGING:
            context['total'] = total = models.count()
            context['per_page'] = per_page = params.get('per_page') or 20
            context['page'] = page = params.get('page')
            offset = (page - 1) * per_page
            models = models.skip(offset).limit(per_page)
            context['pages'] = pages = int(math.ceil(1.0 * total / per_page))
            context['has_prev'] = page > 1
            context['has_next'] = page < pages

        context.update({self.list_key: models})
        return self._render(context)

    def _details(self, extra_params=(), **kwargs):
        """
        GET model by name
        """
        params = self._parse_parameters(extra_params + self.GET_PARAMS)
        fields = self._get_fields_to_show(params)
        model = self._get_details_query(params, fields, **kwargs)
        return self._render({self.OBJECT_NAME: model})

    # Specific actions for GET

    def _get_list_query(self, params, fields, **kwargs):
        filter_names = [v[0] for v in self.FILTER_PARAMS]
        filter_params = dict([(k, v) for k, v in params.iteritems()
                              if not v is None and k in filter_names])
        kwargs.update(filter_params)
        return self.Model.find(kwargs, fields)

    def _get_details_query(self, params, fields, **kwargs):
        return self.Model.find_one(kwargs, fields)

    def _get_model_parser(self, **kwargs):
        raise NotImplemented()

    def _fill_post_data(self, obj, params):
        raise NotImplemented()

    # Utility methods

    def _apply_action(self, action, method='GET', **kwargs):
        if action in getattr(self, '%s_ACTIONS' % method):
            method_name = "_%s_%s_action" % (method.lower(), action)
            return getattr(self, method_name)(**kwargs)
        else:
            return odesk_error_response(404, ERR_NO_SUCH_MODEL,
                                        "Invalid action \
for %s method: %s" % (method, action))

    def _is_list_method(self, **kwargs):
        name = kwargs.get(self.DETAILS_PARAM)
        return not name

    def _parse_parameters(self, extra_params=()):
        parser = reqparse.RequestParser()
        for name, param_type in extra_params:
            parser.add_argument(name, type=param_type)
        return parser.parse_args()

    def _get_fields_to_show(self, params):
        fields = params.get('show', None)
        return fields.split(',') if fields else self.MODEL.__public__

    def _render(self, content, code=200):
        content = json.dumps(content, default=encode_model)
        return app.response_class(content,
                                  mimetype='application/json'), code
