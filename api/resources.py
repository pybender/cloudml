import json
from flask.ext import restful
from flask.ext.restful import reqparse

from api.utils import crossdomain, ERR_NO_SUCH_MODEL, odesk_error_response
from api.decorators import render
from api import db, app
from api.serialization import encode_model


class BaseResource(restful.Resource):
    GET_ACTIONS = ()
    POST_ACTIONS = ()
    PUT_ACTIONS = ()

    DETAILS_PARAM = 'name'
    OBJECT_NAME = 'model'
    NEED_PAGING = False
    GET_PARAMS = (('show', str), )
    PAGING_PARAMS = (('page', int), )
    decorators = [crossdomain(origin='*', headers="accept, origin, content-type")]

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
        model = self._fill_post_data(model, params, **kwargs)
        model.save()
        return self._render({self.OBJECT_NAME: model._id}, code=201)

    def put(self, action=None, **kwargs):
        """
        Updates new model
        """
        if action:
            return self._apply_action(action, method='PUT', **kwargs)

        parser = self._get_model_parser(method=method)
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
        parser_params = extra_params + self.GET_PARAMS
        if self.NEED_PAGING:
            parser_params += self.PAGING_PARAMS
        params = self._parse_parameters(parser_params)
        fields = self._get_fields_to_show(params)
        # opts = self._get_undefer_options(fields)
        models = self._get_list_query(params, fields)
        # if self.NEED_PAGING:
        #     data_paginated = models.paginate(params['page'] or 1, 20, False)
        #     data = {'items': qs2list(data_paginated.items, fields),
        #             'pages': data_paginated.pages,
        #             'total': data_paginated.total,
        #             'page': data_paginated.page,
        #             'has_next': data_paginated.has_next,
        #             'has_prev': data_paginated.has_prev,
        #             'per_page': data_paginated.per_page}
        #     return {self.list_key: data}
        # else:
        return self._render({self.list_key: models})

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
            return odesk_error_response(404, ERR_NO_SUCH_MODEL, "Invalid action \
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
        return app.response_class(content, mimetype='application/json'), code


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
