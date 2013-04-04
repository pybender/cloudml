from flask.ext import restful
from flask.ext.restful import reqparse
from sqlalchemy.orm import undefer

from api.utils import crossdomain, ERR_NO_SUCH_MODEL, odesk_error_response
from api.decorators import render
from api import db


class BaseResource(restful.Resource):
    MODEL = None
    GET_ACTIONS = ()
    OBJECT_NAME = 'model'
    NEED_PAGING = False
    GET_PARAMS = (('show', str), )
    PAGING_PARAMS = (('page', int), )
    decorators = [crossdomain(origin='*', headers="accept, origin, content-type")]

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
        db.session.delete(model)
        db.session.commit()
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
        return fields.split(',') if fields else self.MODEL.__public__

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
