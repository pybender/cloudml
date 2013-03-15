import json
from datetime import datetime


class Serializer(object):
    __public__ = None
    __all_public__ = None
    "Must be implemented by implementors"

    def to_brief_dict(self):
        return self._to_dict(self.__public__)

    def to_full_dict(self):
        return self._to_dict(self.__all_public__ or self.__public__)

    def _to_dict(self, fields):
        dict = {}
        for public_key in fields:
            value = getattr(self, public_key)
            if value:
                dict[public_key] = value
        return dict


class ModelEncoder(json.JSONEncoder):
    method_name = 'to_brief_dict'

    def default(self, obj):
        if isinstance(obj, Serializer):
            method = getattr(obj, self.method_name)
            return method()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)


class BriefDetailsEncoder(ModelEncoder):
    pass


class FullDetailsEncoder(BriefDetailsEncoder):
    method_name = 'to_full_dict'
