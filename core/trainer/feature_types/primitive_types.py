# Author: Nikolay Melnik <nmelnik@upwork.com>

from sklearn.feature_extraction import DictVectorizer

from base import FeatureTypeBase, FeatureTypeInstanceBase, \
    InvalidFeatureTypeException


class PrimitiveFeatureTypeInstance(FeatureTypeInstanceBase):

    def __init__(self, *args, **kwargs):
        python_type = kwargs.pop('python_type')
        super(PrimitiveFeatureTypeInstance, self).__init__(*args, **kwargs)
        self.python_type = python_type

    def transform(self, value):
        default = self.python_type()
        params = self.active_params()
        if params is not None:
            default = params.get('default', default)
        if value is None:
            return default
        try:
            return self.python_type(value)
        except ValueError:
            pass
        return default


class PrimitiveFeatureTypeBase(FeatureTypeBase):
    instance = PrimitiveFeatureTypeInstance

    def get_instance(self, params, input_format=None):
        set_params = set()
        preprocessor = None
        if input_format == 'dict':
            preprocessor = DictVectorizer()
        if params is not None:
            set_params = set(params)
        if set(self.required_params).issubset(set_params):
            return self.instance(params,
                                 self.default_params,
                                 preprocessor=preprocessor,
                                 default_scaler=self.default_scaler,
                                 python_type=self.python_type)
        raise InvalidFeatureTypeException(
            'Not all required parameters set')


class BooleanFeatureType(PrimitiveFeatureTypeBase):
    python_type = bool


class IntFeatureType(PrimitiveFeatureTypeBase):
    python_type = int


class FloatFeatureType(PrimitiveFeatureTypeBase):
    python_type = float


class StrFeatureType(PrimitiveFeatureTypeBase):
    python_type = str
