"""
This module gathers all primitive feature types processors.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

from sklearn.feature_extraction import DictVectorizer

from base import FeatureTypeBase, FeatureTypeInstanceBase, \
    InvalidFeatureTypeException
from cloudml.utils import process_bool


PROCESSOR_MAP = {
    'bool': process_bool,
    'int': int,
    'float': float,
    'str': str
}


class PrimitiveFeatureTypeInstance(FeatureTypeInstanceBase):
    def __init__(self, *args, **kwargs):
        python_type = kwargs.pop('python_type')
        super(PrimitiveFeatureTypeInstance, self).__init__(*args, **kwargs)
        self.python_type = python_type
        self.default = PROCESSOR_MAP[self.python_type]()

    def transform(self, value):
        import numpy as np
        default = np.nan
        params = self.active_params()
        if params is not None:
            default = params.get('default', default)
        if value is None:
            return default
        try:
            return PROCESSOR_MAP[self.python_type](value)
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
    """
    Converts the value to boolean. Uses python bool() function.
    Thus bool(0) = false, bool(null) = false, bool('') = false.
    """
    python_type = 'bool'


class IntFeatureType(PrimitiveFeatureTypeBase):
    """
    Converts each item to an integer. In case the value is null,
    the trainer checks for parameter named default. If it is set,
    then its value is used, otherwise 0 is used.
    """
    python_type = 'int'


class FloatFeatureType(PrimitiveFeatureTypeBase):
    """
    Converts each item to a float.
    """
    python_type = 'float'


class StrFeatureType(PrimitiveFeatureTypeBase):
    """
    String feature type.
    """
    python_type = 'str'
