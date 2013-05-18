from functools import update_wrapper
import re

from sklearn.feature_extraction.text import CountVectorizer

from base import FeatureTypeBase, FeatureTypeInstanceBase,\
    InvalidFeatureTypeException

_author__ = 'nmelnik'


class CategoricalFeatureTypeInstance(FeatureTypeInstanceBase):

    def transform(self, value):
        default = str()
        params = self.active_params()
        if params is not None:
            default = params.get('default', default)
        if value is None:
            return default
        try:
            return str(value)
        except ValueError:
            pass
        return default


class CategoricalFeatureType(FeatureTypeBase):
    instance = CategoricalFeatureTypeInstance

    def get_instance(self, params, input_format=None):
        tokenizer = None
        set_params = set()
        split_pattern = None
        token_pattern = u'(?u)\b\w\w+\b'
        if params is not None:
            split_pattern = params.get('split_pattern', None)
            set_params = set(params)
        if split_pattern:
            tokenizer = tokenizer_dec(tokenizer_func, split_pattern)
        else:
            token_pattern = '.+'
        preprocessor = CountVectorizer(tokenizer=tokenizer,
                                       token_pattern=token_pattern,
                                       min_df=0,
                                       binary=True)
        if set(self.required_params).issubset(set_params):
            return self.instance(params,
                                 self.default_params,
                                 preprocessor=preprocessor)
        raise InvalidFeatureTypeException('Not all required parameters set')


def tokenizer_func(x, split_pattern):
    return re.split(split_pattern, x)


class tokenizer_dec(object):
    def __init__(self, func, split_pattern):
        self.func = func
        self.split_pattern = split_pattern
        try:
            functools.update_wrapper(self, func)
        except:
            pass

    def __call__(self, x):
        return self.func(x, self.split_pattern)
