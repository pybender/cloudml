"""
This module gather classes and utils for categorical and categorical_label
feature types processing.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import re
from functools import update_wrapper
from sklearn.feature_extraction.text import CountVectorizer

from base import FeatureTypeBase, FeatureTypeInstanceBase


class CategoricalFeatureTypeInstance(FeatureTypeInstanceBase):
    def transform(self, value):
        default = str()
        params = self.active_params()
        if params:
            default = params.get('default', default)
        if value is None:
            return default
        return str(value)


class CategoricalFeatureType(FeatureTypeBase):
    optional_params = ['exclude','categories','split_pattern', 'min_df']
    instance = CategoricalFeatureTypeInstance

    def get_instance(self, params, input_format=None):
        tokenizer = None
        split_pattern = None
        min_df = 0
        token_pattern = u'(?u)\b\w\w+\b'
        if params is not None:
            split_pattern = params.get('split_pattern', None)
            min_df = int(params.get('min_df', 0) or 0)
        if split_pattern:
            tokenizer = tokenizer_dec(tokenizer_func, split_pattern)
        else:
            token_pattern = '.+'
        preprocessor = CountVectorizer(tokenizer=tokenizer,
                                       token_pattern=token_pattern,
                                       min_df=min_df,
                                       binary=True)
        return self.instance(params,
                             self.default_params,
                             preprocessor=preprocessor)


def tokenizer_func(x, split_pattern):
    return re.split(split_pattern, x)


class tokenizer_dec(object):
    def __init__(self, func, split_pattern):
        self.func = func
        self.split_pattern = split_pattern
        try:
            update_wrapper(self, func)
        except:
            pass

    def __call__(self, x):
        return self.func(x, self.split_pattern)
