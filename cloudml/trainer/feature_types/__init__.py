from sklearn.preprocessing import LabelEncoder

from categorical import CategoricalFeatureType
from composite import CompositeFeatureType
from ordinal import OrdinalFeatureType
from date import DateFeatureType
from regex import RegexFeatureType, RegexFeatureTypeInstance
from primitive_types import BooleanFeatureType, IntFeatureType, \
    FloatFeatureType, StrFeatureType, PrimitiveFeatureTypeInstance
from base import InvalidFeatureTypeException


__author__ = 'nmelnik'


FEATURE_TYPE_FACTORIES = {
    'boolean': BooleanFeatureType(),
    'int': IntFeatureType(),
    'float': FloatFeatureType(),
    'numeric': FloatFeatureType(),
    'date': DateFeatureType(),
    'map': OrdinalFeatureType(),
    'categorical_label': CategoricalFeatureType(
        preprocessor=LabelEncoder()),
    'categorical': CategoricalFeatureType(),
    'text': StrFeatureType(),
    'regex': RegexFeatureType(),
    'composite': CompositeFeatureType()
}

FEATURE_TYPE_DEFAULTS = {
    # Default is Jan 1st, 2000
    'date': 946684800
}

FEATURE_PARAMS_TYPES = {
    'pattern': {
        'type': 'str',
        'help_text': 'Please enter a pattern'
    },
    'split_pattern': {
        'type': 'str',
        'help_text': 'Please enter a pattern'
    },
    'min_df': {
        'type': 'int',
        'help_text': 'Please enter a int value'
    },
    'mappings': {
        'type': 'dict',
        'help_text': 'Please add parameters to dictionary'
    },
    'chain': {
        'type': 'text',
        'help_text': 'Please enter valid json'
    },
    'categories': {
        'type': 'list',
        'help_text': 'Please enter single of list categories'
    },
    'exclude': {
        'type': 'boolean',
        'help_text': 'If set to true catigories will be exclude'
    }
}
