import numpy
from scipy.sparse import csc_matrix

__author__ = 'ifoukarakis'

from utils import copy_expected, float_or_int, parse_parameters
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import LdaVectorizer, LsiVectorizer
from sklearn.preprocessing import StandardScaler


class ScalerDecorator(object):
    """
    Decorator for scaler. Scalers don't offer the same interface for fit and
    fit_transform, as they require converting data to columns before
    processing.

    """
    def __init__(self, config):
        """
        Creates a ScalerDecorator. Uses the config under the transformer JSON
        object. Internally, a scaler is created using the given params.

        Keyword arguments:
        params -- a map containing the scaler's configuration.

        """
        filtered_params = copy_expected(
            config['params'], ['copy', 'with_mean', 'with_std'])

        self._scaler = StandardScaler(**filtered_params)

    def _to_column(self, X):
        """
        Converts input list X to numpy column.

        Keyword arguments:
        X -- the list to convert to column

        """

        return numpy.transpose(
            csc_matrix([0.0 if item is None else float(item) for item in X]))

    def fit(self, X, y=None):
        """
        Invokes scaler's fit method, converting first X to column.

        Keyword arguments
        X -- list
        y : numpy array of shape [n_samples]

        """
        return self._scaler.fit(self._to_column(X), y)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Invokes scaler's fit_transform method, converting first X to column.

        Keyword arguments
        X -- list
        y : numpy array of shape [n_samples]

        """
        return self._scaler.fit_transform(self._to_column(X), y, **fit_params)

    def transform(self, X):
        """
        Invokes scaler's transform method, converting first X to column.

        Keyword arguments
        X -- list

        """
        return self._scaler.transform(self._to_column(X))


class SuppressTransformer:
    """
    A vectorizer that suppresses the input feature.
    """
    #TODO: Make it a sublcass of vectorizer?


def get_lda_vectorizer(params):
    if 'num_topics' in params:
        params['num_topics'] = int(params['num_topics'])

    return LdaVectorizer(**params)


def get_lsi_vectorizer(params):
    if 'num_topics' in params:
        params['num_topics'] = int(params['num_topics'])

    return LsiVectorizer(**params)


def get_count_vectorizer(params):
    """
    Creates a CountVectorizer. Uses the config under the transformer JSON
    object.

    Keyword arguments:
    params -- a map containing the vectorizer's configuration.

    """
    if 'ngram_range_min' in params and 'ngram_range_max' in params:
        params['ngram_range'] = (params['ngram_range_min'],
                                 params['ngram_range_max'])
    if 'ngram_range_min' in params:
        del params['ngram_range_min']
    if 'ngram_range_max' in params:
        del params['ngram_range_max']

    return CountVectorizer(**params)


def get_tfidf_vectorizer(params):
    """
    Creates a TfidfVectorizer. Uses the config under the transformer JSON
    object.

    Keyword arguments:
    params -- a map containing the vectorizer's configuration.

    """
    if 'ngram_range_min' in params and 'ngram_range_max' in params:
        params['ngram_range'] = (params['ngram_range_min'],
                                 params['ngram_range_max'])
    if 'ngram_range_min' in params:
        del params['ngram_range_min']
    if 'ngram_range_max' in params:
        del params['ngram_range_max']

    return TfidfVectorizer(**params)


def get_dict_vectorizer(params):
    """
    Creates a DictVectorizer. Uses the config under the vectorizer-config.

    Keyword arguments:
    params -- a map containing the vectorizer's configuration.

    """
    return DictVectorizer(**params)


def get_transformer(transformer):
    if transformer is None:
        return None

    transformer_type = transformer.get('type', None)
    if transformer_type is None:
        return None

    if transformer_type not in TRANSFORMERS:
        return None

    settings = TRANSFORMERS[transformer_type]
    filtered_params = parse_parameters(transformer, settings)
    for param in filtered_params:
        param_type = settings['parameters_types'].get(
            param, None)
        if param_type:
            filtered_params[param] = param_type(filtered_params[param])

    return settings['mthd'](filtered_params)


TRANSFORMERS = {
    'Dictionary': {
        'mthd': get_dict_vectorizer,
        'parameters': ['separator', 'sparse'],
        'parameters_types': {},
        'default': {},  # default value
        'defaults': {}  # default values of the parameters
    },
    'Count': {
        'mthd': get_count_vectorizer,
        'parameters': ['charset', 'charset_error',
                       'strip_accents', 'lowercase',
                       'stop_words', 'token_pattern',
                       'analyzer', 'max_df', 'min_df',
                       'max_features', 'vocabulary',
                       'binary', 'ngram_range_min',
                       'ngram_range_max'],
        'parameters_types': {'min_df': float_or_int, 'max_df': float_or_int,
                             'ngram_range_min': int, 'ngram_range_max': int},
        'default': '',
        'defaults': {}
    },
    'Tfidf': {
        'mthd': get_tfidf_vectorizer,
        'parameters': ['charset', 'charset_error',
                       'strip_accents', 'lowercase',
                       'analyzer', 'stop_words',
                       'token_pattern', 'max_df',
                       'min_df', 'max_features',
                       'vocabulary', 'binary',
                       'use_idf', 'smooth_idf',
                       'sublinear_tf', 'ngram_range_min',
                       'ngram_range_max'],
        'parameters_types': {'min_df': float_or_int, 'max_df': float_or_int,
                            'ngram_range_min': int, 'ngram_range_max': int},
        'default': '',
        'defaults': {}
    },
    'Lda': {
        'mthd': get_lda_vectorizer,
        'parameters': ['charset', 'charset_error',
                        'strip_accents', 'lowercase',
                        'stop_words', 'token_pattern',
                        'analyzer', 'max_df', 'min_df',
                        'max_features', 'vocabulary',
                        'binary',
                        'num_topics', 'id2word', 'alpha',
                        'eta', 'distributed', 'topic_file'],
        'parameters_types': {'min_df': float_or_int, 'max_df': float_or_int},
        'default': '',
        'defaults': {}
    },
    'Lsi': {
        'mthd': get_lsi_vectorizer,
        'parameters': ['charset', 'charset_error',
                        'strip_accents', 'lowercase',
                        'stop_words', 'token_pattern',
                        'analyzer', 'max_df', 'min_df',
                        'max_features', 'vocabulary',
                        'binary',
                        'num_topics', 'id2word',
                        'distributed', 'onepass',
                        'power_iters', 'extra_samples',
                        'topic_file'],
        'parameters_types': {'min_df': float_or_int, 'max_df': float_or_int},
        'default': '',
        'defaults': {}
    }
}
