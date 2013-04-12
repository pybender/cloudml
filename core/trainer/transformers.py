import numpy
from scipy.sparse import csc_matrix

__author__ = 'ifoukarakis'

from utils import copy_expected
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler


class ScalerDecorator(object):
    """
    Decorator for scaler. Scalers don't offer the same interface for fit and
    fit_transform, as they require converting data to columns before
    processing.

    """
    def __init__(self, params):
        """
        Creates a ScalerDecorator. Uses the config under the transformer JSON
        object. Internally, a scaler is created using the given params.

        Keyword arguments:
        params -- a map containing the scaler's configuration.

        """
        filtered_params = copy_expected(params, ['copy', 'with_mean',
                                                 'with_std'])

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


def get_count_vectorizer(params):
    """
    Creates a CountVectorizer. Uses the config under the transformer JSON
    object.

    Keyword arguments:
    params -- a map containing the vectorizer's configuration.

    """

    filtered_params = copy_expected(params, ['charset', 'charset_error',
                                             'strip_accents', 'lowercase',
                                             'stop_words', 'token_pattern',
                                             'analyzer', 'max_df', 'min_df',
                                             'max_features', 'vocabulary',
                                             'binary'])
    if 'ngram_range_min' in params and 'ngram_range_max' in params:
        filtered_params['ngram_range'] = (params['ngram_range_min'],
                                          params['ngram_range_max'])
    return CountVectorizer(**filtered_params)


def get_tfidf_vectorizer(params):
    """
    Creates a TfidfVectorizer. Uses the config under the transformer JSON
    object.

    Keyword arguments:
    params -- a map containing the vectorizer's configuration.

    """
    filtered_params = copy_expected(params, ['charset', 'charset_error',
                                             'strip_accents', 'lowercase',
                                             'analyzer', 'stop_words',
                                             'token_pattern', 'max_df',
                                             'min_df', 'max_features',
                                             'vocabulary', 'binary',
                                             'use_idf', 'smooth_idf',
                                             'sublinear_tf'])
    if 'ngram_range_min' in params and 'ngram_range_max' in params:
        filtered_params['ngram_range'] = (params['ngram_range_min'],
                                          params['ngram_range_max'])

    return TfidfVectorizer(**filtered_params)


def get_dict_vectorizer(params):
    """
    Creates a DictVectorizer. Uses the config under the vectorizer-config.

    Keyword arguments:
    params -- a map containing the vectorizer's configuration.

    """
    filtered_params = copy_expected(params, ['separator', 'sparse'])

    return DictVectorizer(**filtered_params)


def get_scaler(params):
    """
    Creates a scaler. Uses the config under the transformer JSON object.

    Keyword arguments:
    params -- a map containing the vectorizer's configuration.

    """
    #filtered_params = copy_expected(params, ['copy', 'with_mean', 'with_std'])

    return ScalerDecorator(params)


def get_transformer(transformer):
    if transformer is None:
        return None

    transformer_type = transformer.get('type', None)
    if transformer_type is None:
        return None

    if transformer_type not in TRANSFORMER_TO_VECTORIZER:
        return None

    return TRANSFORMER_TO_VECTORIZER[transformer_type](transformer)


TRANSFORMER_TO_VECTORIZER = {
    'Dictionary': get_dict_vectorizer,
    'Count': get_count_vectorizer,
    'Tfidf': get_tfidf_vectorizer,
    'Scale': get_scaler
}

# Default values per transformer type
TRANSFORMER_DEFAULTS = {
    'Dictionary': {},
    'Count': '',
    'Tfidf': '',
    'Scale': 0.0
}
