# Authors: Nikolay Melnik <nmelnik@upwork.com>

from copy import deepcopy
import numpy
from scipy.sparse import csc_matrix

from utils import copy_expected, float_or_int, parse_parameters
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import LdaVectorizer, LsiVectorizer
from sklearn.preprocessing import StandardScaler


class Ntile(object):
    def __init__(self, number_tile):
        self.number_tile = number_tile

    def fit(self, X):
        """
        X: list
        """
        tile_size = len(X) / self.number_tile
        extra_first_tile = len(X) % self.number_tile
        X = deepcopy(X)
        X.sort()
        self.ranges = []
        n = extra_first_tile - 1
        for i in range(self.number_tile):
            n = n + tile_size
            self.ranges.append(X[n])

    def fit_transform(self, X, **fit_params):
        """
        X: list
        """
        self.fit(X)
        return self.transform(X)

    def transform(self, X):
        """
        X : list
        """
        for i, v in enumerate(X):
            previous = 0
            for n in range(self.number_tile):
                if v > previous and v <= self.ranges[n]:
                    X[i] = n + 1
                    break
                previous = self.ranges[n]
            if X[i] != n + 1:
                X[i] = n + 1
        import numpy
        import scipy.sparse
        return numpy.transpose(
            scipy.sparse.csc_matrix([0.0 if item is None else float(item)
                                     for item in X]))


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

        params: dict
            a map containing the scaler's configuration.
        """
        filtered_params = copy_expected(
            config['params'], ['copy', 'with_mean', 'with_std'])

        self._scaler = StandardScaler(**filtered_params)

    def _to_column(self, X):
        """
        Converts input list X to numpy column.

        X: the list to convert to column
        """

        return numpy.transpose(
            csc_matrix([0.0 if item is None else float(item) for item in X]))

    def fit(self, X, y=None):
        """
        Invokes scaler's fit method, converting first X to column.

        X: list
        y: numpy array of shape [n_samples]

        """
        return self._scaler.fit(self._to_column(X), y)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Invokes scaler's fit_transform method, converting first X to column.

        X: list
        y: numpy array of shape [n_samples]

        """
        return self._scaler.fit_transform(self._to_column(X), y, **fit_params)

    def transform(self, X):
        """
        Invokes scaler's transform method, converting first X to column.

        X: list
        """
        return self._scaler.transform(self._to_column(X))


class SuppressTransformer:
    """
    A vectorizer that suppresses the input feature.
    """
    # TODO: Make it a sublcass of vectorizer?


def get_transformer(transformer):
    if transformer is None:
        return None

    transformer_type = transformer.get('type', None)
    if transformer_type is None:
        return None

    if transformer_type not in TRANSFORMERS:
        return None

    settings = TRANSFORMERS[transformer_type]
    params = parse_parameters(transformer, settings)
    return settings['class'](**params)


class Params:
    input_ = {'name': 'input', 'type': 'string',
              'choices': ['filename', 'file', 'content']}
    encoding = {'name': "encoding", 'type': 'string', 'default': 'utf-8'}
    decode_error = {'name': "decode_error", 'type': 'string',
                    'choices': ['strict', 'ignore', 'replace']}
    strip_accents = {'name': "strip_accents", 'type': 'string',
                     'choices': ['ascii', 'unicode', None]}
    analyzer = {'name': "analyzer", 'type': 'string',
                'choices': ['word', 'char', 'char_wb']}
    # {'name': "preprocessor", 'type': 'callable'},  # TODO
    # {'name': "tokenizer", 'type': 'callable'},
    ngram_range_min = {'name': "ngram_range_min", 'type': 'integer'}
    ngram_range_max = {'name': "ngram_range_max", 'type': 'integer'}
    # {'name': "stop_words", 'type': 'string_list_none'}, # TODO
    lowercase = {'name': "lowercase", 'type': 'boolean', 'default': True}
    token_pattern = {'name': "token_pattern", 'type': 'string'}
    max_df = {'name': "max_df", 'type': 'float_or_int'}
    min_df = {'name': "min_df", 'type': 'float_or_int'}
    max_features = {'name': "max_features", 'type': 'integer'}
    # {'name': "vocabulary", 'type': 'mapping_iterable'},  # TODO
    binary = {'name': "binary", 'type': 'boolean', 'default': False}
    # {'name': "dtype", 'type': 'type'},  # TODO

    COUNT_PARAMS = [input_, encoding, decode_error,
                    strip_accents, analyzer, ngram_range_min,
                    ngram_range_max, lowercase, token_pattern,
                    max_df, min_df, max_features, binary]

    norm = {'name': "norm", 'type': 'string',
            'choices': ['l1', 'l2']}
    use_idf = {'name': "use_idf", 'type': 'boolean'}
    smooth_idf = {'name': "smooth_idf", 'type': 'boolean'}
    sublinear_tf = {'name': "sublinear_tf", 'type': 'boolean'}

    num_topics = {'name': "num_topics", 'type': 'integer'}
    alpha = {'name': 'alpha', 'type': 'float'}  # TODO: float_vector
    # id2word  # TODO
    eta = {'name': 'eta', 'type': 'float'}  # TODO: float_vector
    distributed = {'name': 'distributed', 'type': 'boolean'}
    topic_file = {'name': 'topic_file', 'type': 'string',
                  'default': 'lda_topics.txt'}

    onepass = {'name': 'onepass', 'type': 'boolean'}
    power_iters = {'name': 'power_iters', 'type': 'integer'}
    extra_samples = {'name': 'extra_samples', 'type': 'integer'}


TRANSFORMERS = {
    'Dictionary': {
        'class': DictVectorizer,
        'parameters': [
            {'name': "separator", 'type': 'string'},
            {'name': "sparse", 'type': 'boolean'},
            {'name': "sort", 'type': 'boolean'}
        ],
        'default': {},  # default value
        'defaults': {}  # default values of the parameters
    },
    'Count': {
        'class': CountVectorizer,
        'parameters': Params.COUNT_PARAMS,
        'default': '',
        'defaults': {}
    },
    'Tfidf': {
        'class': TfidfVectorizer,
        'parameters': Params.COUNT_PARAMS + [
            Params.norm, Params.use_idf, Params.smooth_idf,
            Params.sublinear_tf],
        'default': '',
        'defaults': {}
    },
    'Lda': {
        'class': LdaVectorizer,
        'parameters': Params.COUNT_PARAMS + [
            Params.num_topics, Params.alpha, Params.eta,
            Params.distributed, Params.topic_file],
        'default': '',
        'defaults': {}
    },
    'Lsi': {
        'class': LsiVectorizer,
        'parameters': Params.COUNT_PARAMS + [
            Params.num_topics, Params.distributed, Params.onepass,
            Params.power_iters, Params.extra_samples],
        'default': '',
        'defaults': {}
    },
    'Ntile': {
        'class': Ntile,
        'parameters': [{'name': 'number_tile', 'type': 'integer'}],
        'default': '',
        'defaults': {}
    }
}