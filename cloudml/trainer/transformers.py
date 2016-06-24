# Authors: Nikolay Melnik <nmelnik@cloud.upwork.com>

import numpy
from copy import deepcopy
from scipy.sparse import csc_matrix
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import LdaVectorizer, LsiVectorizer, \
    Word2VecVectorizer, Doc2VecVectorizer
from sklearn.preprocessing import StandardScaler

from utils import copy_expected, parse_parameters, set_defaults, \
    set_params_defaults


class Ntile(object):
    def __init__(self, number_tile=5):
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
              'choices': ['filename', 'file', 'content'], 'default': 'content'}
    encoding = {'name': "encoding", 'type': 'string', 'default': 'utf-8'}
    decode_error = {'name': "decode_error", 'type': 'string',
                    'choices': ['strict', 'ignore', 'replace'],
                    'default': 'strict'}
    strip_accents = {'name': "strip_accents", 'type': 'string',
                     'choices': ['ascii', 'unicode', None], 'default': None}
    analyzer = {'name': "analyzer", 'type': 'string',
                'choices': ['word', 'char', 'char_wb'], 'default': 'word'}
    # {'name': "preprocessor", 'type': 'callable'},  # TODO
    # {'name': "tokenizer", 'type': 'callable'},
    ngram_range_min = {'name': "ngram_range_min", 'type': 'integer',
                       'default': 1}
    ngram_range_max = {'name': "ngram_range_max", 'type': 'integer',
                       'default': 1}
    stop_words = {'name': "stop_words", 'type': 'string_list_none',
                  'default': None}
    lowercase = {'name': "lowercase", 'type': 'boolean', 'default': True}
    token_pattern = {'name': "token_pattern", 'type': 'string',
                     'default': r"(?u)\b\w\w+\b"}
    max_df = {'name': "max_df", 'type': 'float_or_int', 'default': 1.0}
    min_df = {'name': "min_df", 'type': 'float_or_int', 'default': 1}
    max_features = {'name': "max_features", 'type': 'integer', 'default': None}
    # {'name': "vocabulary", 'type': 'mapping_iterable'},  # TODO
    binary = {'name': "binary", 'type': 'boolean', 'default': False}
    # {'name': "dtype", 'type': 'type'},  # TODO

    COUNT_PARAMS = [input_, encoding, decode_error,
                    strip_accents, analyzer, ngram_range_min,
                    ngram_range_max, lowercase, token_pattern,
                    binary, max_df, min_df, max_features, stop_words]

    norm = {'name': "norm", 'type': 'string',
            'choices': ['l1', 'l2'], 'default': 'l2'}
    use_idf = {'name': "use_idf", 'type': 'boolean', 'default': True}
    smooth_idf = {'name': "smooth_idf", 'type': 'boolean', 'default': True}
    sublinear_tf = {'name': "sublinear_tf", 'type': 'boolean',
                    'default': False}

    num_topics = {'name': "num_topics", 'type': 'integer', 'default': 100}
    alpha = {'name': 'alpha', 'type': 'float',
             'default': None}  # TODO: float_vector
    # id2word  # TODO
    eta = {'name': 'eta', 'type': 'float',
           'default': None}  # TODO: float_vector
    distributed = {'name': 'distributed', 'type': 'boolean', 'default': False}
    chunksize = {'name': 'chunksize', 'type': 'integer', 'default': 2000}
    passes = {'name': 'passes', 'type': 'integer', 'default': 1}
    update_every = {'name': 'update_every', 'type': 'integer', 'default': 1}
    decay = {'name': 'decay', 'type': 'float', 'default': 0.5}
    onepass = {'name': 'onepass', 'type': 'boolean', 'default': True}
    power_iters = {'name': 'power_iters', 'type': 'integer', 'default': 2}
    extra_samples = {'name': 'extra_samples', 'type': 'integer',
                     'default': 100}

    # Word2Vec and Doc2Vec params
    train_algorithm_w2v = {'name': 'train_algorithm', 'type': 'string',
                           'choices': ['skip-gram', 'cbow'],
                           'default': 'skip-gram'}
    train_algorithm_d2v = {'name': 'train_algorithm', 'type': 'string',
                           'choices': ['pv-dm', 'pv-dbow', 'both'],
                           'default': 'pv-dm'}
    vector_size = {'name': 'vector_size', 'type': 'integer', 'default': 100}
    window = {'name': 'window', 'type': 'integer', 'default': 5,
              'help_text': 'Maximum distance between current and predicted '
                           'word within a sentence'}
    min_count = {'name': 'min_count', 'type': 'integer', 'default': 5}
    max_vocab_size = {'name': 'max_vocab_size', 'type': 'integer',
                      'default': None}
    vec_alpha = {'name': 'alpha', 'type': 'float', 'default': 0.025,
                 'help_text': 'Initial learning rate (will linearly drop to '
                              'zero as training progresses)'}
    min_alpha = {'name': 'min_alpha', 'type': 'float',
                 'default': 0.0001}
    seed = {'name': 'seed', 'type': 'string', 'default': '1',
            'help_text': 'Used for the random number generator. Initial '
                         'vectors for each word are seeded with a hash of the '
                         'concatenation of word + str(seed)'}
    hierarchical_sampling = {'name': 'hierarchical_sampling',
                             'type': 'boolean', 'default': True}
    negative_sampling = {'name': 'negative_sampling', 'type': 'boolean',
                         'default': False}
    cbow_mean = {'name': 'cbow_mean', 'type': 'boolean', 'default': False}
    null_word = {'name': 'null_word', 'type': 'boolean', 'default': False}

    # TODO: hash by default, probably more functions should be added
    hash_function = {'name': 'hash_function', 'type': 'string',
                     'choices': ['hash'], 'default': 'hash'}
    iterations = {'name': 'iterations', 'type': 'integer', 'default': 1,
                  'help_text': 'Number of iterations over the corpus'}

    # TODO: None by default, probably more functions should be added. Not used
    # it should be a function and take 3 params (word, count, min_count)
    # and return:
    # 2 - to keep word, 1 - discard word,
    # 0 - use default word2vec behavior (something like don't know)
    trim_rule = {'name': 'trim_rule', 'type': 'string',
                 'choices': ['callable']}
    sample = {'name': 'sample', 'type': 'float', 'default': 0,
              'help_text': 'Threshold for configuring which higher-frequency '
                           'words are randomly downsampled; default is 0 (off)'
                           ', useful value is 1e-5'}
    workers = {'name': 'workers', 'type': 'integer', 'default': 1}
    sorted_vocab = {'name': 'sorted_vocab', 'type': 'boolean', 'default': True}

    WORD2VEC_PARAMS = [stop_words, vector_size, window, min_count,
                       seed, hierarchical_sampling, negative_sampling,
                       workers, iterations, sample, vec_alpha, min_alpha,
                       hash_function, sorted_vocab, max_vocab_size]
    dbow_words = {'name': 'dbow_words', 'type': 'boolean', 'default': False}
    dm_mean = {'name': 'dm_mean', 'type': 'boolean', 'default': False}
    dm_concat = {'name': 'dm_concat', 'type': 'boolean', 'default': False}
    dm_tag_count = {'name': 'dm_tag_count', 'type': 'integer', 'default': 1}
    retrain_count = {'name': 'retrain_count', 'type': 'integer', 'default': 10,
                     'help_text': 'Shuffle the dataset and retrain doc2vec '
                                  'model count'}


TRANSFORMERS = {
    'Dictionary': {
        'class': DictVectorizer,
        'parameters': [
            {'name': "separator", 'type': 'string', 'default': '='},
            {'name': "sparse", 'type': 'boolean', 'default': True},
            {'name': "sort", 'type': 'boolean', 'default': True}
        ],
        'default': '',  # default value
        'defaults': {}
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
            Params.distributed, Params.chunksize, Params.decay, Params.passes,
            Params.update_every],
        'default': '',
        'defaults': {'stop_words': 'english'}
    },
    'Lsi': {
        'class': LsiVectorizer,
        'parameters': Params.COUNT_PARAMS + [
            Params.num_topics, Params.distributed,
            Params.onepass, Params.power_iters, Params.extra_samples,
            Params.decay, Params.chunksize],
        'default': '',
        'defaults': {'stop_words': 'english', 'decay': 1.0,
                     'chunk_size': 20000}
    },
    'Ntile': {
        'class': Ntile,
        'parameters': [{'name': 'number_tile', 'type': 'integer',
                        'default': 5}],
        'default': '',
        'defaults': {}
    },
    'Word2Vec': {
        'class': Word2VecVectorizer,
        'parameters': Params.COUNT_PARAMS[:10] + Params.WORD2VEC_PARAMS + [
            Params.train_algorithm_w2v, Params.null_word, Params.cbow_mean],
        'default': '',
        'defaults': {}
    },
    'Doc2Vec': {
        'class': Doc2VecVectorizer,
        'parameters': Params.COUNT_PARAMS[:10] + Params.WORD2VEC_PARAMS + [
            Params.train_algorithm_d2v, Params.dbow_words, Params.dm_mean,
            Params.dm_concat, Params.dm_tag_count, Params.retrain_count],
        'default': '',
        'defaults': {'token_pattern': r"(?u)[\b\w\w+\b]+|[.,!?();:\"{}\[\]]",
                     'vector_size': 300, 'window': 8}
    }
}

for item, settings in TRANSFORMERS.iteritems():
    TRANSFORMERS[item]['defaults'] = set_defaults(settings['defaults'],
                                                  settings['parameters'])
    TRANSFORMERS[item]['parameters'] = set_params_defaults(
        settings['parameters'], settings['defaults'])