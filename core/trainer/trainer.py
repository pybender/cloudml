
_author__ = 'ifoukarakis, papadimitriou'

import json
import logging
import numpy

from collections import defaultdict
from time import gmtime, strftime
from operator import itemgetter
from scipy.sparse import hstack, csc_matrix
from feature_types import FEATURE_TYPE_DEFAULTS
from transformers import TRANSFORMERS, SuppressTransformer
from utils import is_empty

from config import FeatureModel, SchemaException
from metrics import ClassificationModelMetrics, RegressionModelMetrics

#from memory_profiler import profile

class ItemParseException(Exception):
    """
    Exception to be raised if there is an error parsing an item according to
    its feature type

    """

    def __init__(self, message, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        # Now for your custom code...
        self.Errors = Errors


class InvalidTrainerFile(Exception):
    """
    Exception to be raised if trainer could not be unpickled from file.
    """
    pass


class Trainer():
    TYPE_CLASSIFICATION = 'classification'
    TYPE_REGRESSION = 'regression'

    def __init__(self, feature_model):
        """
        Initializes the trainer using the application's configuration. Creates
        classifier, instantiates appropriate vectorizers and scalers for all
        features and prints initialization messages.

        Keyword arguments:
        feature_model -- the feature model object (see class FeatureModel).

        """
        # Classifier to use
        self._classifier_type = feature_model.classifier_type
        logging.info('Using "%s"', self._classifier_type)
        classifier_cls = feature_model.classifier_cls
        self._classifier = classifier_cls(**feature_model.classifier)

        # Remember configuration
        self._feature_model = feature_model
        self._count = 0
        self._ignored = 0
        self.train_time = None

    def set_classifier(self, classifier):
        self._classifier = classifier

    def clear_temp_data(self):
        if hasattr(self, '_raw_data'):
            self._raw_data = None
        if hasattr(self, '_vect_data'):
            self._vect_data = None

    # TODO: Did you mean this?
    @property
    def model_type(self):
        """
        Specifies whether a model is a classification or regression model.
        """
        from classifier_settings import CLASSIFIER_MODELS, REGRESSION_MODELS
        if self._classifier_type in CLASSIFIER_MODELS:
            return self.TYPE_CLASSIFICATION
        elif self._classifier_type in REGRESSION_MODELS:
            return self.TYPE_REGRESSION

    @property
    def metrics_class(self):
        if self.model_type == self.TYPE_CLASSIFICATION:
            return ClassificationModelMetrics
        elif self.model_type == self.TYPE_REGRESSION:
            return RegressionModelMetrics
        else:
            raise NotImplemented('Calculating metrics only for classification\
 and regression model implemented')

    def train(self, iterator=None, percent=0):
        """
        Train the model using the given data. Appropriate SciPy vectorizers
        will be used to bring data to the appropriate format.

        Keyword arguments:
        data_iter -- an iterator that provides a dictionary with keys feature
                    names and value the values of the features per column.

        """
        vectorized_data = []
        labels = None
        from memory_profiler import memory_usage
        logging.info("Memory usage: %f" % 
                     memory_usage(-1, interval=0, timeout=None)[0])

        if iterator:
            self._prepare_data(iterator)
        if percent:
            self._count = self._count - int(self._count * percent / 100)
            for item in self._vect_data:
                item = item[:self._count]
        logging.info('Processed %d lines, ignored %s lines'
                     % (self._count, self._ignored))
        logging.info("Memory usage: %f" % 
                     memory_usage(-1, interval=0, timeout=None)[0])
        
        # Get X and y
        logging.info('Extracting features...')
        for feature_name, feature in self._feature_model.features.iteritems():
            if feature_name != self._feature_model.target_variable:
                
                item = self._train_prepare_feature(
                    feature,
                    self._vect_data[feature_name])

                if item is not None:
                    # Convert item to csc_matrix, since hstack fails with arrays
                    vectorized_data.append(csc_matrix(item))
                logging.info("Memory usage: %f" % 
                             memory_usage(-1, interval=0, timeout=None)[0])
            else:
                labels = self._vect_data[feature_name]
        logging.info('Training model...')

        logging.info("Memory usage: %f" % 
                     memory_usage(-1, interval=0, timeout=None)[0])
        self._vect_data = None
        logging.info("Memory usage: %f" % 
                     memory_usage(-1, interval=0, timeout=None)[0])
        # if(len(vectorized_data) == 1):
        #     true_data = numpy.array(vectorized_data[0])
        # else:
        true_data = hstack(vectorized_data)
        logging.info("Memory usage: %f" % 
                     memory_usage(-1, interval=0, timeout=None)[0])
        vectorized_data = None
        logging.info("Memory usage: %f" % 
                     memory_usage(-1, interval=0, timeout=None)[0])

        logging.info('Number of features: %s' % (true_data.shape[1], ))
        self._classifier.fit(true_data, labels)
        logging.info("Memory usage: %f" % 
                     memory_usage(-1, interval=0, timeout=None)[0])
        true_data = None
        self.train_time = strftime('%Y-%m-%d %H:%M:%S %z', gmtime())
        logging.info('Training completed...')

    def test(self, iterator, percent=0, callback=None, save_raw=True):
        """
        Test the model using the given data. SciPy vectorizers that were
        populated with data during testing will be used.

        Keyword arguments:
        iterator -- an iterator that provides a dictionary with keys feature
                    names and value the values of the features per column.

        """
        from memory_profiler import memory_usage
        vectorized_data = []
        labels = None

        self._prepare_data(iterator, callback, save_raw=save_raw)
        count = self._count
        if percent:
            self._count = int(self._count * percent / 100)
            for item in self._vect_data:
                item = item[count - self._count:]
        logging.info('Processed %d lines, ignored %s lines'
                     % (self._count, self._ignored))

        # Get X and y
        logging.info('Extracting features...')
        for feature_name, feature in self._feature_model.features.iteritems():
            if feature_name != self._feature_model.target_variable:
                item = self._test_prepare_feature(
                    feature,
                    self._vect_data[feature_name])
                if item is not None:
                    # Convert item to csc_matrix, since hstack fails with arrays
                    vectorized_data.append(item)
            else:
                labels = self._vect_data[feature_name]

        self._vect_data = None
        logging.info("Memory usage: %f" % 
                     memory_usage(-1, interval=0, timeout=None)[0])
        logging.info('Evaluating model...')
        metr = self.metrics_class(labels, vectorized_data,
                                  self._classifier)
        logging.info("Memory usage: %f" % 
                     memory_usage(-1, interval=0, timeout=None)[0])
        metr.log_metrics()
        return metr

    def predict(self, iterator, callback=None, ignore_error=True):
        """
        Attempts to predict the class of each of the data in the given
        iterator. Returns the predicted values, the target feature value (if
        present).

        Keyword arguments:
        iterator -- an iterator that provides a dictionary with keys feature
                    names and value the values of the features per column.
        callback -- callback function to invoke on every row coming from the
                    iterator and is not ignored.

        """

        vectorized_data = []
        true_labels = None
        labels = None
        probs = None

        self._prepare_data(iterator, callback, ignore_error)
        logging.info('Processed %d lines, ignored %s lines'
                     % (self._count, self._ignored))
        if self._ignored == self._count:
            logging.info("Don't have valid records")
        else:
            # Get X and y
            logging.info('Extracting features...')
            for feature_name, feature in self._feature_model.features.iteritems():
                if feature_name != self._feature_model.target_variable:
                    item = self._test_prepare_feature(
                        feature,
                        self._vect_data[feature_name])
                    vectorized_data.append(csc_matrix(item))
                else:
                    true_labels = self._vect_data[feature_name]

            logging.info('Evaluating model...')
            if(len(vectorized_data) == 1):
                predict_data = numpy.array(vectorized_data[0])
            else:
                predict_data = hstack(vectorized_data)
            probs = self._classifier.predict_proba(predict_data)
            labels = self._classifier.classes_[probs.argmax(axis=1)]
        return {'probs': probs,
                'true_labels': true_labels,
                'labels': labels,
                'classes': self._classifier.classes_}

    def _process_subfeatures(self, feature, data):
        from collections import defaultdict
        sub_feature_names = []
        sub_features = defaultdict(list)
        default = feature['type'].transform(None)
        for x in data:
            sub_feature_names = sub_feature_names + x.keys()
        for x in data:
            for sub_feature in set(sub_feature_names):
                sub_features[sub_feature].append(x.get(sub_feature, default))
        trans_sub_features = []
        for k, v in sub_features.iteritems():
            if feature['transformer'] is not None:
                trans_sub_features.append(feature['transformer'].fit_transform(v))
            else:
                trans_sub_features.append(self._to_column(v))
        return trans_sub_features

    def _train_prepare_feature(self, feature, data):
        """
        Uses the appropriate vectorizer or scaler on a specific feature and its
        training data.

        Keyword arguments:
        feature -- the name of the feature to prepare. Used to retrieve the
                   appropriate vectorizer.
        data -- a list of the data for extracted for the given feature.

        """
        logging.info('Preparing feature %s for train' % (feature['name'], ))
        input_format = feature.get('input-format', None)
        if input_format == 'list':
            data = map(lambda x: " ".join(x) if isinstance(x, list) else x, data)
        if feature['type'].preprocessor:
            return feature['type'].preprocessor.fit_transform(data)

        if feature['transformer'] is not None:
            try:
                transformed_data = feature['transformer'].fit_transform(data)
                feature['transformer'].num_features = transformed_data.shape[1]
            except ValueError as e:
                logging.warn('Feature %s will be ignored due to '
                             'transformation error: %s.' %
                              (feature['name'], str(e)))
                transformed_data = None
                feature['tranformer'] = SuppressTransformer()
            return transformed_data
        elif feature.get('scaler', None) is not None:
            return feature['scaler'].fit_transform(self._to_column(data).toarray())
        else:
            return self._to_column(data)

    def _test_prepare_feature(self, feature, data):
        """
        Uses the appropriate vectorizer or scaler on a specific feature and its
        test data.

        Keyword arguments:
        feature -- the name of the feature to prepare. Used to retrieve the
                   appropriate vectorizer.
        data -- a list of the data extracted for the given feature.

        """
        logging.debug('Preparing feature %s for test' % (feature['name'], ))
        input_format = feature.get('input-format', None)
        if input_format == 'list':
            data = map(lambda x: " ".join(x) if isinstance(x, list) else x, data)

        if feature['type'].preprocessor:
            return feature['type'].preprocessor.transform(data)
        if feature['transformer'] is not None:
            if isinstance(feature['transformer'], SuppressTransformer):
                return None
            else:
                return feature['transformer'].transform(data)
        elif feature.get('scaler', None) is not None:
            return feature['scaler'].transform(self._to_column(data).toarray())
        else:
            return self._to_column(data)

    def _to_column(self, x):
        return numpy.transpose(
            csc_matrix([0.0 if item is None else float(item) for item in x]))

    def _prepare_data(self, iterator, callback=None, 
                      ignore_error=True, save_raw=False):
        """
        Iterates over input data and stores them by column, ignoring lines
        with required properties missing.

        Keyword arguments:
        iterator -- an iterator providing the rows to use for reading the data.
        callback -- function to invoke on each row of data

        """
        self._count = 0
        self._ignored = 0
        self._raw_data = []
        self._vect_data = defaultdict(list)
        for row in iterator:
            self._count += 1
            try:
                data = self._apply_feature_types(row)
                if save_raw:
                    self._raw_data.append(row)
                for feature_name in self._feature_model.features:
                    self._vect_data[feature_name].append(data[feature_name])

                if callback is not None:
                    callback(row)
            except ItemParseException, e:
                logging.debug('Ignoring item #%d: %s'
                              % (self._count, e))
                if ignore_error:
                    self._ignored += 1
                else:
                    raise e

    def _apply_feature_types(self, row_data):
        """
        Apply the transformation dictated by feature type instance (if any).

        Keyword arguments:
        row_data -- current row's data to be processed.

        """
        result = {}
        for feature_name, feature in self._feature_model.features.iteritems():
            ft = feature.get('type', None)
            item = row_data.get(feature_name, None)
            if feature.get('required', True):
                item = self._find_default(item, feature)
            input_format = feature.get('input-format', 'plain')
            if ft is not None:
                try:
                    if input_format == 'plain':
                        result[feature_name] = ft.transform(item)
                    elif input_format == 'dict':
                        if item is None:
                            item = {}
                        for k, v in item.iteritems():
                            item[k] = ft.transform(v)
                        result[feature_name] = item
                    elif input_format == 'list':
                        map(ft.transform, item)
                        result[feature_name] = item
                except Exception as e:
                    logging.warn('Error processing feature %s: %s'
                                 % (feature_name, e))
                    raise ItemParseException('Error processing feature %s: %s'
                                             % (feature_name, e))
            else:
                result[feature_name] = item

        return result

    def _find_default(self, value, feature):
        """
        Checks if value is None or empty (string, list), and attempts to find
        the default value. Priority for default value is based on the
        following:
        1. If 'default' is set in feature model, then this has top priority.
        2. If feature has a transformer, transformer's default is used.
        3. If feature has a type with a default value, this one is used.
        """
        result = value

        if is_empty(value):
            if feature.get('default') is not None:
                result = feature.get('default')
            elif feature.get('transformer-type') is not None:
                result = TRANSFORMERS[feature['transformer-type']]['default']
            elif feature.get('type') is not None:
                result = FEATURE_TYPE_DEFAULTS.get(feature['type'], value)

        return result

    def get_weights_from_vectorizer(self, feature_name, vectorizer, offset):
        positive = []
        negative = []
         # Vectorizer
        feature_names = vectorizer.get_feature_names()
        logging.info('Number of subfeatures %d' % len(feature_names))
        for j in range(0, len(feature_names)):
            name = '%s->%s' % (feature_name.replace(".", "->"), feature_names[j])
            weight = self._classifier.coef_[0][offset + j]
            weights = {
                'name': name,
                'weight': weight
            }
            if weight > 0:
                positive.append(weights)
            else:
                negative.append(weights)
        return feature_names, positive, negative

    def get_weights(self):
        positive = []
        negative = []
        index = 0

        for feature_name, feature in self._feature_model.features.items():
            if feature_name != self._feature_model.target_variable:
                transformer = feature['transformer']
                preprocessor = feature['type'].preprocessor
                logging.info('Process feature %s' % feature_name )
                if transformer is not None and hasattr(transformer, 'num_topics'):
                    logging.info('Number of topics %d' % transformer.num_features )
                    for j in range(0, transformer.num_features-1):
                        name = '%s->Topic #%d' % (feature_name.replace(".", "->"), j)
                        weight = self._classifier.coef_[0][index + j]
                        weights = {
                            'name': name,
                            'weight': weight
                        }
                        if weight > 0:
                            positive.append(weights)
                        else:
                            negative.append(weights)

                    index += transformer.num_topics
                elif transformer is not None and hasattr(transformer,
                                                       'get_feature_names'):
                    feature_names, p, n = self.get_weights_from_vectorizer(feature_name,
                                                                           transformer,
                                                                           index)
                    index += len(feature_names)
                    positive = positive + p
                    negative = negative + n

                elif preprocessor is not None and hasattr(preprocessor,
                                                          'get_feature_names'):
                    feature_names, p, n = self.get_weights_from_vectorizer(feature_name,
                                                                           preprocessor,
                                                                           index)
                    index += len(feature_names)
                    positive = positive + p
                    negative = negative + n
                else:
                    # Scaler or array
                    weight = self._classifier.coef_[0][index]
                    weights = {
                        'name': feature_name.replace(".", "->"),
                        'weight': weight
                    }
                    if weight > 0:
                        positive.append(weights)
                    else:
                        negative.append(weights)
                    index += 1

        positive = sorted(positive, key=itemgetter('name'), reverse=True)
        negative = sorted(negative, key=itemgetter('name'), reverse=False)
        return {'positive': positive, 'negative': negative}

    def store_feature_weights(self, fp):
        """
        Stores the weight of each feature to the given file.

        Keyword arguments:
        fp -- the file to store the trained model.

        """

        json.dump(self.get_weights(), fp, indent=4)


def list_to_dict(user_params):
    if user_params is not None:
        param_list = [x.split('=', 1) for x in user_params]
        return dict((key, value) for (key, value) in param_list)

    return dict()
