
_author__ = 'ifoukarakis'

import json
import logging
import numpy

from collections import defaultdict
from time import gmtime, strftime
from operator import itemgetter
from scipy.sparse import hstack, csc_matrix
from featuretype import FEATURE_TYPE_DEFAULTS
from transformers import TRANSFORMER_DEFAULTS
from utils import is_empty

from config import FeatureModel, SchemaException
from metrics import ClassificationModelMetrics, RegressionModelMetrics


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

    def train(self, iterator, percent=0):
        """
        Train the model using the given data. Appropriate SciPy vectorizers
        will be used to bring data to the appropriate format.

        Keyword arguments:
        data_iter -- an iterator that provides a dictionary with keys feature
                    names and value the values of the features per column.

        """
        vectorized_data = []
        labels = None

        self._prepare_data(iterator)
        if percent:
            self._count = self._count - int(self._count * percent / 100)
            for item in self._vect_data:
                item = item[:self._count]
        logging.info('Processed %d lines, ignored %s lines'
                     % (self._count, self._ignored))

        # Get X and y
        logging.info('Extracting features...')
        for feature_name, feature in self._feature_model.features.iteritems():
            if feature_name != self._feature_model.target_variable:
                item = self._train_prepare_feature(
                    feature,
                    self._vect_data[feature_name])
                vectorized_data.append(item)
            else:
                labels = self._vect_data[feature_name]
        logging.info('Training model...')
        true_data = hstack(vectorized_data)
        logging.info('Number of features: %s' % (true_data.shape[1], ))
        self._classifier.fit(true_data, labels)
        self.train_time = strftime('%Y-%m-%d %H:%M:%S %z', gmtime())
        logging.info('Training completed...')

    def test(self, iterator, percent=0):
        """
        Test the model using the given data. SciPy vectorizers that were
        populated with data during testing will be used.

        Keyword arguments:
        iterator -- an iterator that provides a dictionary with keys feature
                    names and value the values of the features per column.

        """
        vectorized_data = []
        labels = None

        self._prepare_data(iterator)
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
                vectorized_data.append(item)
            else:
                labels = self._vect_data[feature_name]

        logging.info('Evaluating model...')
        metr = self.metrics_class(labels, vectorized_data,
                                  self._classifier)
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
        labels = None
        probs = None

        self._prepare_data(iterator, callback, ignore_error)
        logging.info('Processed %d lines, ignored %s lines'
                     % (self._count, self._ignored))
        if self._ignored == self._count:
            logging.info("Don't have valid records")
            return {'probs': probs, 'labels': labels}
        # Get X and y
        logging.info('Extracting features...')
        for feature_name, feature in self._feature_model.features.iteritems():
            if feature_name != self._feature_model.target_variable:
                item = self._test_prepare_feature(
                    feature,
                    self._vect_data[feature_name])
                vectorized_data.append(item)
            else:
                labels = self._vect_data[feature_name]

        logging.info('Evaluating model...')
        probs = self._classifier.predict_proba(hstack(vectorized_data))
        return {'probs': probs, 'labels': labels}

    def _train_prepare_feature(self, feature, data):
        """
        Uses the appropriate vectorizer or scaler on a specific feature and its
        training data.

        Keyword arguments:
        feature -- the name of the feature to prepare. Used to retrieve the
                   appropriate vectorizer.
        data -- a list of the data for extracted for the given feature.

        """
        logging.debug('Preparing feature %s for train' % (feature['name'], ))
        if feature['transformer'] is not None:
            # Bug here: If transformer is a scaler, we need to transpose data.
            return feature['transformer'].fit_transform(data)
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
        if feature['transformer'] is not None:
            return feature['transformer'].transform(data)
        else:
            return self._to_column(data)

    def _to_column(self, x):
        return numpy.transpose(
            csc_matrix([0.0 if item is None else float(item) for item in x]))

    def _prepare_data(self, iterator, callback=None, ignore_error=True):
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
                self._raw_data.append(data)
                for feature_name in self._feature_model.features:
                    self._vect_data[feature_name].append(data[feature_name])

                if callback is not None:
                    callback(row)
            except ItemParseException, e:
                logging.debug('Ignoring item #%d: %s'
                              % (self._count, e.message))
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

            if ft is not None:
                try:
                    result[feature_name] = ft.transform(item)
                except Exception, e:
                    logging.warn('Error processing feature %s: %s'
                                 % (feature_name, e.message))
                    raise ItemParseException('Error processing feature %s: %s'
                                             % (feature_name, e.message))
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
                result = TRANSFORMER_DEFAULTS[feature['transformer-type']]
            elif feature.get('type') is not None:
                result = FEATURE_TYPE_DEFAULTS.get(feature['type'], value)

        return result

    def get_weights(self):
        positive = []
        negative = []
        index = 0

        for feature_name, feature in self._feature_model.features.items():
            if feature_name != self._feature_model.target_variable:
                transformer = feature['transformer']
                if transformer is not None and hasattr(transformer,
                                                       'get_feature_names'):
                    # Vectorizer
                    feature_names = transformer.get_feature_names()
                    for j in range(0, len(feature_names)):
                        name = '%s.%s' % (feature_name, feature_names[j])
                        weight = self._classifier.coef_[0][index + j]
                        weights = {
                            'name': name,
                            'weight': weight
                        }
                        if weight > 0:
                            positive.append(weights)
                        else:
                            negative.append(weights)

                    index += len(feature_names)
                else:
                    # Scaler or array
                    weight = self._classifier.coef_[0][index]
                    weights = {
                        'name': feature_name,
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
