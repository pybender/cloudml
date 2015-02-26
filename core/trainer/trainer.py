
_author__ = 'ifoukarakis, papadimitriou'

import json
import logging
import numpy
import csv
import scipy.sparse

from copy import deepcopy
from collections import defaultdict
from time import gmtime, strftime
from operator import itemgetter
from memory_profiler import memory_usage

from feature_types import FEATURE_TYPE_DEFAULTS
from transformers import TRANSFORMERS, SuppressTransformer
from utils import is_empty

from config import FeatureModel, SchemaException
from metrics import ClassificationModelMetrics, RegressionModelMetrics
from model_visualization import TrainedModelVisualizator

DEFAULT_SEGMENT = 'default'


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


class TransformerNotFound(Exception):
    """
    Exception to be raised if predefined transormer could not be found.
    """
    pass


class Trainer(object):
    TYPE_CLASSIFICATION = 'classification'
    TYPE_REGRESSION = 'regression'

    TRAIN_VECT_DATA = 'train_vect_data'

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
        self._classifier = {}
        self._classifier[DEFAULT_SEGMENT] = classifier_cls(
            **feature_model.classifier)

        # Remember configuration
        self._feature_model = feature_model
        self.features = {}
        self._count = 0
        self._ignored = 0
        self.train_time = {}
        self._segments = {}
        self.visualization = {}
        #self._feature_weights = {}

        self.intermediate_data = defaultdict(dict)

        self.model_visualizer = TrainedModelVisualizator.factory(self)

    @property
    def classifier_type(self):
        return self._classifier_type

    @property
    def classifier(self):
        return self.get_classifier()

    @property
    def feature_model(self):
        return self._feature_model

    @property
    def with_segmentation(self):
        return self.feature_model.group_by

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

    def get_classifier(self, segment=DEFAULT_SEGMENT):
        return self._classifier[segment]

    def get_features(self, segment=DEFAULT_SEGMENT):
        return self.features.get(segment)

    def is_target_variable(self, feature_name):
        return feature_name == self.feature_model.target_variable

    def is_group_by_variable(self, feature_name):
        return feature_name in self.feature_model.group_by

    # todo:

    def get_transformer(self, name):
        raise TransformerNotFound

    def set_transformer_getter(self, method):
        self.get_transformer = method

    def set_classifier(self, classifier):
        self._classifier = classifier

    def set_features(self, features):
        self.features = features

    # base methods

    # train model related

    def train(self, iterator=None, percent=0, store_vect_data=False):
        """
        Train the model using the given data. Appropriate SciPy vectorizers
        will be used to bring data to the appropriate format.

        Keyword arguments:
        data_iter -- an iterator that provides a dictionary with keys feature
                    names and value the values of the features per column.

        """
        logging.info("Memory usage: %f" %
                     memory_usage(-1, interval=0, timeout=None)[0])

        if iterator:
            self._segments = self._prepare_data(iterator)

        if self.with_segmentation:
            logging.info(
                'Group by: %s' % ",".join(self.feature_model.group_by))
            logging.info('Segments:')
            for segment, records in self._segments.iteritems():
                logging.info("'%s' - %d records" % (segment, records))

        if percent:
            self._count = self._count - int(self._count * percent / 100)
            for segment in self._vect_data:
                for item in self._vect_data[segment]:
                    item = item[:self._count]
        logging.info('Processed %d lines, ignored %s lines'
                     % (self._count, self._ignored))
        logging.info("Memory usage: %f" %
                     memory_usage(-1, interval=0, timeout=None)[0])
        for segment in self._vect_data:
            logging.info('Starting train "%s" segment' % segment)
            self._train_segment(segment, store_vect_data)

    def _train_segment(self, segment, store_vect_data=False):
        # Get X and y
        logging.info('Extracting features for segment %s ...', segment)

        self.features[segment] = deepcopy(self._feature_model.features)
        labels = self._get_target_variable_labels(segment)
        vectorized_data = self._get_vectorized_data(segment, self._train_prepare_feature)
        logging.info("Memory usage (vectorized data generated): %f" %
                     memory_usage(-1, interval=0, timeout=None)[0])
        logging.info('Training model...')
        #self._vect_data[segment] = None
        #logging.info("Memory usage: %f" %
        #             memory_usage(-1, interval=0, timeout=None)[0])
        # if(len(vectorized_data) == 1):
        #     true_data = numpy.array(vectorized_data[0])
        # else:
        true_data = scipy.sparse.hstack(vectorized_data)
        logging.info("Memory usage (true data generated): %f" %
                     memory_usage(-1, interval=0, timeout=None)[0])
        vectorized_data = None
        logging.info("Memory usage (vectorized data cleared): %f" %
                     memory_usage(-1, interval=0, timeout=None)[0])

        logging.info('Number of features: %s' % (true_data.shape[1], ))
        if segment != DEFAULT_SEGMENT:
            self._classifier[segment] = deepcopy(self._classifier[DEFAULT_SEGMENT])
        self._classifier[segment].fit(true_data, [str(l) for l in labels])
        logging.info("Memory usage (model fitted with true data): %f" %
                     memory_usage(-1, interval=0, timeout=None)[0])

        self.generate_trained_model_visualization(segment, true_data)

        if store_vect_data:
            self.intermediate_data[self.TRAIN_VECT_DATA][segment] = {
                'data': true_data,
                'labels': labels}

        true_data = None
        logging.info("Memory usage (true data cleared): %f" %
                     memory_usage(-1, interval=0, timeout=None)[0])
        self.train_time[segment] = strftime('%Y-%m-%d %H:%M:%S %z', gmtime())
        logging.info('Training completed...')

    def generate_trained_model_visualization(self, segment, true_data):
        self.model_visualizer.generate(segment, true_data)
        self.visualization[segment] = self.model_visualizer.get_visualization(segment)

    def get_visualization(self, segment):
        if not self.visualization or not self.visualization.has_key(segment):
            self.visualization[segment] = self.model_visualizer.get_visualization(segment)
        return self.visualization[segment]

    def get_weights(self, segment=DEFAULT_SEGMENT):
        if not self.visualization or not self.visualization.has_key(segment):
            self.visualization[segment] = self.model_visualizer.get_visualization(segment)
        return self.visualization[segment]['weights']

    # evaluating model related

    def test(self, iterator, percent=0, callback=None, save_raw=True):
        """
        Test the model using the given data. SciPy vectorizers that were
        populated with data during testing will be used.

        Keyword arguments:
        iterator -- an iterator that provides a dictionary with keys feature
                    names and value the values of the features per column.

        """
        self.metrics = self.metrics_class()

        self._prepare_data(iterator, callback, save_raw=save_raw)
        self._test_empty_labels = self._check_data_for_test()
        count = self._count
        if percent:
            self._count = int(self._count * percent / 100)
            for segment in self._vect_data:
                for item in self._vect_data[segment]:
                    item = item[:self._count]
        logging.info('Processed %d lines, ignored %s lines'
                     % (self._count, self._ignored))

        for segment in self._vect_data:
            logging.info('Starting test "%s" segment' % segment)
            self._evaluate_segment(segment)
        self.metrics.log_metrics()

        return self.metrics

    def predict(self, iterator, callback=None, ignore_error=True,
                store_vect_data=False):
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

        labels = {}
        probs = {}
        true_labels = {}
        indexes = defaultdict(int)
        ordered_probs = []
        ordered_labels = []
        ordered_true_labels = []
        self.predict_data = {}

        self._prepare_data(iterator, callback, ignore_error)
        logging.info('Processed %d lines, ignored %s lines'
                     % (self._count, self._ignored))
        if self._ignored == self._count:
            logging.info("Don't have valid records")
        else:
            for segment in self._vect_data:
                vectorized_data = []
                # Get X and y
                logging.info('Extracting features...')
                true_labels[segment] = self._get_target_variable_labels(segment)
                vectorized_data = self._get_vectorized_data(
                    segment, self._test_prepare_feature)
                logging.info('Evaluating model...')
                if len(vectorized_data) == 1:
                    predict_data = numpy.array(vectorized_data[0])
                else:
                    predict_data = scipy.sparse.hstack(vectorized_data)
                if store_vect_data:
                    self.predict_data[segment] = predict_data
                probs[segment] = self._classifier[segment].predict_proba(predict_data)
                labels[segment] = self._classifier[segment].classes_[probs[segment].argmax(axis=1)]
            # Restore order
            for segment in self._order_data:
                indexes[segment] += 1
                ordered_probs.append(probs[segment][indexes[segment]-1])
                ordered_labels.append(labels[segment][indexes[segment]-1])
                ordered_true_labels.append(
                    true_labels[segment][indexes[segment]-1])

        return {'probs': ordered_probs,
                'true_labels': ordered_true_labels,
                'labels': ordered_labels,
                'classes': self._classifier[segment].classes_}

    def grid_search(self, parameters, train_iterator, test_iterator, score=None):
        from sklearn import grid_search
        classifier = self._classifier[DEFAULT_SEGMENT]
        clf = grid_search.GridSearchCV(classifier, parameters, scoring=score)
        results = {}
        if train_iterator:
            self._segments = self._prepare_data(train_iterator)
        for segment in self._vect_data:
            logging.info('Starting search params for "%s" segment' % segment)
            self.features[segment] = deepcopy(self._feature_model.features)
            labels = self._get_target_variable_labels(segment)
            vectorized_data = self._get_vectorized_data(segment, self._train_prepare_feature)
            true_data = scipy.sparse.hstack(vectorized_data)
            clf.fit(true_data, [str(l) for l in labels])
            results[segment] = clf
        return results

    def transform(self, iterator):
        """
        Transforms input data according to the trainer model, the model should
        have been trained before
        :param iterator:
        :return: dictionary keyed on segments with vectorized data
        """
        self._prepare_data(iterator)
        segments = {}
        logging.info('Vectorization & Tansformation Starting')
        for segment in self._vect_data:
            logging.info('Processing Segment: %s', segment)
            segments[segment] = {
                'Y': self._get_target_variable_labels(segment),
                'X': scipy.sparse.hstack(self._get_vectorized_data(
                    segment, self._test_prepare_feature))
            }
        return segments

    def clear_temp_data(self):
        if hasattr(self, '_raw_data'):
            self._raw_data = None
        if hasattr(self, '_vect_data'):
            self._vect_data = None

    def store_vect_data(self, data, file_name):
        numpy.savez_compressed(file_name, *data)

    def vect_data2csv(self, file_name):
        if not self.intermediate_data[self.TRAIN_VECT_DATA]:
            raise ValueError("Execute train with store_vect_data parameter")

        few_segments = len(self.intermediate_data[self.TRAIN_VECT_DATA]) > 1
        for segment, data in self.intermediate_data[self.TRAIN_VECT_DATA].iteritems():
            if few_segments:
                segment_file_name = "{1}-{0}".format(segment, file_name)
            else:
                segment_file_name = file_name

            logging.info("Storing vectorized data of segment %s to %s",
                         segment, segment_file_name)
            with open(segment_file_name, 'wb') as csvfile:
                writer = csv.writer(csvfile)
                true_data = data['data']
                num = true_data.shape[0]
                matrix = true_data.tocsr()
                for i in xrange(num):
                    writer.writerow([data['labels'][i]]
                                    + matrix.getrow(i).todense().tolist()[0])

    def store_feature_weights(self, fp):
        """
        Stores the weight of each feature to the given file.

        Keyword arguments:
        fp -- the file to store the trained model.
        """
        json.dump(self.get_weights(), fp, indent=4)

    def _get_labels(self):
        if self.with_segmentation:
            classes_ = {}
            for segment, count in self._segments.iteritems():
                # Note: Possible problems when value of the group_by field
                # equals `DEFAULT_SEGMENT`
                # if segment != DEFAULT_SEGMENT:
                classifier = self._classifier[segment]
                if hasattr(classifier, 'classes_'):
                    classes_[segment] = map(str, classifier.classes_.tolist())

            assert all(map(
                lambda x: x == classes_.values()[0], classes_.values())), \
                'The assumption is that all segments should have the same ' \
                'classifier.classes_'
            return classes_.values()[0]
        else:
            return map(str, self._classifier[DEFAULT_SEGMENT].classes_.tolist())

    # utility methods

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
        self._raw_data = defaultdict(list)
        self._order_data = []
        self._vect_data = {}
        segments = {}
        for row in iterator:
            self._count += 1
            try:
                data = self._apply_feature_types(row)
                if self.with_segmentation:
                    segment = self._get_segment_name(data)
                else:
                    segment = DEFAULT_SEGMENT

                if segment not in self._vect_data:
                    self._vect_data[segment] = defaultdict(list)
                    segments[segment] = 0
                segments[segment] += 1
                self._order_data.append(segment)

                for feature_name in self._feature_model.features:
                    # if feature_name in self._feature_model.group_by:
                    #     continue
                    self._vect_data[segment][feature_name].append(data[feature_name])

                if save_raw:
                    self._raw_data[segment].append(row)

                if callback is not None:
                    callback(row)
            except ItemParseException, e:
                logging.debug('Ignoring item #%d: %s'
                              % (self._count, e))
                if ignore_error:
                    self._ignored += 1
                else:
                    raise e
        return segments

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
                        result[feature_name] = map(ft.transform, item)
                except Exception as e:
                    logging.warn('Error processing feature %s: %s'
                                 % (feature_name, e))
                    raise ItemParseException('Error processing feature %s: %s'
                                             % (feature_name, e))
            else:
                result[feature_name] = item

        return result

    def _get_segments_info(self):
         return self._segments

    def _get_target_variable_labels(self, segment):
        """
        using `_feature_model.target_variable` retrieves the features
        labels/values from `_vect_data` for the given segment
        :param segment:
        :return: list of string values of `_feature_model.target_variable` in
         `_vect_data`
        """
        if self._vect_data is None or self._vect_data == {}:
            raise Exception('trainer._vect_data was not prepared')
        return self._vect_data[segment][self.feature_model.target_variable]

    def _get_vectorized_data(self, segment, fn_prepare_feature):
        """
        applies transforms to features values in `_vect_data`
        :param segment:
        :param fn_prepare_feature: function responsible for preparing feature
        :return: sparse matrix of tranformed data
        """
        if self._vect_data is None or self._vect_data == {}:
            raise Exception('trainer._vect_data was not prepared')

        vectorized_data = []
        for feature_name, feature in self.features[segment].iteritems():
            if feature_name not in self.feature_model.group_by and \
                    not feature_name == self.feature_model.target_variable:
                item = fn_prepare_feature(feature,
                                          self._vect_data[segment][
                                              feature_name])
                if item is not None:
                    # Convert item to csc_matrix, since hstack fails with arrays
                    vectorized_data.append(scipy.sparse.csc_matrix(item))
        return vectorized_data

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
            elif feature.get('transformer-type') is not None and \
                    feature.get('transformer') is not None:
                result = TRANSFORMERS[feature['transformer-type']]['default']
            elif feature.get('type') is not None:
                result = FEATURE_TYPE_DEFAULTS.get(feature['type'], value)

        return result

    def _evaluate_segment(self, segment):
        # Get X and y
        logging.info('Extracting features for segment %s ...', segment)
        labels = self._get_target_variable_labels(segment)
        classes = self._get_classifier_adjusted_classes(segment)
        vectorized_data = self._get_vectorized_data(
            segment, self._test_prepare_feature)
        logging.info("Memory usage (vectorized data generated): %f" %
                     memory_usage(-1, interval=0, timeout=None)[0])
        logging.info('Evaluating model...')

        self.metrics.evaluate_model(labels, classes, vectorized_data,
                                    self._classifier[segment],
                                    self._test_empty_labels[segment], segment)
        
        self.generate_trained_model_visualization(segment, self.metrics._true_data[segment])
        logging.info("Memory usage: %f" %
                     memory_usage(-1, interval=0, timeout=None)[0])

    def _get_segment_name(self, row_data):
        return "_".join(
            [str(row_data[feature_name]) for feature_name in
             self._feature_model.group_by])

    def _train_prepare_feature(self, feature, data):
        """
        Uses the appropriate vectorizer or scaler on a specific feature and its
        training data. Used for unfitted feature in untrained model.

        :param feature: the name of the feature to prepare. Used to retrieve the
            appropriate vectorizer.
        :param data: a list of the data for extracted for the given feature.
        :return: feature data with transformation applied
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
                if feature['transformer-type'] in ('Lda', 'Lsi'):
                    feature['transformer'].num_features = transformed_data.shape[1]
            except ValueError as e:
                logging.warn('Feature %s will be ignored due to '
                             'transformation error: %s.' %
                             (feature['name'], str(e)))
                transformed_data = None
                feature['tranformer'] = SuppressTransformer()
            return transformed_data
        elif feature['transformer'] is None and feature['transformer-type'] is not None:
            feature['transformer'] = self.get_transformer(feature['transformer-type'])
            transformed_data = feature['transformer'].transform(data)
            feature['transformer'].num_features = transformed_data.shape[1]
            return transformed_data
        elif feature.get('scaler', None) is not None:
            return feature['scaler'].fit_transform(self._to_column(data).toarray())
        else:
            return self._to_column(data)

    def _test_prepare_feature(self, feature, data):
        """
        Uses the appropriate vectorizer or scaler on a specific feature and its
        training data. Used for fitted feature in a trained model.

        :param feature: the name of the feature to prepare. Used to retrieve the
            appropriate vectorizer.
        :param data: a list of the data for extracted for the given feature.
        :return: feature data with transformation applied
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
            scipy.sparse.csc_matrix([0.0 if item is None else float(item) for item in x]))

    def _check_data_for_test(self):
        """
        Checks input data (examples) in a trained model for testing. The checks
        are for potential incomplete data/examples that might prevent successful
        testing.
        Raises exception if checks fail.
        :return: dictionary keyed on segments, every key is a list labels with
        no corresponding examples for that label in the keyed segment
        """
        empty_labels = {}
        for segment in self._segments:
            empty_labels[segment] = []
            labels = self._get_classifier_adjusted_classes(segment)
            target_feature = self._get_target_feature(segment)['name']
            examples_per_label = dict((c, 0) for c in labels if c is not None)
            if segment in self._vect_data:
                for label in self._vect_data[segment][target_feature]:
                    if label is not None:
                        examples_per_label[label] += 1
            for label, _ in filter(lambda (label, c): c == 0,
                                   examples_per_label.iteritems()):
                msg = 'In Segment: %s, Class: %s, has no examples. ' \
                      'Test evaluation will fail' % \
                      (segment, label)
                logging.warn(msg)
                empty_labels[segment].append(label)
        return empty_labels

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
                        result[feature_name] = map(ft.transform, item)
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
            elif feature.get('transformer-type') is not None and \
                    feature.get('transformer') is not None:
                result = TRANSFORMERS[feature['transformer-type']]['default']
            elif feature.get('type') is not None:
                result = FEATURE_TYPE_DEFAULTS.get(feature['type'], value)

        return result

    def _get_target_variable_labels(self, segment):
        """
        using `_feature_model.target_variable` retrieves the features
        labels/values from `_vect_data` for the given segment
        :param segment:
        :return: list of string values of `_feature_model.target_variable` in
         `_vect_data`
        """
        if self._vect_data is None or self._vect_data == {}:
            raise Exception('trainer._vect_data was not prepared')
        return self._vect_data[segment][self._feature_model.target_variable]

    def _get_classifier_adjusted_classes(self, segment):
        """
        :param segment:
        :return: adjusted underlying classifier classes/values to match
         the target feature/variable real type
        """
        target_feature = self._get_target_feature(segment)
        return [_adjust_classifier_class(target_feature, c)
                for c in self._classifier[segment].classes_.tolist()]

    def _get_target_feature(self, segment):
        """
        :param segment:
        :return: The corresponding feature object to the model's target variable
        """
        return self.features[segment][self._feature_model.target_variable]

    def _get_vectorized_data(self, segment, fn_prepare_feature):
        """
        applies transforms to features values in `_vect_data`
        :param segment:
        :param fn_prepare_feature: function responsible for preparing feature
        :return: sparse matrix of tranformed data
        """
        if self._vect_data is None or self._vect_data == {}:
            raise Exception('trainer._vect_data was not prepared')

        vectorized_data = []
        for feature_name, feature in self.features[segment].iteritems():
            if feature_name not in self._feature_model.group_by and \
                    not feature_name == self._feature_model.target_variable:
                item = fn_prepare_feature(feature,
                                          self._vect_data[segment][
                                              feature_name])
                if item is not None:
                    # Convert item to csc_matrix, since hstack fails with arrays
                    vectorized_data.append(scipy.sparse.csc_matrix(item))
        return vectorized_data

    def get_nonzero_vectorized_data(self):
        vectorized_data = {}
        res = {}
        for segment in self._vect_data:
            for feature_name, feature in self.features[segment].iteritems():
                if feature_name not in self._feature_model.group_by and \
                    not feature_name == self._feature_model.target_variable:

                    item = self._test_prepare_feature(feature,
                                              self._vect_data[segment][
                                                  feature_name])
                    transformer = feature['transformer']
                    preprocessor = feature['type'].preprocessor
                    if item is not None:
                        if isinstance(item, numpy.ndarray):
                            value = item.tolist()[0][0]
                            if value:
                                vectorized_data[feature_name] = item.tolist()[0][0]
                        else:
                            vectorized_data[feature_name] = {}
                            if transformer is not None and hasattr(transformer, 'num_topics'):
                                item = item.todense().tolist()
                                for j in range(0, transformer.num_features):
                                    subfeature = '%s->Topic #%d' % (feature_name.replace(".", "->"), j)
                                    if item[0][j] != 0:
                                        vectorized_data[feature_name][subfeature] = item[0][j]
                            elif transformer is not None and hasattr(transformer,
                                                           'get_feature_names'):
                                index = 0
                                item = item.todense().tolist()
                                for subfeature in transformer.get_feature_names():
                                    if item[0][index]:
                                        vectorized_data[feature_name][subfeature] = item[0][index]
                                    index +=1
                            elif preprocessor is not None and hasattr(preprocessor,
                                                          'get_feature_names'):
                                index = 0
                                item = item.todense().tolist()
                                for subfeature in preprocessor.get_feature_names():
                                    if item[0][index]:
                                        vectorized_data[feature_name][subfeature] = item[0][index]
                                    index +=1
                            if not vectorized_data[feature_name].items():
                                vectorized_data.pop(feature_name)
            res[segment] = vectorized_data
        return res


def list_to_dict(user_params):
    if user_params is not None:
        param_list = [x.split('=', 1) for x in user_params]
        return dict((key, value) for (key, value) in param_list)

    return dict()


# TODO: we need something better than that. We should also consider
# other features types that can cause similar problems
def _adjust_classifier_class(feature, str_value):
    """
    The classifier treats every class as string, while the data labels are
    getting converted by features transforms. So a mapping
    {"Class1": 1, "Class2": 2} gets labels of say [1, 2, 2, 1, 1]
    while the classes in the classifier is ['1', '2']

    :param feature: the feature responsible for the transform
    :param value: value of the feature at the data point as stored by the classifier
    :return: The feature value at the data point with correct data type
    """
    from core.trainer.feature_types.ordinal import OrdinalFeatureTypeInstance
    from core.trainer.feature_types.primitive_types import \
        PrimitiveFeatureTypeInstance
    assert isinstance(str_value, str) or isinstance(str_value, unicode), \
        'str_value should be string it is of type %s' % (type(str_value))

    if isinstance(feature['type'], OrdinalFeatureTypeInstance):
        try:
            value = int(str_value)
        except ValueError:
            value = 0
        return value
    elif isinstance(feature['type'], PrimitiveFeatureTypeInstance) and \
            feature['type'].python_type is bool:
        return str_value.lower() in ['true', '1']
    else:
        return feature['type'].transform(str_value)
