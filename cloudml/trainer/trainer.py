"""
This module gathers cloudml classes and methods for training
and evaluating the model.
"""

# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Panagiotis Papadimitriou <papadimitriou@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>
#          Nader Soliman <nsoliman@cloud.upwork.com>

import json
import logging
import numpy
import csv
import scipy.sparse
import platform
from copy import deepcopy
from collections import defaultdict
from time import gmtime, strftime
from memory_profiler import memory_usage
from sklearn.preprocessing import Imputer

from feature_types import FEATURE_TYPE_DEFAULTS
from classifier_settings import TYPE_CLASSIFICATION
from transformers import TRANSFORMERS, SuppressTransformer
from model_visualization import TrainedModelVisualizer
from exceptions import ItemParseException, ItemParseIgnore, \
    EmptyDataException, TransformerNotFound
from utils import is_empty

DEFAULT_SEGMENT = 'default'


class Trainer(object):
    """
    Model trainer, evaluator and predictor.
    """
    TRAIN_VECT_DATA = 'train_vect_data'

    def __init__(self, feature_model):
        """
        Initializes the trainer using the application's configuration. Creates
        classifier, instantiates appropriate vectorizers and scalers for all
        features and prints initialization messages.

        feature_model: cloudml.trainer.config.FeatureModel
             the feature model object
        """
        self._feature_model = feature_model
        self._initialize_classifier(feature_model)

        self.features = {}
        self._count = 0
        self._ignored = 0
        self.train_time = {}
        self._segments = {}
        self.visualization = {}
        self._strict_mode = False
        self.intermediate_data = defaultdict(dict)

        self.model_visualizer = TrainedModelVisualizer.factory(self)

    @property
    def classifier_type(self):
        return self._classifier_type

    def set_strict_mode(self, value=True):
        self._strict_mode = value

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
        from classifier_settings import get_model_type
        return get_model_type(self._classifier_type)

    def get_classifier(self, segment=DEFAULT_SEGMENT):
        return self._classifier[segment]

    def get_features(self, segment=DEFAULT_SEGMENT):
        return self.features.get(segment)

    def is_target_variable(self, feature_name):
        return feature_name == self.feature_model.target_variable

    def is_group_by_variable(self, feature_name):
        return feature_name in self.feature_model.group_by

    def get_transformer(self, name):
        raise TransformerNotFound(
            "Transformer with name {} not found".format(name))

    def set_transformer_getter(self, method):
        """
        Sets pretrained transformer getter for the model.
        """
        self.get_transformer = method

    def set_classifier(self, classifier):
        self._classifier = classifier

    def set_features(self, features):
        self.features = features

    def _initialize_classifier(self, feature_model):
        # Classifier to use
        self._classifier_type = feature_model.classifier_type
        logging.info('Using "%s"', self._classifier_type)

        classifier_cls = feature_model.classifier_cls
        self._classifier = {}
        self._classifier[DEFAULT_SEGMENT] = classifier_cls(
            **feature_model.classifier)

    # base methods

    # train model related

    def train(self, iterator, percent=0, store_vect_data=False):
        """
        Train the model using the given data. Appropriate SciPy vectorizers
        will be used to bring data to the appropriate format.

        iterator: iterator
            an iterator that provides a dictionary with keys feature
            names and value the values of the features per column.
        percent: integer from 0 to 100
        store_vect_data: boolean
            determines whether we need to store vectorized data token
            'intermediate_data' field.
        """
        log_memory_usage("Memory usage")

        if iterator:
            self._segments = self._prepare_data(iterator)
        else:
            raise EmptyDataException("No rows found in the iterator")

        if self.with_segmentation:
            logging.info('Group by: %s',
                         ",".join(self.feature_model.group_by))
            logging.info('Segments:')
            for segment, records in self._segments.iteritems():
                logging.info("'%s' - %d records", segment, records)

        if percent:
            if percent < 0 or percent > 100:
                raise ValueError("percent parameter should be from 0 to 100")

            self._count = self._count - int(self._count * percent / 100)
            for segment in self._vect_data:
                for item in self._vect_data[segment]:
                    item = item[:self._count]
        logging.info('Processed %d lines, ignored %s lines',
                     self._count, self._ignored)
        log_memory_usage("Memory usage")
        for segment in self._vect_data:
            logging.debug('Starting train "%s" segment', segment)
            self._train_segment(segment, store_vect_data)

    def _train_segment(self, segment, store_vect_data=False):
        # Get X and y
        logging.info('Extracting features for segment %s ...', segment)

        self.features[segment] = deepcopy(self._feature_model.features)
        labels = self._get_target_variable_labels(segment)
        vectorized_data = self._get_vectorized_data(
            segment, self._train_prepare_feature)
        log_memory_usage("Memory usage (vectorized data generated)")
        logging.info('Training model...')
        true_data = scipy.sparse.hstack(vectorized_data)
        log_memory_usage("Memory usage (true data generated)")
        vectorized_data = None
        log_memory_usage("Memory usage (vectorized data cleared)")

        logging.info('Number of features: %s', true_data.shape[1])
        if segment != DEFAULT_SEGMENT:
            self._classifier[segment] = \
                deepcopy(self._classifier[DEFAULT_SEGMENT])
        self._classifier[segment].fit(true_data, [str(l) for l in labels])
        log_memory_usage("Memory usage (model fitted with true data)")

        self.generate_trained_model_visualization(segment, true_data)

        if store_vect_data:
            self.intermediate_data[self.TRAIN_VECT_DATA][segment] = {
                'data': true_data,
                'labels': labels}

        true_data = None
        log_memory_usage("Memory usage (true data cleared)")
        self.train_time[segment] = strftime('%Y-%m-%d %H:%M:%S %z', gmtime())
        logging.info('Training completed...')

    def generate_trained_model_visualization(self, segment, true_data):
        log_memory_usage("Memory usage")
        logging.info("Generate trained model visualization")
        self.model_visualizer.generate(segment, true_data)
        self.get_visualization(segment)
        log_memory_usage("Memory usage (after gen model visualization)")

    def get_visualization(self, segment, **kwargs):
        if not self.visualization or segment not in self.visualization:
            self.visualization[segment] = \
                self.model_visualizer.get_visualization(segment, **kwargs)
        return self.visualization[segment]

    def get_weights(self, segment=DEFAULT_SEGMENT):
        if not self.visualization or segment not in self.visualization:
            self.visualization[segment] = \
                self.model_visualizer.get_visualization(segment)
        return self.visualization[segment]['weights']

    # evaluating model related

    def test(self, iterator, percent=0, callback=None, save_raw=True):
        """
        Test the model using the given data. SciPy vectorizers that were
        populated with data during testing will be used.

        iterator: iter
            an iterator that provides a dictionary with keys feature
            names and value the values of the features per column.
        percent: integer from 0 to 100
        callback: function
            callback function to invoke on every row coming from the
            iterator and is not ignored.
        save_raw: boolean
            determines whether we need to store raw data to '_save_raw'
            field.
        """
        from metrics import Metrics
        self.metrics = Metrics.factory(self.model_type)

        self._prepare_data(iterator, callback, save_raw=save_raw)
        if self.model_type == TYPE_CLASSIFICATION:
            self._test_empty_labels = self._check_data_for_test()
        else:
            self._test_empty_labels = []

        if percent:
            self._count = int(self._count * percent / 100)
            for segment in self._vect_data:
                for item in self._vect_data[segment]:
                    item = item[:self._count]
        logging.info('Processed %d lines, ignored %s lines',
                     self._count, self._ignored)

        for segment in self._vect_data:
            logging.info('Starting test "%s" segment', segment)
            self._evaluate_segment(segment)
        self.metrics.log_metrics()

        return self.metrics

    def predict(self, iterator, callback=None, ignore_error=True,
                store_vect_data=False, callback_store_data=None):
        """
        Attempts to predict the class of each of the data in the given
        iterator. Returns the predicted values, the target feature value (if
        present).

        iterator: iter
            an iterator that provides a dictionary with keys feature names
            and value the values of the features per column.
        callback: function
            callback function to invoke on every row coming from the iterator
            and is not ignored.
        ignore_error: boolean
            determines whether we need increase ignored counter instead of
            raising the exception.
        store_vect_data: bollean
        """
        labels = {}
        probs = {}
        true_labels = {}
        indexes = defaultdict(int)
        ordered_probs = []
        ordered_labels = []
        ordered_true_labels = []
        self.predict_data = {}

        self._prepare_data(iterator, callback, ignore_error, is_predict=True)
        logging.info('Processed %d lines, ignored %s lines',
                     self._count, self._ignored)
        if self._ignored == self._count:
            logging.info("Don't have valid records")
            return {'error': 'all records was ignored'}
        else:
            for segment in self._vect_data:
                vectorized_data = []
                # Get X and y
                logging.info('Extracting features...')
                true_labels[segment] = \
                    self._get_target_variable_labels(segment)
                vectorized_data = self._get_vectorized_data(
                    segment, self._test_prepare_feature,
                    callback_store_data)
                logging.info('Evaluating model...')
                if len(vectorized_data) == 1:
                    predict_data = numpy.array(vectorized_data[0])
                else:
                    predict_data = scipy.sparse.hstack(vectorized_data)
                if store_vect_data:
                    self.predict_data[segment] = predict_data
                probs[segment] = \
                    self._classifier[segment].predict_proba(predict_data)
                labels[segment] = \
                    self._classifier[segment].classes_[probs[segment].
                                                       argmax(axis=1)]
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

    def grid_search(self, parameters, train_iterator, test_iterator,
                    score=None):
        """
        Searches over specified parameter values for an estimator.

        parameters: dict or list of dictionaries
            Dictionary with parameters names (string) as keys and lists of
            parameter settings to try as values, or a list of such
            dictionaries, in which case the grids spanned by each dictionary
            in the list are explored. This enables searching over any sequence
            of parameter settings.
        train_iterator: iter
            an iterator that provides a dictionary with keys feature names
            and value the values of the features per column that would be
            used for training the model.
        test_iterator: iter
            an iterator that provides a dictionary with keys feature names
            and value the values of the features per column that would be
            used for evaluating the model.
        score: string, callable or None, optional, default: None
            A string (see model evaluation documentation) or
            a scorer callable object / function with signature
            ``scorer(estimator, X, y)``.
        """
        from sklearn import grid_search
        classifier = self._classifier[DEFAULT_SEGMENT]
        clf = grid_search.GridSearchCV(classifier, parameters, scoring=score)
        results = {}
        if train_iterator:
            self._segments = self._prepare_data(train_iterator)
        for segment in self._vect_data:
            logging.info('Starting search params for "%s" segment', segment)
            self.features[segment] = deepcopy(self._feature_model.features)
            labels = self._get_target_variable_labels(segment)
            vectorized_data = \
                self._get_vectorized_data(segment,
                                          self._train_prepare_feature)
            true_data = scipy.sparse.hstack(vectorized_data)
            clf.fit(true_data, [str(l) for l in labels])
            results[segment] = clf
        return results

    def transform(self, iterator):
        """
        Transforms input data according to the trainer model, the model should
        have been trained before. Returns a dictionary keyed on segments with
        vectorized data.

        iterator: iter
            an iterator that provides a dictionary with keys feature names
            and value the values of the features per column
        """
        self._prepare_data(iterator)
        segments = {}
        logging.info('Vectorization & Transformation Starting')
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
        for segment, data in \
                self.intermediate_data[self.TRAIN_VECT_DATA].iteritems():
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
                    writer.writerow([data['labels'][i]] +
                                    matrix.getrow(i).todense().tolist()[0])

    def store_feature_weights(self, fp):
        """
        Stores the weight of each feature to the given file.

        Keyword arguments:
        fp -- the file to store the trained model.
        """
        # FIXME: JSON always dumps all keys as string!
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
            return map(str,
                       self._classifier[DEFAULT_SEGMENT].classes_.tolist())

    # utility methods

    def _prepare_data(self, iterator, callback=None,
                      ignore_error=True, save_raw=False, is_predict=False):
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
                data = self._apply_feature_types(row, is_predict)
                if self.with_segmentation:
                    segment = self._get_segment_name(data)

                    if segment in (None, numpy.nan, 'nan'):
                        logging.warning(
                            "Group by value is Null,"
                            " records will be ignored")
                        raise ItemParseException("Group by value is Null")
                else:
                    segment = DEFAULT_SEGMENT

                if segment not in self._vect_data:
                    self._vect_data[segment] = defaultdict(list)
                    segments[segment] = 0

                from feature_types.categorical import \
                    CategoricalFeatureTypeInstance
                for feature_name in self._feature_model.features:
                    if isinstance(self._feature_model.features[feature_name]['type'], CategoricalFeatureTypeInstance) and self._feature_model.features[feature_name]['type']._params:
                        categories = self._feature_model.features[feature_name]['type']._params.get('categories', None)
                        exclude = self._feature_model.features[feature_name]['type']._params.get('exclude', False)
                        if exclude and categories and \
                                data[feature_name] in categories:
                            raise ItemParseIgnore(
                                "Item will be exclude: '%s' feature value is "
                                "'%s', but should not be in %s" %
                                (feature_name, data[feature_name], categories))
                        elif not exclude and categories and \
                                data[feature_name] not in categories:
                            raise ItemParseIgnore(
                                "Item not in categories: '%s' feature value is"
                                " '%s', but should be in %s" %
                                (feature_name, data[feature_name], categories))

                for feature_name in self._feature_model.features:
                    # if feature_name in self._feature_model.group_by:
                    #     continue
                    self._vect_data[segment][feature_name].append(
                        data[feature_name])
                segments[segment] += 1
                self._order_data.append(segment)
                if save_raw:
                    self._raw_data[segment].append(row)

                if callback is not None:
                    callback(row)
            except ItemParseIgnore, e:
                # logging.debug('Ignoring item #%d: %s', self._count, e)
                if ignore_error:
                    self._ignored += 1
                else:
                    raise e
            except ItemParseException, e:
                logging.debug('Ignoring item #%d: %s', self._count, e)
                if ignore_error:
                    self._ignored += 1
                else:
                    raise e
        return segments

    def _get_segments_info(self):
        return self._segments

    def _evaluate_segment(self, segment):
        # Get X and y
        logging.info('Extracting features for segment %s ...', segment)
        labels = self._get_target_variable_labels(segment)
        vectorized_data = self._get_vectorized_data(
            segment, self._test_prepare_feature)
        log_memory_usage("Memory usage (vectorized data generated)")
        logging.info('Evaluating model...')

        if self.model_type == TYPE_CLASSIFICATION:
            classes = self._get_classifier_adjusted_classes(segment)
            self.metrics.evaluate_model(labels, classes, vectorized_data,
                                        self._classifier[segment],
                                        self._test_empty_labels[segment],
                                        segment)
        else:
            # TODO: _test_empty_labels
            self.metrics.evaluate_model(labels, vectorized_data,
                                        self._classifier[segment],
                                        [], segment)

        log_memory_usage("Memory usage")

    def _get_segment_name(self, row_data):
        return "_".join(
            [str(row_data[feature_name]) for feature_name in
             self._feature_model.group_by])

    def _train_prepare_feature(self, feature, data):
        """
        Uses the appropriate vectorizer or scaler on a specific feature and its
        training data. Used for unfitted feature in untrained model.

        :param feature: the name of the feature to prepare. Used to retrieve
        the appropriate vectorizer.
        :param data: a list of the data for extracted for the given feature.
        :return: feature data with transformation applied
        """
        logging.info('Preparing feature %s for train', feature['name'])
        input_format = feature.get('input-format', None)
        if input_format == 'list':
            data = map(lambda x: " ".join(x) if isinstance(x, list)
                       else x, data)
        if feature['type'].preprocessor:
            return feature['type'].preprocessor.fit_transform(data)

        if feature['transformer'] is not None:
            try:
                transformed_data = feature['transformer'].fit_transform(data)
                if feature['transformer-type'] in ('Lda', 'Lsi', 'Word2Vec',
                                                   'Doc2Vec'):
                    feature['transformer'].num_features = \
                        transformed_data.shape[1]
            except ValueError as e:
                logging.warn('Feature %s will be ignored due to '
                             'transformation error: %s.',
                             feature['name'], e)
                transformed_data = None
                feature['transformer'].num_features = 0
            return transformed_data
        elif feature['transformer'] is None and \
                feature['transformer-type'] is not None:
            feature['transformer'] = self.get_transformer(
                feature['transformer-type'])
            transformed_data = feature['transformer'].transform(data)
            feature['transformer'].num_features = transformed_data.shape[1]
            return transformed_data
        else:
            feature['imputer'] = Imputer(missing_values='NaN',
                                         strategy='median', axis=0)
            count = len(data)
            data = feature['imputer'].fit_transform(
                self._to_column(data).toarray())
            if data.shape[1] < 1:
                data = [feature['type'].default] * count
                data = self._to_column(data).toarray()
                logging.warning("All values of feature %s are null"
                                % (feature['name']))

        if feature.get('scaler', None) is not None:
            return feature['scaler'].fit_transform(data)
        else:
            return data

    def _test_prepare_feature(self, feature, data):
        """
        Uses the appropriate vectorizer or scaler on a specific feature and its
        training data. Used for fitted feature in a trained model.

        :param feature: the name of the feature to prepare. Used to retrieve
        the appropriate vectorizer.
        :param data: a list of the data for extracted for the given feature.
        :return: feature data with transformation applied
        """
        logging.debug('Preparing feature %s for test', feature['name'])
        input_format = feature.get('input-format', None)
        if input_format == 'list':
            data = map(lambda x: " ".join(x)
                       if isinstance(x, list) else x, data)

        if feature['type'].preprocessor:
            return feature['type'].preprocessor.transform(data)
        if feature['transformer'] is not None:
            if isinstance(feature['transformer'], SuppressTransformer):
                return None
            else:
                return feature['transformer'].transform(data)
        else:
            data = self._to_column(data).toarray()
            count = len(data)
            if 'imputer' in feature and \
                    feature['imputer'] is not None:
                data = feature['imputer'].transform(data)
            else:
                feature['imputer'] = Imputer(
                    missing_values='NaN',
                    strategy='median', axis=0)
                data = feature['imputer'].fit_transform(data)
            if data.shape[1] < 1:
                from feature_types.primitive_types import PROCESSOR_MAP
                default = 0
                if hasattr(feature['type'], 'python_type'):
                    default = PROCESSOR_MAP[feature['type'].python_type]()
                data = [default] * count
                data = self._to_column(data).toarray()
                logging.warning(
                    "All values of feature %s are null", feature['name'])

        if feature.get('scaler', None) is not None:
            return feature['scaler'].transform(data)
        else:
            return data

    def _to_column(self, x):
        return numpy.transpose(
            scipy.sparse.csc_matrix(
                [0.0 if item is None else float(item) for item in x]))

    def _check_data_for_test(self):
        """
        Checks input data (examples) in a trained model for testing. The
        checks are for potential incomplete data/examples that might
        prevent successful testing.
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
            for l in examples_per_label.keys():
                logging.error("%s=%s" % (l, type(l)))
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

    def _apply_feature_types(self, row_data, is_predict=False):
        """
        Apply the transformation dictated by feature type instance (if any).

        Keyword arguments:
        row_data -- current row's data to be processed.

        """
        result = {}
        for feature_name, feature in self._feature_model.features.iteritems():
            ft = feature.get('type', None)
            item = row_data.get(feature_name, None)
            if is_empty(item) and \
                self._feature_model.target_variable == feature_name and \
                    not is_predict:
                raise ItemParseException('Target feature is null')
            if feature.get('required', True) and not (self._feature_model.target_variable == feature_name and \
                    is_predict):
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
                    raise ItemParseException(
                        'Error processing feature %s: %s' %
                        (feature_name, e))
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
        empty = is_empty(value)
        if empty and self._strict_mode:
            raise ItemParseException('Feature %s is emty' % feature['name'])

        if empty:
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
        :return: The corresponding feature object to the model's target
        variable
        """
        return self.features[segment][self._feature_model.target_variable]

    def _get_vectorized_data(self, segment, fn_prepare_feature,
                             callback_store_data=None):
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
                    # Convert item to csc_matrix, since hstack fails
                    # with arrays
                    if callback_store_data is not None:
                        callback_store_data(segment, feature_name, item)
                    vectorized_data.append(scipy.sparse.csc_matrix(item))
        return vectorized_data

    def get_nonzero_vectorized_data(self):
        vectorized_data = {}
        res = {}
        for segment in self._vect_data:
            for feature_name, feature in self.features[segment].iteritems():
                if feature_name not in self._feature_model.group_by and \
                        feature_name != self._feature_model.target_variable:

                    item = self._test_prepare_feature(
                        feature, self._vect_data[segment][feature_name])
                    transformer = feature['transformer']
                    preprocessor = feature['type'].preprocessor
                    if item is not None:
                        if isinstance(item, numpy.ndarray):
                            value = item.tolist()[0][0]
                            if value:
                                vectorized_data[feature_name] = \
                                    item.tolist()[0][0]
                        else:
                            vectorized_data[feature_name] = {}
                            if transformer is not None and hasattr(
                                    transformer, 'num_topics'):
                                item = item.todense().tolist()
                                for j in range(0, transformer.num_features):
                                    subfeature = '%s->Topic #%d' % (
                                        feature_name.replace(".", "->"), j)
                                    if item[0][j] != 0:
                                        vectorized_data[feature_name][subfeature] = item[0][j]
                            elif transformer is not None and hasattr(
                                    transformer, 'get_feature_names'):
                                index = 0
                                item = item.todense().tolist()
                                for subfeature in \
                                        transformer.get_feature_names():
                                    if item[0][index]:
                                        vectorized_data[feature_name][subfeature] = item[0][index]
                                    index += 1
                            elif preprocessor is not None and hasattr(
                                    preprocessor, 'get_feature_names'):
                                index = 0
                                item = item.todense().tolist()
                                for subfeature in \
                                        preprocessor.get_feature_names():
                                    if item[0][index]:
                                        vectorized_data[feature_name][subfeature] = \
                                            item[0][index]
                                    index += 1
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

    feature: dict
        the feature responsible for the transform
    value: string
        value of the feature at the data point as stored by the classifier

    Returns the feature value at the data point with correct data type
    """
    from cloudml.trainer.feature_types.ordinal import \
        OrdinalFeatureTypeInstance
    assert isinstance(str_value, basestring), \
        'str_value should be string it is of type %s' % (type(str_value))

    if isinstance(feature['type'], OrdinalFeatureTypeInstance):
        try:
            value = int(str_value)
        except ValueError:
            value = 0
        return value
    return feature['type'].transform(str_value)


def log_memory_usage(msg):
    if platform.system() == 'Windows':
        return
    logging.debug(
        "%s: %f" % (msg, memory_usage(-1, interval=0, timeout=None)[0]))
