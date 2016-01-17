"""
This module holds classes and utils to calculate evaluation model
metrics.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import logging
from collections import OrderedDict

import numpy
from scipy.sparse import hstack, csc_matrix
import sklearn.metrics as sk_metrics

from classifier_settings import TYPE_CLASSIFICATION, TYPE_REGRESSION


__all__ = ['Metrics']


class BaseMetrics(object):
    METRICS_TO_CALC = {}
    DEFAULTS = {}

    def __init__(self):
        self._labels = []
        self._vectorized_data = []
        self._classifier = {}
        self._preds = None
        self._probs = None
        self._true_data = OrderedDict()
        self._classes_set = []
        self._empty_labels = []

    def evaluate_model(self, labels, classes, vectorized_data, classifier,
                       empty_labels, segment='default'):
        self._labels += labels
        self._empty_labels = empty_labels
        logging.info('Classes set: %s' % classes)
        if self._classes_set and not classes == self._classes_set:
            raise ValueError(
                'Classes was set before to %s, now it is being set with %s, '
                'which should be equal', self._classes_set, classes)
        self._classes_set = classes
        self._classifier[segment] = classifier

        # Evaluating model...
        if(len(vectorized_data) == 1):
            if isinstance(vectorized_data[0], csc_matrix):
                true_data = vectorized_data[0]
            else:
                true_data = numpy.array(vectorized_data[0])
        else:
            try:
                true_data = hstack(vectorized_data)
            except ValueError:
                true_data = numpy.hstack(vectorized_data)
        try:
            self._true_data[segment] = true_data.tocsr()
        except:
            self._true_data[segment] = true_data
        probs = classifier.predict_proba(true_data)
        preds = classifier.predict(true_data)

        if self._preds is None:
            self._preds = preds
        else:
            self._preds = numpy.append(self._preds, preds)

        if self._probs is None:
            self._probs = probs
        else:
            self._probs = numpy.vstack((self._probs, probs))

    @property
    def classes_set(self):
        """
        Returns classes as recognized by underline classifer
        """
        return self._classes_set

    @property
    def classes_count(self):
        return len(self._classes_set)

    def get_metrics_dict(self):
        """
        Returns dict with metrics values.

        Note: Now it used in the REST API.
        """
        def recursive_convert(val):
            value = val
            if isinstance(val, list) or isinstance(val, tuple):
                value = [recursive_convert(item) for item in val]
            elif isinstance(val, numpy.ndarray):
                value = [recursive_convert(item) for item in val.tolist()]
            elif isinstance(val, dict):
                value = dict([(x, recursive_convert(y))
                              for x, y in val.iteritems()])

            return value

        # TODO: Think about moving logic to serializer
        # and make Metrics class serializable
        res = {}
        res.update(self.DEFAULTS)
        metrics = self._get_metrics_names()
        for metric_name in metrics.keys():
            res[metric_name] = recursive_convert(getattr(self, metric_name))
        return res

    # TODO: is it used?
    def to_serializable_dict(self):
        return self.get_metrics_dict()

    def log_metrics(self):
        """
        Outputs metrics values using logging.
        """
        metrics = self._get_metrics_names()
        for metric_name, descr in metrics.iteritems():
            value = getattr(self, metric_name)
            logging.info('%s: %s', descr, str(value))

    def _get_metrics_names(self):
        return self.METRICS_TO_CALC


class ClassificationModelMetrics(BaseMetrics):
    """
    Represents metrics for classification model
    """
    BINARY_METRICS = {'roc_curve': 'ROC curve',
                      'roc_auc': 'Area under ROC curve',
                      'confusion_matrix': 'Confusion Matrix',
                      'accuracy': 'Accuracy',
                      'average_precision': 'Avarage Precision',
                      'precision_recall_curve': 'Precision-recall curve'}
    MORE_DIMENSIONAL_METRICS = {'confusion_matrix': 'Confusion Matrix',
                                'accuracy': 'Accuracy',
                                'roc_curve': 'ROC curve',
                                'roc_auc': 'Area under ROC curve',
                                }
    DEFAULTS = {'type': 'classification'}

    @property
    def roc_curve(self):
        """
        :return: {label1: [fpr tpr], label2: [fpr tpr], etc}
        """
        if not hasattr(self, '_roc_curve'):
            self._roc_curve = {}
            calculation_range = [1] if self.classes_count == 2 \
                else range(len(self.classes_set))
            for i in calculation_range:
                pos_label = self.classes_set[i]
                # on-vs-all labeling
                labels = [1 if label == pos_label else 0
                          for label in self._labels]
                fpr, tpr, thresholds = \
                    sk_metrics.roc_curve(labels, self._probs[:, i])

                # definitely we can set NaN arrays directly to 0 without
                # checking empty_labels, but it would leave a backdoor
                # for bugs, where unknown reasons for producing NaN would
                # be ignored as a byproduct
                if pos_label in self._empty_labels:
                    tpr = numpy.zeros_like(tpr)

                # A very edge case where there is only one label in the
                # test set
                if self.classes_count - len(self._empty_labels) == 1 and \
                        numpy.all(numpy.isnan(fpr)):
                    fpr = numpy.zeros_like(fpr)

                self._roc_curve[pos_label] = [fpr, tpr]

        return self._roc_curve

    @property
    def average_precision(self):
        from ml_metrics import apk
        if not hasattr(self, '_apk'):
            self._apk = apk(self._labels, self._preds)
        return self._apk

    @property
    def roc_auc(self):
        """
        Calc Area under the ROC curve
        :return: {label1: auc, label2: auc, etc}
        """
        if not hasattr(self, '_roc_auc'):
            self._roc_auc = {}
            calculation_range = [1] if self.classes_count == 2 \
                else range(len(self.classes_set))
            for i in calculation_range:
                pos_label = self.classes_set[i]
                fpr = self.roc_curve[pos_label][0]
                tpr = self.roc_curve[pos_label][1]
                self._roc_auc[pos_label] = sk_metrics.auc(fpr, tpr)
        return self._roc_auc

    @property
    def confusion_matrix(self):
        if not hasattr(self, '_confusion_matrix'):
            y_true_type = type(self._labels[0])
            y_pred_type = type(self._preds[0])
            if y_true_type != y_pred_type:
                y_true = [y_pred_type(y) for y in self._labels]
            else:
                y_true = self._labels
            classes = [y_pred_type(y) for y in self._classes_set]
            self._confusion_matrix = \
                sk_metrics.confusion_matrix(y_true, self._preds, classes)
        return self._confusion_matrix

    @property
    def precision_recall_curve(self):
        if not hasattr(self, '_precision') or not hasattr(self, '_recall'):
            self._precision, self._recall, thresholds = \
                sk_metrics.precision_recall_curve(self._labels,
                                                  self._probs[:, 1])
        return self._precision, self._recall

    @property
    def accuracy(self):
        if not hasattr(self, '_accuracy'):
            labels, preds = prepare_labels_preds(self._labels, self._preds)
            self._accuracy = sk_metrics.accuracy_score(labels, preds)
        return self._accuracy

    def _get_metrics_names(self):
        if self.classes_count == 2:
            return self.BINARY_METRICS
        else:
            return self.MORE_DIMENSIONAL_METRICS


class RegressionModelMetrics(BaseMetrics):
    """
    Represents metrics for regression model
    """
    METRICS_TO_CALC = {
        'explained_variance_score': 'Explained variance regression score'
                                    'function',
        'mean_absolute_error': 'Mean absolute error regression loss',
        'mean_squared_error': 'Mean squared error regression loss',
        'r2_score': 'R^2 (coefficient of determination) regression'
                    'score function.',
    }
    DEFAULTS = {'type': 'regression'}

    @property
    def explained_variance_score(self):
        if not hasattr(self, '_explained_variance_score'):
            labels, preds = prepare_labels_preds(self._labels, self._preds)
            self._explained_variance_score = \
                sk_metrics.explained_variance_score(labels, preds)
        return self._explained_variance_score

    @property
    def mean_absolute_error(self):
        if not hasattr(self, '_mean_absolute_error'):
            labels, preds = prepare_labels_preds(self._labels, self._preds)
            self._mean_absolute_error = \
                sk_metrics.mean_absolute_error(labels, preds)
        return self._mean_absolute_error

    @property
    def mean_squared_error(self):
        if not hasattr(self, '_mean_squared_error'):
            labels, preds = prepare_labels_preds(self._labels, self._preds)
            self._mean_squared_error = \
                sk_metrics.mean_squared_error(labels, preds)
        return self._mean_squared_error

    @property
    def r2_score(self):
        if not hasattr(self, '_r2_score'):
            labels, preds = prepare_labels_preds(self._labels, self._preds)
            self._r2_score = sk_metrics.r2_score(labels, preds)
        return self._r2_score

    def evaluate_model(self, labels, vectorized_data, classifier,
                       empty_labels, segment='default'):
        self._labels += labels
        self._empty_labels = empty_labels
        self._classifier[segment] = classifier

        # Evaluating model...
        if(len(vectorized_data) == 1):
            if isinstance(vectorized_data[0], csc_matrix):
                true_data = vectorized_data[0]
            else:
                true_data = numpy.array(vectorized_data[0])
        else:
            try:
                true_data = hstack(vectorized_data)
            except ValueError:
                true_data = numpy.hstack(vectorized_data)
        try:
            self._true_data[segment] = true_data.tocsr()
        except:
            self._true_data[segment] = true_data

        preds = classifier.predict(true_data)

        if self._preds is None:
            self._preds = preds
        else:
            self._preds = numpy.append(self._preds, preds)

        self._prob = None


def prepare_labels_preds(labels, preds):
    y_true_type = type(labels[0])
    y_pred_type = type(preds[0])
    if y_true_type != y_pred_type:
        labels = [y_pred_type(y) for y in labels]
    else:
        labels = labels
    return labels, preds


class Metrics(object):
    CONFIG = {
        TYPE_CLASSIFICATION: ClassificationModelMetrics,
        TYPE_REGRESSION: RegressionModelMetrics
    }

    @classmethod
    def factory(cls, model_type):
        if model_type not in cls.CONFIG:
            from exceptions import SchemaException
            raise SchemaException(
                '{0} model type isn\'t supported'.format(model_type))
        return cls.CONFIG[model_type]()
