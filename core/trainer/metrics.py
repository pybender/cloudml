import logging

import numpy
from scipy.sparse import hstack
import sklearn.metrics as sk_metrics


class Metrics(object):
    METRICS_TO_CALC = ()

    def __init__(self, labels, vectorized_data, classifier):
        self._labels = labels
        self._vectorized_data = vectorized_data
        self._classifier = classifier

        # Evaluating model...
        if(len(self._vectorized_data) == 1):
            self._true_data = numpy.array(self._vectorized_data[0])
        else:
            self._true_data = hstack(self._vectorized_data)
        self._vectorized_data = None
        self._probs = classifier.predict_proba(self._true_data)
        self._preds = classifier.predict(self._true_data)


    @property
    def classes_set(self):
        if not hasattr(self, '_classes_set'):
            self._classes_set = set(self._labels)
        return self._classes_set

    @property
    def classes_count(self):
        if not hasattr(self, '_classes_count'):
            self._classes_count = len(self.classes_set)
        return self._classes_count

    def get_metrics_dict(self):
        """
        Returns dict with metrics values.

        Note: Now it used in the REST API.
        """
        # TODO: Think about moving logic to serializer
        # and make Metrics class serializable
        res = {}
        metrics = self._get_metrics_names()
        for metric_name in metrics.keys():
            value = getattr(self, metric_name)
            if isinstance(value, list) or isinstance(value, tuple):
                value = [val.tolist()
                         if isinstance(val, numpy.ndarray) else val
                         for val in value]

            if isinstance(value, numpy.ndarray):
                value = value.tolist()

            res[metric_name] = value
        return res

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


class ClassificationModelMetrics(Metrics):
    """
    Represents metrics for classification model
    """
    BINARY_METRICS = {'roc_curve': 'ROC curve',
                      'roc_auc': 'Area under ROC curve',
                      'confusion_matrix': 'Confusion Matrix',
                      'accuracy': 'Accuracy',
                      'avarage_precision': 'Avarage Precision',
                      'precision_recall_curve': 'Precision-recall curve'}
    MORE_DIMENSIONAL_METRICS = {'confusion_matrix': 'Confusion Matrix',
                                'accuracy': 'Accuracy'}

    @property
    def roc_curve(self):
        """ Calc roc curve only for binary classification """
        assert self.classes_count == 2
        if not hasattr(self, '_fpr') or not hasattr(self, '_tpr'):
            self._fpr, self._tpr, thresholds = \
                sk_metrics.roc_curve(self._labels, self._probs[:, 1])
        return self._fpr, self._tpr

    @property
    def avarage_precision(self):
        from ml_metrics import apk #, mapk
        if not hasattr(self, '_apk'):
            self._apk = apk(self._labels, self._preds)
        return self._apk

    @property
    def roc_auc(self):
        """ Calc Area under the ROC curve only for binary classification """
        if not hasattr(self, '_roc_auc'):
            self._roc_auc = sk_metrics.auc(*self.roc_curve)
        return self._roc_auc

    @property
    def confusion_matrix(self):
        if not hasattr(self, '_confusion_matrix'):
            self._confusion_matrix = \
                sk_metrics.confusion_matrix(self._labels, self._preds)
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
            self._accuracy = self._classifier.score(self._true_data,
                                                    self._labels)
        return self._accuracy

    def _get_metrics_names(self):
        if self.classes_count == 2:
            return self.BINARY_METRICS
        else:
            return self.MORE_DIMENSIONAL_METRICS


class RegressionModelMetrics(Metrics):
    """
    Represents metrics for regression model
    """
    METRICS_TO_CALC = {'rsme': 'Root square mean error'}

    @property
    def rsme(self):
        if not hasattr(self, '_rsme'):
            sum = 0
            for i, y in enumerate(self._labels):
                x = self._true_data.getrow(i)
                yp = self._classifier.predict(x)[0]
                sum += (y - yp) ** 2
            self._rsme = numpy.sqrt(sum / len(self._labels))
        return self._rsme
