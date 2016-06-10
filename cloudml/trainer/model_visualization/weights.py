# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import logging
import numpy
from operator import itemgetter
from ..classifier_settings import LOGISTIC_REGRESSION, SVR, SGD_CLASSIFIER


class WeightsCalculator(object):
    def __init__(self, trainer, clf_weights_field=None):
        # Structura of weights
        # {
        #   <segment>: {
        #      <class_label>: [
        #         {'name': 'name',
        #          'feature_weight': 'feature_weight',
        #          'weight': 'clf coef for this feature'}, ...]},
        # ...
        # }
        self.weights = {}

        self._trainer = trainer

        if clf_weights_field is None:
            # determine field by clf type
            if self._trainer.classifier_type in (
                    LOGISTIC_REGRESSION, SVR, SGD_CLASSIFIER):
                self.clf_weights_field = 'coef_'
            else:
                self.clf_weights_field = 'feature_importances_'
        else:
            self.clf_weights_field = clf_weights_field

    def generate(self, segment, true_data):
        """
        Calculates list of feature weights for specified segment
        and store it to the weights dict.
        """
        self.weights[segment] = {}  # weights by the class_label
        features = self._trainer.get_features(segment)

        logging.info('Calculate feature weights for %s segment', segment)
        clf = self._trainer.get_classifier(segment)
        if clf.classes_ is None:
            logging.warning('There are no classes in the segment %s', segment)
            return

        true_data.data = numpy.absolute(true_data.data)
        mean_data = true_data.mean(0).transpose()

        if len(clf.classes_) == 2:
            enumeration = [(0, clf.classes_[1])]
        else:
            enumeration = enumerate(clf.classes_)

        fm = self._trainer.feature_model
        target_feature = fm.features[fm.target_variable]
        from ..trainer import _adjust_classifier_class
        for class_index, label in enumeration:
            label = _adjust_classifier_class(target_feature, label)
            logging.info('Get weights for label %s', label)
            self.weights[segment][label] = []

            # filling weights
            class_weights = self._get_clf_class_weights(
                clf, class_index=class_index)
            for i, coef in enumerate(class_weights):
                t = mean_data[i].tolist()[0][0]
                feature_weight = t * numpy.abs(coef)
                self.weights[segment][label].append({
                    'feature_weight': feature_weight,
                    'weight': coef
                })

            self.determine_feature_names(features, segment, label=label)

            if len(clf.classes_) == 2:
                break

    def determine_feature_names(self, features, segment, label):
        index = 0
        for feature_name, feature in features.items():
            base_name = feature_name.replace(".", "->")
            if self._trainer.is_target_variable(feature_name) or \
                    self._trainer.is_group_by_variable(feature_name):
                logging.debug(
                    "We don't set name (%s) to the group or target value."
                    "Segment %s, label %s", base_name, segment, label)
            else:
                transformer = feature['transformer']
                preprocessor = feature['type'].preprocessor

                # Lda, Lsi
                if transformer is not None and \
                        hasattr(transformer, 'num_topics'):
                    for tj in range(0, transformer.num_features - 1):
                        name = '%s->Topic #%d' % (base_name, tj)
                        self.weights[segment][label][index + tj]['name'] = name
                    if transformer.num_features:
                        index += transformer.num_topics

                # Word2Vec, Doc2Vec
                elif transformer is not None and \
                        hasattr(transformer, 'vector_size'):
                    for vj in range(0, transformer.num_features):
                        name = '%s->Vector Element #%d' % (base_name, vj)
                        self.weights[segment][label][index + vj]['name'] = name
                    if transformer.num_features:
                        index += transformer.num_features

                elif transformer is not None and \
                        hasattr(transformer, 'get_feature_names'):
                    names = self._get_feature_names_from_vectorizer(
                        base_name, transformer)
                    for vj, name in enumerate(names):
                        self.weights[segment][label][index + vj]['name'] = name
                    index += len(names)

                elif preprocessor is not None and \
                        hasattr(preprocessor, 'get_feature_names'):
                    names = self._get_feature_names_from_vectorizer(
                        base_name, preprocessor)
                    for vj, name in enumerate(names):
                        self.weights[segment][label][index + vj]['name'] = name
                    index += len(names)

                else:
                    self.weights[segment][label][index]['name'] = base_name
                    index += 1

    def get_weights(self, segment, signed=True):
        """
        Returning positive and negative weights separately
        :param segment:
        :return: {<class_label>:{'positive':[...], 'negative':[...]},
                <class_label2>: etc}
        """
        if segment not in self.weights:
            raise ValueError("Feature weights for {1!s} segment wasn't "
                             "filled: {0}".format(self.weights, segment))

        if signed:
            result = {}
            for class_label, weights_list in self.weights[segment].iteritems():
                result[class_label] = self._get_signed_weights(
                    segment, class_label)
            return result
        else:
            return self.weights[segment]

    # utils

    def _get_clf_weights(self, clf):
        return getattr(clf, self.clf_weights_field)

    def _get_clf_class_weights(self, clf, class_index=None):
        """
        Returns classifier coeficients (logistic regression, etc) or
        gini importance (decision tree classifier) list.
        """
        clf_weights = self._get_clf_weights(clf)
        if self._trainer.classifier_type in (
                LOGISTIC_REGRESSION, SGD_CLASSIFIER):
            return clf_weights[class_index]
        else:
            # TODO: Why???
            return clf_weights

    def _get_feature_names_from_vectorizer(self, name, vectorizer):
        try:
            feature_names = vectorizer.get_feature_names()
        except ValueError:
            return []

        return ['%s->%s' % (name, feature_names[j])
                for j in range(0, len(feature_names))]

    def _get_signed_weights(self, segment, class_label):
        positive = []
        negative = []
        for item in self.weights[segment][class_label]:
            if 'name' not in item:
                debug = item['debug'] if 'debug' in item else 'no debug'
                logging.debug("There isn't a name in weight: %s."
                              " Segment %s, label %s. %s",
                              item, segment, class_label, debug)
            else:
                if item['weight'] > 0:
                    positive.append(item)
                else:
                    negative.append(item)

        positive = sorted(positive, key=itemgetter('name'), reverse=True)
        negative = sorted(negative, key=itemgetter('name'), reverse=False)
        return {'positive': positive, 'negative': negative}


class SVRWeightsCalculator(WeightsCalculator):
    LABEL = 'default'

    def generate(self, segment, true_data):
        """
        Calculates list of feature weights for specified segment
        and store it to the weights dict.
        """
        self.weights[segment] = {}  # weights by the class_label
        features = self._trainer.get_features(segment)

        logging.info('Calculating feature weights for %s segment' % segment)

        # filling weights
        clf = self._trainer.get_classifier(segment)
        # TODO: !!!
        clf_weights = self._get_clf_weights(clf).toarray()[0]

        self.fill_weights(clf_weights, true_data, segment, label=self.LABEL)
        self.determine_feature_names(features, segment, label=self.LABEL)
        # self._calculated_segments.append(segment)

    def fill_weights(self, weights, true_data, segment, label):
        true_data.data = numpy.absolute(true_data.data)
        mean_data = true_data.mean(0).transpose()
        self.weights[segment][label] = []
        for i, coef in enumerate(weights):
            t = mean_data[i].tolist()[0][0]
            feature_weight = t * numpy.abs(coef)
            self.weights[segment][label].append({
                'feature_weight': feature_weight,
                'weight': coef
            })
