# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

from ..classifier_settings import LOGISTIC_REGRESSION, SVR, \
    SGD_CLASSIFIER, DECISION_TREE_CLASSIFIER, \
    GRADIENT_BOOSTING_CLASSIFIER, EXTRA_TREES_CLASSIFIER, \
    RANDOM_FOREST_CLASSIFIER, RANDOM_FOREST_REGRESSOR
from weights import WeightsCalculator, SVRWeightsCalculator


class BaseTrainedModelVisualizer(object):
    WEIGHTS_CLS = WeightsCalculator

    def __init__(self, trainer):
        self._trainer = trainer
        self.weights_calc = self.WEIGHTS_CLS(trainer)

    def generate(self, segment, true_data):
        self.weights_calc.generate(segment, true_data)

    def get_weights(self, segment, **kwargs):
        return self.weights_calc.get_weights(segment)

    def get_visualization(self, segment, **kwargs):
        return {
            'weights': self.get_weights(segment, **kwargs),
            'classifier_type': self._trainer.classifier_type
        }


class LRTrainingVisualizer(BaseTrainedModelVisualizer):
    pass


class SVRTrainingVisualizer(BaseTrainedModelVisualizer):
    WEIGHTS_CLS = SVRWeightsCalculator

    def __init__(self, trainer):
        from ..trainer import DEFAULT_SEGMENT
        clf = trainer.get_classifier(DEFAULT_SEGMENT)
        self.kernel = clf.kernel
        if self.kernel == 'linear':
            super(SVRTrainingVisualizer, self).__init__(trainer)
        else:
            self._trainer = trainer

    def generate(self, segment, true_data):
        if self.kernel == 'linear':
            return super(SVRTrainingVisualizer, self).generate(
                segment, true_data)

    def get_weights(self, segment):
        if self.kernel == 'linear':
            return super(SVRTrainingVisualizer, self).get_weights(segment)
        else:
            raise ValueError("Storing weights are unavailable: coef_ is only "
                             "available when using a linear kernel")

    def get_visualization(self, segment):
        res = {
            'classifier_type': self._trainer.classifier_type,
            'kernel': self.kernel
        }
        if self.kernel == 'linear':
            res['weights'] = self.get_weights(segment)
        return res


class SGDTrainingVisualizer(BaseTrainedModelVisualizer):
    pass


class DecisionTreeTrainingVisualizer(BaseTrainedModelVisualizer):
    DEFAULT_DEEP = 100

    def regenerate_tree(self, segment, weights, deep=DEFAULT_DEEP):
        from utils import build_tree
        clf = self._trainer.get_classifier(segment)
        return build_tree(clf.tree_, weights, max_deep=deep)

    def get_visualization(self, segment, deep=DEFAULT_DEEP):
        res = super(DecisionTreeTrainingVisualizer,
                    self).get_visualization(segment)
        weights = self.weights_calc.get_weights(segment, signed=False)
        res['all_weights'] = weights
        res['tree'] = self.regenerate_tree(segment, weights, deep)
        res['parameters'] = {'deep': deep}
        # exporting to dot file
        # from sklearn import tree
        # tree.export_graphviz(clf, out_file='tree.dot')
        return res


class GBTrainingVisualizer(BaseTrainedModelVisualizer):
    pass


class ExtraTreesTrainingVisualizer(BaseTrainedModelVisualizer):
    DEFAULT_DEEP = 100

    def regenerate_trees(self, segment, weights, deep=DEFAULT_DEEP):
        from utils import build_tree
        trees = []
        trees_clf = self._trainer.get_classifier(segment)
        for clf in trees_clf.estimators_:
            tree = build_tree(
                clf.tree_,
                # self.weights_calc.get_weights(segment, signed=False),
                weights,
                max_deep=deep
            )
            trees.append(tree)
        return trees

    def get_visualization(self, segment, deep=DEFAULT_DEEP):
        res = super(ExtraTreesTrainingVisualizer,
                    self).get_visualization(segment)
        weights = self.weights_calc.get_weights(segment, signed=False)
        res['trees'] = self.regenerate_trees(segment, weights, deep)
        return res


class RandomForestTrainingVisualizer(ExtraTreesTrainingVisualizer):
    pass


class RandomForestRegressorTV(BaseTrainedModelVisualizer):
    def generate(self, segment, true_data):
        pass

    def get_visualization(self, segment):
        res = {
            'classifier_type': self._trainer.classifier_type,
        }
        return res


class Visualizer(object):
    TRAINING_VISUALIZER_DICT = {
        LOGISTIC_REGRESSION: LRTrainingVisualizer,
        SVR: SVRTrainingVisualizer,
        SGD_CLASSIFIER: SGDTrainingVisualizer,
        DECISION_TREE_CLASSIFIER: DecisionTreeTrainingVisualizer,
        GRADIENT_BOOSTING_CLASSIFIER: GBTrainingVisualizer,
        EXTRA_TREES_CLASSIFIER: ExtraTreesTrainingVisualizer,
        RANDOM_FOREST_CLASSIFIER: RandomForestTrainingVisualizer,
        RANDOM_FOREST_REGRESSOR: RandomForestRegressorTV
    }

    @classmethod
    def factory(cls, trainer):
        return cls.TRAINING_VISUALIZER_DICT[trainer.classifier_type](trainer)
