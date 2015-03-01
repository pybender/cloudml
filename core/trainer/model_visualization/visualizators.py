from ..classifier_settings import *
from weights import WeightsCalculator, SVRWeightsCalculator


class BaseTrainedModelVisualizator(object):
    WEIGHTS_CLS = WeightsCalculator

    def __init__(self, trainer):
        self._trainer = trainer
        self.weights_calc = self.WEIGHTS_CLS(trainer)

    def generate(self, segment, true_data):
        self.weights_calc.generate(segment, true_data)

    def get_weights(self, segment):
        return self.weights_calc.get_weights(segment)

    def get_visualization(self, segment):
        return {
            'weights': self.get_weights(segment),
            'classifier_type': self._trainer.classifier_type
        }


class LRTrainingVisualizer(BaseTrainedModelVisualizator):
    pass


class SVRTrainingVisualizer(BaseTrainedModelVisualizator):
    WEIGHTS_CLS = SVRWeightsCalculator

    def __init__(self, trainer):
        from ..trainer import DEFAULT_SEGMENT
        self._trainer = trainer
        clf = self._trainer.get_classifier(DEFAULT_SEGMENT)
        self.kernel = clf.kernel
        if self.kernel == 'linear':
            self.weights_calc = self.WEIGHTS_CLS(trainer)

    def generate(self, segment, true_data):
        if self.kernel == 'linear':
            return super(SVRTrainingVisualizer, self).generate(segment, true_data)

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
        # clf = self._trainer.get_classifier(segment)
        #res['support_vectors_'] = clf.support_vectors_
        return res


class SGDTrainingVisualizer(BaseTrainedModelVisualizator):
    pass


class DecisionTreeTrainingVisualizer(BaseTrainedModelVisualizator):
    def get_visualization(self, segment):
        res = super(DecisionTreeTrainingVisualizer,
                    self).get_visualization(segment)
        from utils import build_tree
        clf = self._trainer.get_classifier(segment)
        res['tree'] = build_tree(
            clf.tree_,
            self.weights_calc.get_weights(segment, signed=False)
        )
        # exporting to dot file
        # from sklearn import tree
        # tree.export_graphviz(clf, out_file='tree.dot')
        return res


class GBTrainingVisualizer(BaseTrainedModelVisualizator):
    pass


class ExtraTreesTrainingVisualizer(BaseTrainedModelVisualizator):
    def get_visualization(self, segment):
        res = super(ExtraTreesTrainingVisualizer,
                    self).get_visualization(segment)
        from utils import build_tree
        trees_clf = self._trainer.get_classifier(segment)
        res['trees'] = []
        for clf in trees_clf.estimators_:
            tree = build_tree(
                clf.tree_,
                self.weights_calc.get_weights(segment, signed=False)
            )
            res['trees'].append(tree)
        return res


class RandomForestTrainingVisualizer(ExtraTreesTrainingVisualizer):
    pass


class Visualizator(object):
    TRAINING_VISUALIZER_DICT = {
        LOGISTIC_REGRESSION: LRTrainingVisualizer,
        SVR: SVRTrainingVisualizer,
        SGD_CLASSIFIER: SGDTrainingVisualizer,
        DECISION_TREE_CLASSIFIER: DecisionTreeTrainingVisualizer,
        GRADIENT_BOOSTING_CLASSIFIER: GBTrainingVisualizer,
        EXTRA_TREES_CLASSIFIER: ExtraTreesTrainingVisualizer,
        RANDOM_FOREST_CLASSIFIER: RandomForestTrainingVisualizer
    }

    @classmethod
    def factory(cls, trainer):
        return cls.TRAINING_VISUALIZER_DICT[trainer.classifier_type](trainer)
