from ..classifier_settings import *
from weights import WeightsCalculator


class BaseTrainedModelVisualizator(object):
    def __init__(self, trainer):
        self._trainer = trainer
        self.weights_calc = WeightsCalculator(trainer)

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
    pass


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
