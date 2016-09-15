# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>
import logging
import re

from ..classifier_settings import (LOGISTIC_REGRESSION, SVR,
                                   SGD_CLASSIFIER, DECISION_TREE_CLASSIFIER,
                                   GRADIENT_BOOSTING_CLASSIFIER, EXTRA_TREES_CLASSIFIER,
                                   RANDOM_FOREST_CLASSIFIER, RANDOM_FOREST_REGRESSOR,
                                   XGBOOST_REGRESSOR, XGBOOST_CLASSIFIER)
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

class XGBClassifierTree():
    _NODEPAT = re.compile(r'(\d+):\[(.+)\]')
    _LEAFPAT = re.compile(r'(\d+):(leaf=(.+))')
    _EDGEPAT = re.compile(r'yes=(\d+),no=(\d+),missing=(\d+)')
    _EDGEPAT2 = re.compile(r'yes=(\d+),no=(\d+)')

    def __init__(self, model):
        self.node = []
        self.graph = {}
        self.trees = {}
        self.parents = {}
        booster = model.booster()
        tree = booster.get_dump()[0]
        self.intree = tree.split()

    def upper_bound(self, key):
        left = -1
        right = len(self.node)
        while right > left + 1:
            middle = (left + right) // 2
            if self.node[middle] > key:
                right = middle
            else:
                left = middle
        return right

    def _parse_node(self, text):
        try:
            match = self._NODEPAT.match(text)
            if match is not None:
                node = match.group(1)
                label=match.group(2)
                self.graph[node] = {'label':label, 'node_type': 'branch'}
                try:
                    self.graph[node]['parent'] = self.parents[node]['parent']
                    self.graph[node]['type'] = self.parents[node]['type']
                except:
                    pass
                return node

            match = self._LEAFPAT.match(text)
            if match is not None:
                node = match.group(1)
                label=match.group(2)
                try:
                    leafpatvalue = float(match.group(3))
                except:
                    leafpatvalue = 0
                self.graph[node] = {'label':label, 'leaf': leafpatvalue, 'node_type': 'leaf'}
                try:
                    self.graph[node]['parent'] = self.parents[node]['parent']
                    self.graph[node]['type'] = self.parents[node]['type']
                except:
                    pass
                return node
        except:
            pass
        raise ValueError('Unable to parse node: {0}'.format(text))

    def _parse_children(self, node, text):

        try:
            match = self._EDGEPAT.match(text)
            if match is not None:
                yes, no, missing = match.groups()
                self.graph[node]['yes'] = yes
                self.graph[node]['no'] = no
                self.graph[node]['missing'] = missing
                if  not hasattr(self.parents, yes):
                    self.parents[yes] = {'parent': node, 'type': 'yes'}
                if  not hasattr(self.parents, no):
                    self.parents[no] = {'parent': node, 'type': 'no'}
                return
            match = self._EDGEPAT2.match(text)
            if match is not None:
                yes, no = match.groups()
                self.graph[node]['yes'] = yes
                self.graph[node]['no'] = no
                if  not hasattr(self.parents, yes):
                    self.parents[yes] = {'parent': node, 'type': 'yes'}
                if  not hasattr(self.parents, no):
                    self.parents[no] = {'parent': node, 'type': 'no'}
                return
        except ValueError:
            pass
        raise ValueError('Unable to parse edge: {0}'.format(text))

    def parse_graph(self):
        node = 0
        for i, text in enumerate(self.intree):
            if text[0].isdigit():
                node = self._parse_node(text)
                self.node.insert(self.upper_bound(int(node)), int(node))
            else:
                if i == 0:
                    raise ValueError('Unable to parse given string as tree')
                self._parse_children(node, text)

    def make_tree(self, node=None, parent=None):
        if True:
            parent = self.graph[node]
            children = []
            samples = 1
            try:
                yeschildren = self.make_tree(node=parent['yes'])
                children.append(yeschildren)
                samples = samples + yeschildren['samples']
            except:
                pass
            try:
                nochildren = self.make_tree(node=parent['no'])
                children.append(nochildren)
                samples = samples + nochildren['samples']
            except:
                pass
            trees = {'node_type': parent['node_type'],'impurity': 0,'samples': samples,'type': 'yes', 'id': node,'rule': u'y','name': parent['label']}
            if len(children) > 0:
                trees['children'] = children
            #
            return trees
        try:
            pass
        except:
            pass
        raise ValueError('Unable to parse main node')

class XGBOOSTTrainingVisualizer(BaseTrainedModelVisualizer):
    def regenerate_trees(self, segment, weights):
        clf = self._trainer.get_classifier(segment)
        xgbtree = XGBClassifierTree(clf)
        xgbtree.parse_graph()
        trees = [xgbtree.make_tree(node = str(xgbtree.node[0]))]
        return trees

    def get_visualization(self, segment, **kwargs):
        res = super(XGBOOSTTrainingVisualizer,self).get_visualization(segment)
        weights = self.weights_calc.get_weights(segment, signed=False)
        res['trees'] = self.regenerate_trees(segment, weights)
        return res

class XGBOOSTTrainingRegressorVisualizer(BaseTrainedModelVisualizer):
    logging.info('Visualize model: XGBOOSTTrainingRegressorVisualizer')

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
        RANDOM_FOREST_REGRESSOR: RandomForestRegressorTV,
        XGBOOST_REGRESSOR: XGBOOSTTrainingRegressorVisualizer,
        XGBOOST_CLASSIFIER: XGBOOSTTrainingVisualizer
    }
    @classmethod
    def factory(cls, trainer):
        return cls.TRAINING_VISUALIZER_DICT[trainer.classifier_type](trainer)
