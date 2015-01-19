import sklearn
import json
import logging


def build_tree(weights, decision_tree):
    logging.error(weights)
    feature_names = [f['name'] for f in weights]

    root = {}

    def node_to_str(tree, node_id, criterion):
        if not isinstance(criterion, sklearn.tree.tree.six.string_types):
            criterion = "impurity"

        value = tree.value[node_id]
        if tree.n_outputs == 1:
            value = value[0, :]

        if tree.children_left[node_id] == sklearn.tree._tree.TREE_LEAF:
            return {
                "id": str(node_id),
                "criterion": criterion,
                "impurity": tree.impurity[node_id],
                "samples": int(tree.n_node_samples[node_id]),
                "value": value.tolist()
            }
        else:
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = tree.feature[node_id]

            return {
                "id": str(node_id),
                "rule": feature,
                criterion:  round(tree.impurity[node_id], 4),
                "samples": int(tree.n_node_samples[node_id])
            }

    def recurse(item, tree, node_id, criterion, parent=None, depth=0):
        tabs = "  " * depth

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        item['item'] = node_to_str(tree, node_id, criterion)
        if left_child != sklearn.tree._tree.TREE_LEAF and depth < 100:
            item['left'] = {}
            item['right'] = {}
            recurse(item['left'], tree,
                    left_child,
                    criterion=criterion,
                    parent=node_id,
                    depth=depth + 1)
            recurse(item['right'], tree,
                    left_child,
                    criterion=criterion,
                    parent=node_id,
                    depth=depth + 1)

    recurse(root, decision_tree, 0, criterion="impurity")
    return root
