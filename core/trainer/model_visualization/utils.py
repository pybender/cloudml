import sklearn
import json
import logging


def build_tree(decision_tree, weights_list):
    weights = []
    for class_label, class_weights in weights_list.iteritems():
        weights += class_weights

    feature_names = [f['name'] if 'name' in f else 'noname' for f in weights]

    root = []

    def node_to_str(tree, node_id, criterion, type_='yes'):
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
                "value": value.tolist(),
                "node_type": "leaf",
                "type": type_
            }
        else:
            if feature_names is not None:
                feature = feature_names[tree.feature[node_id]]
            else:
                feature = tree.feature[node_id]

            return {
                "id": str(node_id),
                "rule": feature,
                "node_type": "branch",
                criterion:  round(tree.impurity[node_id], 4),
                "samples": int(tree.n_node_samples[node_id]),
                "name": "%s <= %.4f" % (feature, tree.threshold[node_id]),
                "type": type_,
                "children": []
            }

    def recurse(item, tree, node_id, criterion,
                parent=None, type_='yes'):

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        node = node_to_str(tree, node_id, criterion, type_)
        item.append(node)
        if left_child != sklearn.tree._tree.TREE_LEAF:
            recurse(node['children'], tree,
                    left_child,
                    criterion=criterion,
                    parent=node_id)
            recurse(node['children'], tree,
                    right_child,
                    criterion=criterion,
                    parent=node_id,
                    type_='no')

    recurse(root, decision_tree, 0, criterion="impurity")
    return root[0] if root else None
