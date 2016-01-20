"""
This module gathers classifiers description.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>
# TODO: Maybe this settings should be moved to another place,
# for example xml file.

LOGISTIC_REGRESSION = 'logistic regression'
SVR = 'support vector regression'
SGD_CLASSIFIER = 'stochastic gradient descent classifier'
DECISION_TREE_CLASSIFIER = 'decision tree classifier'
DECISION_TREE_REGRESSOR = 'decision tree regressor'

# don't support sparse matrix
GRADIENT_BOOSTING_CLASSIFIER = 'gradient boosting classifier'

EXTRA_TREES_CLASSIFIER = 'extra trees classifier'
RANDOM_FOREST_CLASSIFIER = 'random forest classifier'
RANDOM_FOREST_REGRESSOR = 'random forest regressor'

CLASSIFIER_MODELS = (
    LOGISTIC_REGRESSION, SGD_CLASSIFIER, DECISION_TREE_CLASSIFIER,
    EXTRA_TREES_CLASSIFIER, RANDOM_FOREST_CLASSIFIER,
    GRADIENT_BOOSTING_CLASSIFIER)
REGRESSION_MODELS = (SVR, DECISION_TREE_REGRESSOR, RANDOM_FOREST_REGRESSOR)


TYPE_CLASSIFICATION = 'classification'
TYPE_REGRESSION = 'regression'


class Params:
    """
    Different parameters description.
    """
    n_estimators = {'name': "n_estimators", 'type': 'integer', 'default': 10}
    tree_criterion = {
        'name': "criterion",
        'type': 'string',
        'choices': ['gini', 'entropy'],
        'default': 'gini'}

    max_features = {
        'name': "max_features",
        'type': 'int_float_string_none',
        'choices': ['auto', 'sqrt', 'log2'],
        'default': 'auto'}
    max_depth = {'name': "max_depth", 'type': 'integer'}
    min_samples_split = {
        'name': "min_samples_split",
        'type': 'integer', 'default': 2}
    min_samples_leaf = {
        'name': "min_samples_leaf",
        'type': 'integer', 'default': 1}
    max_leaf_nodes = {'name': "max_leaf_nodes", 'type': 'integer'}
    bootstrap = {'name': "bootstrap", 'type': 'boolean', 'default': True}
    oob_score = {'name': "oob_score", 'type': 'boolean'}
    n_jobs = {'name': "n_jobs", 'type': 'integer', 'default': 1}
    random_state = {'name': "random_state", 'type': 'integer'}
    verbose = {'name': "verbose", 'type': 'integer', 'default': 0}
    tree_splitter = {
        'name': "splitter",
        'type': 'string',
        'choices': ['best', 'random'],
        'default': 'best'}
    min_density = {'name': "min_density", 'type': 'integer'}
    compute_importances = {'name': "compute_importances", 'type': 'boolean'}
    tol = {'name': "tol", 'type': 'float'}
    C = {'name': "C", 'type': 'float',
         'help_text': 'Inverse of regularization strength; must \
be a positive float. Smaller values specify stronger regularization.'}
    dual = {'name': "dual", 'type': 'boolean'}
    fit_intercept = {'name': "fit_intercept", 'type': 'boolean'}
    epsilon = {'name': 'epsilon', 'type': 'float'}
    class_weight = {'name': 'class_weight', 'type': 'auto_dict'}
    intercept_scaling = {'name': "intercept_scaling", 'type': 'float'}
    tree_regressor_criterion = {
        'name': "criterion",
        'type': 'string',
        'choices': ['mse'],
        'default': 'mse'}
    warm_start = {'name': 'warm_start', 'type': 'boolean'}
    min_weight_fraction_leaf = {
        'name': "min_weight_fraction_leaf",
        'type': 'float', 'default': 0.0}


CLASSIFIERS = {
    LOGISTIC_REGRESSION: {
        'cls': 'sklearn.linear_model.LogisticRegression',
        'parameters': [
            {'name': "penalty", 'type': 'string',
             'choices': ['l1', 'l2'], 'default': 'l2',
             'help_text': 'Used to specify the norm used in the penalization. \
The newton-cg and lbfgs solvers support only l2 penalties.'},
            Params.C, Params.dual, Params.fit_intercept,
            Params.intercept_scaling, Params.class_weight,
            {'name': 'max_iter', 'type': 'integer'},
            {'name': 'solver', 'type': 'string',
             'choices': ['newton-cg', 'lbfgs', 'liblinear']},
            {'name': 'multi_class', 'type': 'string',
             'choices': ['ovr', 'multinomial']},
            Params.tol, Params.verbose, Params.random_state],
        'defaults': {'penalty': 'l2'}},
    SGD_CLASSIFIER: {
        'cls': 'sklearn.linear_model.SGDClassifier',
        'parameters': (
            {'name': 'loss', 'type': 'string',
             'choices': [
                 'hinge', 'log', 'modified_huber', 'squared_hinge',
                 'perceptron', 'squared_loss', 'huber',
                 'epsilon_insensitive', 'squared_epsilon_insensitive']},
            {'name': 'penalty', 'type': 'string',
             'choices': ['l1', 'l2', 'elasticnet']},
            {'name': 'alpha', 'type': 'float'},
            {'name': 'l1_ratio', 'type': 'float'},
            Params.fit_intercept,
            {'name': 'n_iter', 'type': 'integer'},
            {'name': 'shuffle', 'type': 'boolean'},
            Params.verbose, Params.epsilon,
            {'name': 'n_jobs', 'type': 'integer'},
            Params.random_state,
            {'name': 'learning_rate', 'type': 'string'},
            {'name': 'eta0', 'type': 'float'},
            {'name': 'power_t', 'type': 'float'},
            Params.class_weight, Params.warm_start,
            {'name': 'rho', 'type': 'string'},
            {'name': 'seed', 'type': 'string'}),
        'defaults': {'n_iter': 20, 'shuffle': True}},
    SVR: {
        'cls': 'sklearn.svm.SVR',
        'parameters': (
            Params.C, Params.epsilon,
            {'name': 'kernel',
             'type': 'string',
             'default': 'linear',
             'choices': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']},
            {'name': 'degree', 'type': 'integer'},
            {'name': 'gamma', 'type': 'float'},
            {'name': 'coef0', 'type': 'float'},
            {'name': 'probability', 'type': 'boolean'},
            {'name': 'shrinking', 'type': 'boolean'},
            Params.tol, Params.verbose),
        'defaults': {'C': 1.0, 'epsilon': 0.1}},
    DECISION_TREE_CLASSIFIER: {
        'cls': 'sklearn.tree.DecisionTreeClassifier',
        'parameters': [
            Params.tree_criterion, Params.tree_splitter,
            Params.max_features, Params.max_depth,
            Params.min_samples_split, Params.min_samples_leaf,
            Params.max_leaf_nodes, Params.random_state,
            Params.min_density,
            Params.class_weight
        ]},
    EXTRA_TREES_CLASSIFIER: {
        'cls': 'sklearn.ensemble.ExtraTreesClassifier',
        'parameters': [
            Params.tree_criterion, Params.n_estimators,
            Params.max_features, Params.max_depth,
            Params.min_samples_split, Params.min_samples_leaf,
            Params.max_leaf_nodes, Params.bootstrap,
            Params.oob_score, Params.n_jobs,
            Params.random_state, Params.verbose]},
    RANDOM_FOREST_CLASSIFIER: {
        'cls': 'sklearn.ensemble.RandomForestClassifier',
        'parameters': [
            Params.tree_criterion, Params.n_estimators,
            Params.max_features, Params.max_depth,
            Params.min_samples_split, Params.min_samples_leaf,
            Params.max_leaf_nodes, Params.bootstrap,
            Params.oob_score, Params.n_jobs,
            Params.random_state, Params.verbose,
            Params.class_weight]},
    RANDOM_FOREST_REGRESSOR: {
        'cls': 'sklearn.ensemble.RandomForestRegressor',
        'parameters': [
            Params.n_estimators, Params.tree_regressor_criterion,
            Params.max_features, Params.max_depth,
            Params.min_samples_split, Params.min_samples_leaf,
            Params.min_weight_fraction_leaf,
            Params.max_leaf_nodes, Params.bootstrap,
            Params.oob_score, Params.n_jobs,
            Params.random_state, Params.verbose,
            Params.warm_start]}
}


def get_model_type(classifier_type):
    """
    >>> get_model_type('support vector regression')
    'regression'
    >>> get_model_type('logistic regression')
    'classification'
    >>> get_model_type('some clf')
    Traceback (most recent call last):
        ...
    SchemaException: classifier some clf not supported
    """
    if classifier_type in CLASSIFIER_MODELS:
        return TYPE_CLASSIFICATION
    elif classifier_type in REGRESSION_MODELS:
        return TYPE_REGRESSION
    from exceptions import SchemaException
    raise SchemaException(
        'classifier {0} not supported'.format(classifier_type))
