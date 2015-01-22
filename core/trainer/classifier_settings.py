# TODO: Maybe this settings should be moved to another place

LOGISTIC_REGRESSION = 'logistic regression'
SVR = 'support vector regression'
SGD_CLASSIFIER = 'stochastic gradient descent classifier'
DECISION_TREE_CLASSIFIER = 'decision tree classifier'
DECISION_TREE_REGRESSOR = 'decision tree regressor'

# don't support sparse matrix
GRADIENT_BOOSTING_CLASSIFIER = 'gradient boosting classifier'

EXTRA_TREES_CLASSIFIER = 'extra trees classifier'
RANDOM_FOREST_CLASSIFIER = 'random forest classifier'

CLASSIFIER_MODELS = (
    LOGISTIC_REGRESSION, SGD_CLASSIFIER, DECISION_TREE_CLASSIFIER,
    EXTRA_TREES_CLASSIFIER, RANDOM_FOREST_CLASSIFIER, GRADIENT_BOOSTING_CLASSIFIER)
REGRESSION_MODELS = (SVR, DECISION_TREE_REGRESSOR)

DECISION_TREE_PARAMS = [
    {'name': "splitter",
     'type': 'string',
     'choices': ['best', 'random'],
     'default': 'best'},
    #{'name': 'max_features', 'type': 'integer'},
    {'name': "max_depth", 'type': 'integer'},
    {'name': "min_samples_split", 'type': 'integer', 'default': 2},
    {'name': "min_samples_leaf", 'type': 'integer', 'default': 1},
    {'name': "max_leaf_nodes", 'type': 'integer'},
    # TODO: support RandomState instance
    {'name': "random_state", 'type': 'integer'},
    # {'name': "min_density", 'type': 'integer'},
    # {'name': "compute_importances", 'type': 'boolean'}
]


DECISION_TREE_CLASSIFIER_PARAMS = [
    {'name': "criterion",
     'type': 'string',
     'choices': ['gini', 'entropy'],
     'default': 'gini'}] + DECISION_TREE_PARAMS


CLASSIFIERS = {
    LOGISTIC_REGRESSION: {
        'cls': 'sklearn.linear_model.LogisticRegression',
        'parameters': [
            {'name': "penalty",
             'type': 'string',
             'choices': ['l1', 'l2'],
             'default': 'l2',
             'required': True},
            {'name': "C", 'type': 'float'},
            {'name': "dual", 'type': 'boolean'},
            {'name': "fit_intercept", 'type': 'boolean'},
            {'name': "intercept_scaling", 'type': 'float'},
            {'name': "class_weight", 'type': 'auto_dict'},
            {'name': "tol", 'type': 'float'}],
        'defaults': {'penalty': 'l2'}},
    SGD_CLASSIFIER: {
        'cls': 'sklearn.linear_model.SGDClassifier',
        'parameters': (
            {'name': 'loss',
             'type': 'string',
             'choices': [
                 'hinge', 'log', 'modified_huber', 'squared_hinge',
                 'perceptron', 'squared_loss', 'huber',
                 'epsilon_insensitive', 'squared_epsilon_insensitive']},
            {'name': 'penalty',
             'type': 'string',
             'choices': ['l1', 'l2', 'elasticnet']},
            {'name': 'alpha', 'type': 'float'},
            {'name': 'l1_ratio', 'type': 'float'},
            {'name': 'fit_intercept', 'type': 'boolean'},
            {'name': 'n_iter', 'type': 'integer'},
            {'name': 'shuffle', 'type': 'boolean'},
            {'name': 'verbose', 'type': 'integer'},
            {'name': 'epsilon', 'type': 'float'},
            {'name': 'n_jobs', 'type': 'integer'},
            {'name': 'random_state', 'type': 'integer'},
            {'name': 'learning_rate', 'type': 'string'},
            {'name': 'eta0', 'type': 'float'},
            {'name': 'power_t', 'type': 'float'},
            {'name': 'class_weight', 'type': 'dict'},
            {'name': 'warm_start', 'type': 'boolean'},
            {'name': 'rho', 'type': 'string'},
            {'name': 'seed', 'type': 'string'}),
        'defaults': {'n_iter': 20, 'shuffle': True}},
    SVR: {
        'cls': 'sklearn.svm.SVR',
        'parameters': (
            {'name': "C", 'type': 'float'},
            {'name': 'epsilon', 'type': 'float'},
            {'name': 'kernel', 'type': 'string'},
            {'name': 'degree', 'type': 'integer'},
            {'name': 'gamma', 'type': 'float'},
            {'name': 'coef0', 'type': 'float'},
            {'name': 'probability', 'type': 'boolean'},
            {'name': 'shrinking', 'type': 'boolean'}),
        'defaults': {'C': 1.0, 'epsilon': 0.1}},
    DECISION_TREE_CLASSIFIER: {
        'cls': 'sklearn.tree.DecisionTreeClassifier',
        'parameters': DECISION_TREE_CLASSIFIER_PARAMS},
    GRADIENT_BOOSTING_CLASSIFIER: {
        'cls': 'sklearn.ensemble.GradientBoostingClassifier',
        'parameters': [
            {'name': "loss",
             'type': 'string',
             'choices': ['deviance'],
             'default': 'deviance'},
            {'name': "learning_rate", 'type': 'float', 'default': 0.1},
            {'name': "n_estimators", 'type': 'integer', 'default': 100},
            {'name': "max_depth", 'type': 'integer', 'default': 3},
            {'name': "min_samples_split", 'type': 'integer', 'default': 2},
            {'name': "min_samples_leaf", 'type': 'integer', 'default': 1},
            {'name': "subsample", 'type': 'float', 'default': 1.0},
            {'name': "max_features",
             'type': 'string',
             'choices': ['auto', 'sqrt', 'log2'],
             'default': 'auto'},
            {'name': "max_leaf_nodes", 'type': 'integer'},
            {'name': "verbose", 'type': 'integer', 'default': 0},
            {'name': "warm_start", 'type': 'boolean'}]},
    EXTRA_TREES_CLASSIFIER: {
        'cls': 'sklearn.ensemble.ExtraTreesClassifier',
        'parameters': [
            {'name': "criterion",
             'type': 'string',
             'choices': ['gini', 'entropy'],
             'default': 'gini'},
            {'name': "n_estimators", 'type': 'integer', 'default': 10}]},
    RANDOM_FOREST_CLASSIFIER: {
        'cls': 'sklearn.ensemble.RandomForestClassifier',
        'parameters': [
            {'name': "criterion",
             'type': 'string',
             'choices': ['gini', 'entropy'],
             'default': 'gini'},
            {'name': "n_estimators", 'type': 'integer', 'default': 10}]}
}
