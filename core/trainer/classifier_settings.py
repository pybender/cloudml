# TODO: Maybe this settings should be moved to another place

LOGISTIC_REGRESSION = 'logistic regression'
SVR = 'support vector regression'
SGD_CLASSIFIER = 'stochastic gradient descent classifier'

CLASSIFIER_MODELS = (LOGISTIC_REGRESSION, SGD_CLASSIFIER, DECISION_TREE)
REGRESSION_MODELS = (SVR, DECISION_TREE)


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
        'defaults': {'C': 1.0, 'epsilon': 0.1}}
}
