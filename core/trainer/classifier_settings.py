# TODO: Maybe this settings should be moved to another place

LOGISTIC_REGRESSION = 'logistic regression'
SVR = 'support vector regression'
SGD_CLASSIFIER = 'stochastic gradient descent classifier'

CLASSIFIER_MODELS = (LOGISTIC_REGRESSION, SGD_CLASSIFIER)
REGRESSION_MODELS = (SVR, )

CLASSIFIERS = {LOGISTIC_REGRESSION: {
                   'cls': 'sklearn.linear_model.LogisticRegression',
                    'parameters': ('penalty', 'C', 'dual', 'fit_intercept',
                                   'intercept_scaling', 'class_weight', 'tol'),
                    'defaults': {'penalty': 'l2'}
               },
               SGD_CLASSIFIER: {
                   'cls': 'sklearn.linear_model.SGDClassifier',
                   'parameters': ('loss', 'penalty', 'alpha', 'l1_ratio', 
                                  'fit_intercept', 'n_iter', 'shuffle', 
                                  'verbose', 'epsilon', 'n_jobs', 
                                  'random_state', 'learning_rate', 'eta0', 
                                  'power_t', 'class_weight', 'warm_start',
                                  'rho', 'seed'),
                   'defaults': {'n_iter': 20, 'shuffle': True}
               },
               SVR: {
                   'cls': 'sklearn.svm.SVR',
                   'parameters': ('C', 'epsilon', 'kernel', 'degree', 'gamma', 
                                  'coef0', 'probability', 'shrinking')}
               }
