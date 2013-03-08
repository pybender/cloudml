# TODO: Maybe this settings should be moved to another place

LOGISTIC_REGRESSION = 'logistic regression'
SVR = 'support vector regression'

CLASSIFIER_MODELS = (LOGISTIC_REGRESSION, )
REGRESSION_MODELS = (SVR, )

CLASSIFIERS = {LOGISTIC_REGRESSION: {'cls': 'sklearn.linear_model.LogisticRegression',
                                     'parameters': ('penalty', 'C', 'dual',
                                                    'fit_intercept',
                                                    'intercept_scaling',
                                                    'class_weight',
                                                    'tol'),
                                     'defaults': {'penalty': 'l2'}},
                SVR: {'cls': 'sklearn.svm.SVR',
                     'parameters': ('C', 'epsilon',
                                    'kernel', 'degree',
                                    'gamma', 'coef0',
                                    'probability',
                                    'shrinking')}
               }
