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

TYPE_CLASSIFICATION = 'classification'
TYPE_REGRESSION = 'regression'

FULL_SUPPORT = [
    LOGISTIC_REGRESSION, SVR, SGD_CLASSIFIER,
    DECISION_TREE_CLASSIFIER, EXTRA_TREES_CLASSIFIER,
    RANDOM_FOREST_REGRESSOR, RANDOM_FOREST_CLASSIFIER
]

Infinity = 100000000000

#### generated classifiers config start


CLASSIFIER_MODELS = ['KNeighborsClassifier', 'RadiusNeighborsClassifier', 'NearestCentroid', 'EllipticEnvelope', 'decision tree classifier', 'BaggingClassifier', 'AdaBoostClassifier', 'gradient boosting classifier', 'OneVsRestClassifier', 'OneVsOneClassifier', 'OutputCodeClassifier', 'QDA', 'RidgeClassifier', 'RidgeClassifierCV', 'LinearSVC', 'logistic regression', 'LogisticRegressionCV', 'LDA', 'stochastic gradient descent classifier', 'PassiveAggressiveClassifier', 'Perceptron', 'LogisticRegressionCV', 'SVC', 'NuSVC', 'random forest classifier', 'extra trees classifier', 'MultinomialNB', 'BernoulliNB', 'LabelPropagation', 'LabelSpreading']
REGRESSION_MODELS = ['KNeighborsRegressor', 'RadiusNeighborsRegressor', 'LinearRegression', 'BayesianRidge', 'ARDRegression', 'Lars', 'ElasticNet', 'LassoCV', 'ElasticNetCV', 'MultiTaskElasticNetCV', 'MultiTaskLassoCV', 'Ridge', 'RidgeCV', 'LinearSVR', 'support vector regression', 'NuSVR', 'OrthogonalMatchingPursuit', 'OrthogonalMatchingPursuitCV', 'RANSACRegressor', 'TheilSenRegressor', 'IsotonicRegression', 'DummyRegressor', 'decision tree regressor', 'BaggingRegressor', 'AdaBoostRegressor', 'GradientBoostingRegressor', 'GaussianProcess', 'KernelRidge', 'LassoLars', 'LarsCV', 'LassoLarsIC', 'LassoLarsCV', 'Lasso', 'MultiTaskElasticNet', 'MultiTaskLasso', 'SGDRegressor', 'PassiveAggressiveRegressor', 'PLSRegression', 'PLSCanonical', 'CCA', 'random forest regressor', 'ExtraTreesRegressor']
CLASSIFIERS = {
    "ARDRegression": {
        "cls": "sklearn.linear_model.bayes.ARDRegression", 
        "defaults": {
            "alpha_1": 1e-06, 
            "alpha_2": 1e-06, 
            "compute_score": False, 
            "copy_X": True, 
            "fit_intercept": True, 
            "lambda_1": 1e-06, 
            "lambda_2": 1e-06, 
            "n_iter": 300, 
            "normalize": False, 
            "threshold_lambda": 10000.0, 
            "tol": 0.001, 
            "verbose": False
        }, 
        "help_text": "Bayesian ARD regression.  Fit the weights of a regression model, using an ARD prior. The weights of  the regression model are assumed to be in Gaussian distributions.  Also estimate the parameters lambda (precisions of the distributions of the  weights) and alpha (precision of the distribution of the noise).  The estimation is done by an iterative procedures (Evidence Maximization)", 
        "parameters": [
            {
                "default": 300, 
                "help_text": "Maximum number of iterations. Default is 300\n", 
                "name": "n_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.001, 
                "help_text": "Stop the algorithm if w has converged. Default is 1.e-3.\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 1e-06, 
                "name": "alpha_1", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 1e-06, 
                "name": "alpha_2", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 1e-06, 
                "name": "lambda_1", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 1e-06, 
                "name": "lambda_2", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "If True, compute the objective function at each step of the model.\nDefault is False.\n", 
                "name": "compute_score", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 10000.0, 
                "help_text": "threshold for removing (pruning) weights with high precision from\nthe computation. Default is 1.e+4.\n", 
                "name": "threshold_lambda", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\nDefault is True.\n", 
                "name": "fit_intercept", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If True, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Verbose mode when fitting the model.\n\nAttributes\n----------", 
                "name": "verbose", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "AdaBoostClassifier": {
        "cls": "sklearn.ensemble.weight_boosting.AdaBoostClassifier", 
        "defaults": {
            "algorithm": "SAMME.R", 
            "learning_rate": 1.0, 
            "n_estimators": 50
        }, 
        "help_text": "An AdaBoost classifier.  An AdaBoost [1] classifier is a meta-estimator that begins by fitting a  classifier on the original dataset and then fits additional copies of the  classifier on the same dataset but where the weights of incorrectly  classified instances are adjusted such that subsequent classifiers focus  more on difficult cases.  This class implements the algorithm known as AdaBoost-SAMME [2].", 
        "parameters": [
            {
                "help_text": "The base estimator from which the boosted ensemble is built.\nSupport for sample weighting is required, as well as proper `classes_`\nand `n_classes_` attributes.\n", 
                "name": "base_estimator", 
                "required": False, 
                "type": "object"
            }, 
            {
                "default": 50, 
                "help_text": "The maximum number of estimators at which boosting is terminated.\nIn case of perfect fit, the learning procedure is stopped early.\n", 
                "name": "n_estimators", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1.0, 
                "help_text": "Learning rate shrinks the contribution of each classifier by\n``learning_rate``. There is a trade-off between ``learning_rate`` and\n``n_estimators``.\n", 
                "name": "learning_rate", 
                "required": False, 
                "type": "float"
            }, 
            {
                "choices": [
                    "SAMME", 
                    "SAMME.R"
                ], 
                "default": "SAMME.R", 
                "help_text": "If 'SAMME.R' then use the SAMME.R real boosting algorithm.\n``base_estimator`` must support calculation of class probabilities.\nIf 'SAMME' then use the SAMME discrete boosting algorithm.\nThe SAMME.R algorithm typically converges faster than SAMME,\nachieving a lower test error with fewer boosting iterations.\n", 
                "name": "algorithm", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "If int, random_state is the seed used by the random number generator;\nIf RandomState instance, random_state is the random number generator;\nIf None, the random number generator is the RandomState instance used\nby `np.random`.\n\nAttributes\n----------", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "AdaBoostRegressor": {
        "cls": "sklearn.ensemble.weight_boosting.AdaBoostRegressor", 
        "defaults": {
            "learning_rate": 1.0, 
            "loss": "linear", 
            "n_estimators": 50
        }, 
        "help_text": "An AdaBoost regressor.  An AdaBoost [1] regressor is a meta-estimator that begins by fitting a  regressor on the original dataset and then fits additional copies of the  regressor on the same dataset but where the weights of instances are  adjusted according to the error of the current prediction. As such,  subsequent regressors focus more on difficult cases.  This class implements the algorithm known as AdaBoost.R2 [2].", 
        "parameters": [
            {
                "help_text": "The base estimator from which the boosted ensemble is built.\nSupport for sample weighting is required.\n", 
                "name": "base_estimator", 
                "required": False, 
                "type": "object"
            }, 
            {
                "default": 50, 
                "help_text": "The maximum number of estimators at which boosting is terminated.\nIn case of perfect fit, the learning procedure is stopped early.\n", 
                "name": "n_estimators", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1.0, 
                "help_text": "Learning rate shrinks the contribution of each regressor by\n``learning_rate``. There is a trade-off between ``learning_rate`` and\n``n_estimators``.\n", 
                "name": "learning_rate", 
                "required": False, 
                "type": "float"
            }, 
            {
                "choices": [
                    "linear", 
                    "square", 
                    "exponential"
                ], 
                "default": "linear", 
                "help_text": "The loss function to use when updating the weights after each\nboosting iteration.\n", 
                "name": "loss", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "If int, random_state is the seed used by the random number generator;\nIf RandomState instance, random_state is the random number generator;\nIf None, the random number generator is the RandomState instance used\nby `np.random`.\n\nAttributes\n----------", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "BaggingClassifier": {
        "cls": "sklearn.ensemble.bagging.BaggingClassifier", 
        "defaults": {
            "bootstrap": True, 
            "bootstrap_features": False, 
            "max_features": 1.0, 
            "max_samples": 1.0, 
            "n_estimators": 10, 
            "n_jobs": 1, 
            "oob_score": False, 
            "verbose": 0
        }, 
        "help_text": "A Bagging classifier.  A Bagging classifier is an ensemble meta-estimator that fits base  classifiers each on random subsets of the original dataset and then  aggregate their individual predictions (either by voting or by averaging)  to form a final prediction. Such a meta-estimator can typically be used as  a way to reduce the variance of a black-box estimator (e.g., a decision  tree), by introducing randomization into its construction procedure and  then making an ensemble out of it.  This algorithm encompasses several works from the literature. When random  subsets of the dataset are drawn as random subsets of the samples, then  this algorithm is known as Pasting [1]_. If samples are drawn with  replacement, then the method is known as Bagging [2]_. When random subsets  of the dataset are drawn as random subsets of the features, then the method  is known as Random Subspaces [3]_. Finally, when base estimators are built  on subsets of both samples and features, then the method is known as  Random Patches [4]_.", 
        "parameters": [
            {
                "help_text": "The base estimator to fit on random subsets of the dataset.\nIf None, then the base estimator is a decision tree.\n", 
                "name": "base_estimator", 
                "required": False, 
                "type": "object_or_none"
            }, 
            {
                "default": 10, 
                "help_text": "The number of base estimators in the ensemble.\n", 
                "name": "n_estimators", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1.0, 
                "help_text": "The number of samples to draw from X to train each base estimator.\n- If int, then draw `max_samples` samples.\n- If float, then draw `max_samples * X.shape[0]` samples.\n", 
                "name": "max_samples", 
                "required": False, 
                "type": "float_or_integer"
            }, 
            {
                "default": 1.0, 
                "help_text": "The number of features to draw from X to train each base estimator.\n- If int, then draw `max_features` features.\n- If float, then draw `max_features * X.shape[1]` features.\n", 
                "name": "max_features", 
                "required": False, 
                "type": "float_or_integer"
            }, 
            {
                "default": True, 
                "help_text": "Whether samples are drawn with replacement.\n", 
                "name": "bootstrap", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Whether features are drawn with replacement.\n", 
                "name": "bootstrap_features", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Whether to use out-of-bag samples to estimate\nthe generalization error.\n", 
                "name": "oob_score", 
                "type": "boolean"
            }, 
            {
                "default": 1, 
                "help_text": "The number of jobs to run in parallel for both `fit` and `predict`.\nIf -1, then the number of jobs is set to the number of cores.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "If int, random_state is the seed used by the random number generator;\nIf RandomState instance, random_state is the random number generator;\nIf None, the random number generator is the RandomState instance used\nby `np.random`.\n", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "Controls the verbosity of the building process.\n\nAttributes\n----------", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "BaggingRegressor": {
        "cls": "sklearn.ensemble.bagging.BaggingRegressor", 
        "defaults": {
            "bootstrap": True, 
            "bootstrap_features": False, 
            "max_features": 1.0, 
            "max_samples": 1.0, 
            "n_estimators": 10, 
            "n_jobs": 1, 
            "oob_score": False, 
            "verbose": 0
        }, 
        "help_text": "A Bagging regressor.  A Bagging regressor is an ensemble meta-estimator that fits base  regressors each on random subsets of the original dataset and then  aggregate their individual predictions (either by voting or by averaging)  to form a final prediction. Such a meta-estimator can typically be used as  a way to reduce the variance of a black-box estimator (e.g., a decision  tree), by introducing randomization into its construction procedure and  then making an ensemble out of it.  This algorithm encompasses several works from the literature. When random  subsets of the dataset are drawn as random subsets of the samples, then  this algorithm is known as Pasting [1]_. If samples are drawn with  replacement, then the method is known as Bagging [2]_. When random subsets  of the dataset are drawn as random subsets of the features, then the method  is known as Random Subspaces [3]_. Finally, when base estimators are built  on subsets of both samples and features, then the method is known as  Random Patches [4]_.", 
        "parameters": [
            {
                "help_text": "The base estimator to fit on random subsets of the dataset.\nIf None, then the base estimator is a decision tree.\n", 
                "name": "base_estimator", 
                "required": False, 
                "type": "object_or_none"
            }, 
            {
                "default": 10, 
                "help_text": "The number of base estimators in the ensemble.\n", 
                "name": "n_estimators", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1.0, 
                "help_text": "The number of samples to draw from X to train each base estimator.\n- If int, then draw `max_samples` samples.\n- If float, then draw `max_samples * X.shape[0]` samples.\n", 
                "name": "max_samples", 
                "required": False, 
                "type": "float_or_integer"
            }, 
            {
                "default": 1.0, 
                "help_text": "The number of features to draw from X to train each base estimator.\n- If int, then draw `max_features` features.\n- If float, then draw `max_features * X.shape[1]` features.\n", 
                "name": "max_features", 
                "required": False, 
                "type": "float_or_integer"
            }, 
            {
                "default": True, 
                "help_text": "Whether samples are drawn with replacement.\n", 
                "name": "bootstrap", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Whether features are drawn with replacement.\n", 
                "name": "bootstrap_features", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Whether to use out-of-bag samples to estimate\nthe generalization error.\n", 
                "name": "oob_score", 
                "type": "boolean"
            }, 
            {
                "default": 1, 
                "help_text": "The number of jobs to run in parallel for both `fit` and `predict`.\nIf -1, then the number of jobs is set to the number of cores.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "If int, random_state is the seed used by the random number generator;\nIf RandomState instance, random_state is the random number generator;\nIf None, the random number generator is the RandomState instance used\nby `np.random`.\n", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "Controls the verbosity of the building process.\n\nAttributes\n----------", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "BayesianRidge": {
        "cls": "sklearn.linear_model.bayes.BayesianRidge", 
        "defaults": {
            "alpha_1": 1e-06, 
            "alpha_2": 1e-06, 
            "compute_score": False, 
            "copy_X": True, 
            "fit_intercept": True, 
            "lambda_1": 1e-06, 
            "lambda_2": 1e-06, 
            "n_iter": 300, 
            "normalize": False, 
            "tol": 0.001, 
            "verbose": False
        }, 
        "help_text": "Bayesian ridge regression  Fit a Bayesian ridge model and optimize the regularization parameters  lambda (precision of the weights) and alpha (precision of the noise).", 
        "parameters": [
            {
                "default": 300, 
                "help_text": "Maximum number of iterations. Default is 300.\n", 
                "name": "n_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.001, 
                "help_text": "Stop the algorithm if w has converged. Default is 1.e-3.\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 1e-06, 
                "name": "alpha_1", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 1e-06, 
                "name": "alpha_2", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 1e-06, 
                "name": "lambda_1", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 1e-06, 
                "name": "lambda_2", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "If True, compute the objective function at each step of the model.\nDefault is False\n", 
                "name": "compute_score", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\nDefault is True.\n", 
                "name": "fit_intercept", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If True, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Verbose mode when fitting the model.\n\n\nAttributes\n----------", 
                "name": "verbose", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "BernoulliNB": {
        "cls": "sklearn.naive_bayes.BernoulliNB", 
        "defaults": {
            "alpha": 1.0, 
            "binarize": 0.0, 
            "fit_prior": True
        }, 
        "help_text": "Naive Bayes classifier for multivariate Bernoulli models.  Like MultinomialNB, this classifier is suitable for discrete data. The  difference is that while MultinomialNB works with occurrence counts,  BernoulliNB is designed for binary/boolean features.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Additive (Laplace/Lidstone) smoothing parameter\n(0 for no smoothing).\n", 
                "name": "alpha", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.0, 
                "help_text": "Threshold for binarizing (mapping to booleans) of sample features.\nIf None, input is presumed to already consist of binary vectors.\n", 
                "name": "binarize", 
                "required": False, 
                "type": "float_or_none"
            }, 
            {
                "default": True, 
                "help_text": "Whether to learn class prior probabilities or not.\nIf False, a uniform prior will be used.\n", 
                "name": "fit_prior", 
                "type": "boolean"
            }, 
            {
                "help_text": "Prior probabilities of the classes. If specified the priors are not\nadjusted according to the data.\n\nAttributes\n----------", 
                "name": "class_prior", 
                "type": "list"
            }
        ]
    }, 
    "CCA": {
        "cls": "sklearn.cross_decomposition.cca_.CCA", 
        "defaults": {
            "copy": True, 
            "max_iter": 500, 
            "n_components": 2, 
            "scale": True, 
            "tol": 1e-06
        }, 
        "help_text": "CCA Canonical Correlation Analysis.  CCA inherits from PLS with mode=\"B\" and deflation_mode=\"canonical\".", 
        "parameters": [
            {
                "default": 2, 
                "help_text": "number of components to keep.\n", 
                "name": "n_components", 
                "type": "integer"
            }, 
            {
                "default": True, 
                "help_text": "whether to scale the data?\n", 
                "name": "scale", 
                "type": "boolean"
            }, 
            {
                "default": 500, 
                "help_text": "the maximum number of iterations of the NIPALS inner loop\n", 
                "name": "max_iter", 
                "type": "integer"
            }, 
            {
                "default": 1e-06, 
                "help_text": "the tolerance used in the iterative algorithm\n", 
                "name": "tol", 
                "type": "string"
            }, 
            {
                "default": True, 
                "help_text": "Whether the deflation be done on a copy. Let the default value\nto True unless you don't care about side effects\n\nAttributes\n----------", 
                "name": "copy", 
                "type": "boolean"
            }
        ]
    }, 
    "DummyRegressor": {
        "cls": "sklearn.dummy.DummyRegressor", 
        "defaults": {
            "strategy": "mean"
        }, 
        "help_text": "DummyRegressor is a regressor that makes predictions using  simple rules.  This regressor is useful as a simple baseline to compare with other  (real) regressors. Do not use it for real problems.", 
        "parameters": [
            {
                "default": "mean", 
                "help_text": "Strategy to use to generate predictions.\n", 
                "name": "strategy", 
                "type": "string"
            }, 
            {
                "help_text": "The explicit constant as predicted by the \"constant\" strategy. This\nparameter is useful only for the \"constant\" strategy.\n", 
                "name": "constant", 
                "type": "float_or_list"
            }, 
            {
                "name": "quantile", 
                "type": "float"
            }
        ]
    }, 
    "ElasticNet": {
        "cls": "sklearn.linear_model.coordinate_descent.ElasticNet", 
        "defaults": {
            "alpha": 1.0, 
            "copy_X": True, 
            "fit_intercept": True, 
            "l1_ratio": 0.5, 
            "max_iter": 1000, 
            "normalize": False, 
            "positive": False, 
            "precompute": False, 
            "selection": "cyclic", 
            "tol": 0.0001, 
            "warm_start": False
        }, 
        "help_text": "Linear regression with combined L1 and L2 priors as regularizer.  Minimizes the objective function::      1 / (2 * n_samples) * ||y - Xw||^2_2 +      + alpha * l1_ratio * ||w||_1      + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2  If you are interested in controlling the L1 and L2 penalty  separately, keep in mind that this is equivalent to::      a * L1 + b * L2  where::      alpha = a + b and l1_ratio = a / (a + b)  The parameter l1_ratio corresponds to alpha in the glmnet R package while  alpha corresponds to the lambda parameter in glmnet. Specifically, l1_ratio  = 1 is the lasso penalty. Currently, l1_ratio <= 0.01 is not reliable,  unless you supply your own sequence of alpha.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Constant that multiplies the penalty terms. Defaults to 1.0\nSee the notes for the exact mathematical meaning of this\nparameter.\n``alpha = 0`` is equivalent to an ordinary least square, solved\nby the :class:`LinearRegression` object. For numerical\nreasons, using ``alpha = 0`` with the Lasso object is not advised\nand you should prefer the LinearRegression object.\n", 
                "name": "alpha", 
                "type": "float"
            }, 
            {
                "default": 0.5, 
                "help_text": "The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For\n``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it\nis an L1 penalty. For ``0 < l1_ratio < 1``, the penalty is a\ncombination of L1 and L2.\n", 
                "name": "l1_ratio", 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Whether the intercept should be estimated or not. If ``False``, the\ndata is assumed to be already centered.\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If ``True``, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "default": False, 
                "help_text": "Whether to use a precomputed Gram matrix to speed up\ncalculations. If set to ``'auto'`` let us decide. The Gram\nmatrix can also be passed as argument. For sparse input\nthis option is always ``True`` to preserve sparsity.", 
                "name": "precompute", 
                "type": "list"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of iterations\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 0.0001, 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, reuse the solution of the previous call to fit as\ninitialization, otherwise, just erase the previous solution.\n", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, forces the coefficients to be positive.\n", 
                "name": "positive", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": "cyclic", 
                "help_text": "If set to 'random', a random coefficient is updated every iteration\nrather than looping over features sequentially by default. This\n(setting to 'random') often leads to significantly faster convergence\nespecially when tol is higher than 1e-4.\n", 
                "name": "selection", 
                "type": "string"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator that selects\na random feature to update. Useful only when selection is set to\n'random'.\n\nAttributes\n----------", 
                "name": "random_state", 
                "type": "integer"
            }
        ]
    }, 
    "ElasticNetCV": {
        "cls": "sklearn.linear_model.coordinate_descent.ElasticNetCV", 
        "defaults": {
            "copy_X": True, 
            "eps": 0.001, 
            "fit_intercept": True, 
            "l1_ratio": 0.5, 
            "max_iter": 1000, 
            "n_alphas": 100, 
            "n_jobs": 1, 
            "normalize": False, 
            "positive": False, 
            "precompute": "auto", 
            "selection": "cyclic", 
            "tol": 0.0001, 
            "verbose": 0
        }, 
        "help_text": "Elastic Net model with iterative fitting along a regularization path  The best model is selected by cross-validation.", 
        "parameters": [
            {
                "default": 0.5, 
                "help_text": "float between 0 and 1 passed to ElasticNet (scaling between\nl1 and l2 penalties). For ``l1_ratio = 0``\nthe penalty is an L2 penalty. For ``l1_ratio = 1`` it is an L1 penalty.\nFor ``0 < l1_ratio < 1``, the penalty is a combination of L1 and L2\nThis parameter can be a list, in which case the different\nvalues are tested by cross-validation and the one giving the best\nprediction score is used. Note that a good choice of list of\nvalues for l1_ratio is often to put more values close to 1\n(i.e. Lasso) and less close to 0 (i.e. Ridge), as in ``[.1, .5, .7,\n.9, .95, .99, 1]``\n", 
                "name": "l1_ratio", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.001, 
                "help_text": "Length of the path. ``eps=1e-3`` means that\n``alpha_min / alpha_max = 1e-3``.\n", 
                "name": "eps", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 100, 
                "help_text": "Number of alphas along the regularization path, used for each l1_ratio.\n", 
                "name": "n_alphas", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "List of alphas where to compute the models.\nIf None alphas are set automatically\n", 
                "name": "alphas", 
                "required": False, 
                "type": "list"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "default": "auto", 
                "help_text": "Whether to use a precomputed Gram matrix to speed up\ncalculations. If set to ``'auto'`` let us decide. The Gram\nmatrix can also be passed as argument.\n", 
                "name": "precompute", 
                "type": "list"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of iterations\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0001, 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "If an integer is passed, it is the number of fold (default 3).\nSpecific cross-validation objects can be passed, see the\n:mod:`sklearn.cross_validation` module for the list of possible\nobjects.\n", 
                "name": "cv", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "Amount of verbosity.\n", 
                "name": "verbose", 
                "type": "boolean_or_integer"
            }, 
            {
                "default": 1, 
                "help_text": "Number of CPUs to use during the cross validation. If ``-1``, use\nall the CPUs.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, forces the coefficients to be positive.\n", 
                "name": "positive", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": "cyclic", 
                "help_text": "If set to 'random', a random coefficient is updated every iteration\nrather than looping over features sequentially by default. This\n(setting to 'random') often leads to significantly faster convergence\nespecially when tol is higher than 1e-4.\n", 
                "name": "selection", 
                "type": "string"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator that selects\na random feature to update. Useful only when selection is set to\n'random'.\n", 
                "name": "random_state", 
                "type": "integer"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If ``True``, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, X will be copied; else, it may be overwritten.\n\nAttributes\n----------", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "EllipticEnvelope": {
        "cls": "sklearn.covariance.outlier_detection.EllipticEnvelope", 
        "defaults": {
            "assume_centered": False, 
            "contamination": 0.1, 
            "store_precision": True
        }, 
        "help_text": "An object for detecting outliers in a Gaussian distributed dataset.  Attributes  ----------  `contamination` : float, 0. < contamination < 0.5   The amount of contamination of the data set, i.e. the proportion of    outliers in the data set.  location_ : array-like, shape (n_features,)    Estimated robust location  covariance_ : array-like, shape (n_features, n_features)    Estimated robust covariance matrix  precision_ : array-like, shape (n_features, n_features)    Estimated pseudo inverse matrix.    (stored only if store_precision is True)  support_ : array-like, shape (n_samples,)    A mask of the observations that have been used to compute the    robust estimates of location and shape.", 
        "parameters": [
            {
                "default": True, 
                "help_text": "Specify if the estimated precision is stored.\n", 
                "name": "store_precision", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If True, the support of robust location and covariance estimates\nis computed, and a covariance estimate is recomputed from it,\nwithout centering the data.\nUseful to work with data whose mean is significantly equal to\nzero but is not exactly zero.\nIf False, the robust location and covariance are directly computed\nwith the FastMCD algorithm without additional treatment.\n", 
                "name": "assume_centered", 
                "type": "boolean"
            }, 
            {
                "help_text": "The proportion of points to be included in the support of the raw\nMCD estimate. Default is ``None``, which implies that the minimum\nvalue of support_fraction will be used within the algorithm:\n`[n_sample + n_features + 1] / 2`.\n", 
                "name": "support_fraction", 
                "type": "float"
            }, 
            {
                "default": 0.1, 
                "name": "contamination", 
                "type": "float"
            }
        ]
    }, 
    "ExtraTreesRegressor": {
        "cls": "sklearn.ensemble.forest.ExtraTreesRegressor", 
        "defaults": {
            "bootstrap": False, 
            "criterion": "mse", 
            "max_features": "auto", 
            "min_samples_leaf": 1, 
            "min_samples_split": 2, 
            "min_weight_fraction_leaf": 0.0, 
            "n_estimators": 10, 
            "n_jobs": 1, 
            "oob_score": False, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "An extra-trees regressor.  This class implements a meta estimator that fits a number of  randomized decision trees (a.k.a. extra-trees) on various sub-samples  of the dataset and use averaging to improve the predictive accuracy  and control over-fitting.", 
        "parameters": [
            {
                "default": 10, 
                "help_text": "The number of trees in the forest.\n", 
                "name": "n_estimators", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "mse", 
                "help_text": "The function to measure the quality of a split. The only supported\ncriterion is \"mse\" for the mean squared error.", 
                "name": "criterion", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": "auto", 
                "help_text": "The number of features to consider when looking for the best split:\n\n- If int, then consider `max_features` features at each split.\n- If float, then `max_features` is a percentage and\n`int(max_features * n_features)` features are considered at each\nsplit.\n- If \"auto\", then `max_features=n_features`.\n- If \"sqrt\", then `max_features=sqrt(n_features)`.\n- If \"log2\", then `max_features=log2(n_features)`.\n- If None, then `max_features=n_features`.\n", 
                "name": "max_features", 
                "required": False, 
                "type": "int_float_string_none"
            }, 
            {
                "help_text": "The maximum depth of the tree. If None, then nodes are expanded until\nall leaves are pure or until all leaves contain less than\nmin_samples_split samples.\nIgnored if ``max_leaf_nodes`` is not None.", 
                "name": "max_depth", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2, 
                "help_text": "The minimum number of samples required to split an internal node.", 
                "name": "min_samples_split", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "The minimum number of samples in newly created leaves. A split is\ndiscarded if after the split, one of the leaves would contain less then\n``min_samples_leaf`` samples.", 
                "name": "min_samples_leaf", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "The minimum weighted fraction of the input samples required to be at a\nleaf node.", 
                "name": "min_weight_fraction_leaf", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "Grow trees with ``max_leaf_nodes`` in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.\nIf not None then ``max_depth`` will be ignored.", 
                "name": "max_leaf_nodes", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Whether bootstrap samples are used when building trees.", 
                "name": "bootstrap", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Whether to use out-of-bag samples to estimate\nthe generalization error.\n", 
                "name": "oob_score", 
                "type": "boolean"
            }, 
            {
                "default": 1, 
                "help_text": "The number of jobs to run in parallel for both `fit` and `predict`.\nIf -1, then the number of jobs is set to the number of cores.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "If int, random_state is the seed used by the random number generator;\nIf RandomState instance, random_state is the random number generator;\nIf None, the random number generator is the RandomState instance used\nby `np.random`.\n", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "Controls the verbosity of the tree building process.\n", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, reuse the solution of the previous call to fit\nand add more estimators to the ensemble, otherwise, just fit a whole\nnew forest.\n\nAttributes\n----------", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "GaussianProcess": {
        "cls": "sklearn.gaussian_process.gaussian_process.GaussianProcess", 
        "defaults": {
            "corr": "squared_exponential", 
            "normalize": True, 
            "nugget": 2.2204460492503131e-15, 
            "optimizer": "fmin_cobyla", 
            "random_start": 1, 
            "regr": "constant", 
            "storage_mode": "full", 
            "theta0": 0.1, 
            "verbose": False
        }, 
        "help_text": "The Gaussian Process model class.", 
        "parameters": [
            {
                "default": "constant", 
                "help_text": "A regression function returning an array of outputs of the linear\nregression functional basis. The number of observations n_samples\nshould be greater than the size p of this basis.\nDefault assumes a simple constant regression trend.\nAvailable built-in regression models are::\n\n'constant', 'linear', 'quadratic'\n", 
                "name": "regr", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": "squared_exponential", 
                "help_text": "A stationary autocorrelation function returning the autocorrelation\nbetween two points x and x'.\nDefault assumes a squared-exponential autocorrelation model.\nBuilt-in correlation models are::\n\n'absolute_exponential', 'squared_exponential',\n'generalized_exponential', 'cubic', 'linear'\n", 
                "name": "corr", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "The regression weight vector to perform Ordinary Kriging (OK).\nDefault assumes Universal Kriging (UK) so that the vector beta of\nregression weights is estimated using the maximum likelihood\nprinciple.\n", 
                "name": "beta0", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": "full", 
                "help_text": "A string specifying whether the Cholesky decomposition of the\ncorrelation matrix should be stored in the class (storage_mode =\n'full') or not (storage_mode = 'light').\nDefault assumes storage_mode = 'full', so that the\nCholesky decomposition of the correlation matrix is stored.\nThis might be a useful parameter when one is not interested in the\nMSE and only plan to estimate the BLUP, for which the correlation\nmatrix is not required.\n", 
                "name": "storage_mode", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": False, 
                "help_text": "A boolean specifying the verbose level.\nDefault is verbose = False.\n", 
                "name": "verbose", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 0.1, 
                "help_text": "An array with shape (n_features, ) or (1, ).\nThe parameters in the autocorrelation model.\nIf thetaL and thetaU are also specified, theta0 is considered as\nthe starting point for the maximum likelihood estimation of the\nbest set of parameters.\nDefault assumes isotropic autocorrelation model with theta0 = 1e-1.\n", 
                "name": "theta0", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "An array with shape matching theta0's.\nLower bound on the autocorrelation parameters for maximum\nlikelihood estimation.\nDefault is None, so that it skips maximum likelihood estimation and\nit uses theta0.\n", 
                "name": "thetaL", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "An array with shape matching theta0's.\nUpper bound on the autocorrelation parameters for maximum\nlikelihood estimation.\nDefault is None, so that it skips maximum likelihood estimation and\nit uses theta0.\n", 
                "name": "thetaU", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Input X and observations y are centered and reduced wrt\nmeans and standard deviations estimated from the n_samples\nobservations provided.\nDefault is normalize = True so that data is normalized to ease\nmaximum likelihood estimation.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 2.2204460492503131e-15, 
                "help_text": "Introduce a nugget effect to allow smooth predictions from noisy\ndata. If nugget is an ndarray, it must be the same length as the\nnumber of data points used for the fit.\nThe nugget is added to the diagonal of the assumed training covariance;\nin this way it acts as a Tikhonov regularization in the problem. In\nthe special case of the squared exponential correlation function, the\nnugget mathematically represents the variance of the input values.\nDefault assumes a nugget close to machine precision for the sake of\nrobustness (nugget = 10. * MACHINE_EPSILON).\n", 
                "name": "nugget", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": "fmin_cobyla", 
                "help_text": "A string specifying the optimization algorithm to be used.\nDefault uses 'fmin_cobyla' algorithm from scipy.optimize.\nAvailable optimizers are::\n\n'fmin_cobyla', 'Welch'\n\n'Welch' optimizer is dued to Welch et al., see reference [WBSWM1992]_.\nIt consists in iterating over several one-dimensional optimizations\ninstead of running one single multi-dimensional optimization.\n", 
                "name": "optimizer", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 1, 
                "help_text": "The number of times the Maximum Likelihood Estimation should be\nperformed from a random starting point.\nThe first MLE always uses the specified starting point (theta0),\nthe next starting points are picked at random according to an\nexponential distribution (log-uniform on [thetaL, thetaU]).\nDefault does not use random starting point (random_start = 1).\n", 
                "name": "random_start", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "The generator used to shuffle the sequence of coordinates of theta in\nthe Welch optimizer. If an integer is given, it fixes the seed.\nDefaults to the global numpy random number generator.\n\n\nAttributes\n----------", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "GradientBoostingRegressor": {
        "cls": "sklearn.ensemble.gradient_boosting.GradientBoostingRegressor", 
        "defaults": {
            "alpha": 0.9, 
            "learning_rate": 0.1, 
            "loss": "ls", 
            "max_depth": 3, 
            "min_samples_leaf": 1, 
            "min_samples_split": 2, 
            "min_weight_fraction_leaf": 0.0, 
            "n_estimators": 100, 
            "subsample": 1.0, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "Gradient Boosting for regression.  GB builds an additive model in a forward stage-wise fashion;  it allows for the optimization of arbitrary differentiable loss functions.  In each stage a regression tree is fit on the negative gradient of the  given loss function.", 
        "parameters": [
            {
                "choices": [
                    "ls", 
                    "lad", 
                    "huber", 
                    "quantile"
                ], 
                "default": "ls", 
                "help_text": "loss function to be optimized. 'ls' refers to least squares\nregression. 'lad' (least absolute deviation) is a highly robust\nloss function solely based on order information of the input\nvariables. 'huber' is a combination of the two. 'quantile'\nallows quantile regression (use `alpha` to specify the quantile).\n", 
                "name": "loss", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 0.1, 
                "help_text": "learning rate shrinks the contribution of each tree by `learning_rate`.\nThere is a trade-off between learning_rate and n_estimators.\n", 
                "name": "learning_rate", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 100, 
                "help_text": "The number of boosting stages to perform. Gradient boosting\nis fairly robust to over-fitting so a large number usually\nresults in better performance.\n", 
                "name": "n_estimators", 
                "type": "integer"
            }, 
            {
                "default": 3, 
                "help_text": "maximum depth of the individual regression estimators. The maximum\ndepth limits the number of nodes in the tree. Tune this parameter\nfor best performance; the best value depends on the interaction\nof the input variables.\nIgnored if ``max_leaf_nodes`` is not None.\n", 
                "name": "max_depth", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2, 
                "help_text": "The minimum number of samples required to split an internal node.\n", 
                "name": "min_samples_split", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "The minimum number of samples required to be at a leaf node.\n", 
                "name": "min_samples_leaf", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "The minimum weighted fraction of the input samples required to be at a\nleaf node.\n", 
                "name": "min_weight_fraction_leaf", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 1.0, 
                "help_text": "The fraction of samples to be used for fitting the individual base\nlearners. If smaller than 1.0 this results in Stochastic Gradient\nBoosting. `subsample` interacts with the parameter `n_estimators`.\nChoosing `subsample < 1.0` leads to a reduction of variance\nand an increase in bias.\n", 
                "name": "subsample", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "The number of features to consider when looking for the best split:\n- If int, then consider `max_features` features at each split.\n- If float, then `max_features` is a percentage and\n`int(max_features * n_features)` features are considered at each\nsplit.\n- If \"auto\", then `max_features=n_features`.\n- If \"sqrt\", then `max_features=sqrt(n_features)`.\n- If \"log2\", then `max_features=log2(n_features)`.\n- If None, then `max_features=n_features`.\n\nChoosing `max_features < n_features` leads to a reduction of variance\nand an increase in bias.\n", 
                "name": "max_features", 
                "required": False, 
                "type": "int_float_string_none"
            }, 
            {
                "help_text": "Grow trees with ``max_leaf_nodes`` in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.\n", 
                "name": "max_leaf_nodes", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.9, 
                "help_text": "The alpha-quantile of the huber loss function and the quantile\nloss function. Only if ``loss='huber'`` or ``loss='quantile'``.\n", 
                "name": "alpha", 
                "type": "float"
            }, 
            {
                "help_text": "An estimator object that is used to compute the initial\npredictions. ``init`` has to provide ``fit`` and ``predict``.\nIf None it uses ``loss.init_estimator``.\n", 
                "name": "init", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 0, 
                "help_text": "Enable verbose output. If 1 then it prints progress and performance\nonce in a while (the more trees the lower the frequency). If greater\nthan 1 then it prints progress and performance for every tree.\n", 
                "name": "verbose", 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, reuse the solution of the previous call to fit\nand add more estimators to the ensemble, otherwise, just erase the\nprevious solution.\n\n\nAttributes\n----------", 
                "name": "warm_start", 
                "type": "boolean"
            }
        ]
    }, 
    "IsotonicRegression": {
        "cls": "sklearn.isotonic.IsotonicRegression", 
        "defaults": {
            "increasing": True, 
            "out_of_bounds": "nan"
        }, 
        "help_text": "Isotonic regression model.  The isotonic regression optimization problem is defined by::    min sum w_i (y[i] - y_[i]) ** 2    subject to y_[i] <= y_[j] whenever X[i] <= X[j]    and min(y_) = y_min, max(y_) = y_max  where:    - ``y[i]`` are inputs (real numbers)    - ``y_[i]`` are fitted    - ``X`` specifies the order.     If ``X`` is non-decreasing then ``y_`` is non-decreasing.    - ``w[i]`` are optional strictly positive weights (default to 1.0)", 
        "parameters": [
            {
                "help_text": "If not None, set the lowest value of the fit to y_min.\n", 
                "name": "y_min", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "If not None, set the highest value of the fit to y_max.\n", 
                "name": "y_max", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": True, 
                "help_text": "If boolean, whether or not to fit the isotonic regression with y\nincreasing or decreasing.\n\nThe string value \"auto\" determines whether y should\nincrease or decrease based on the Spearman correlation estimate's\nsign.\n", 
                "name": "increasing", 
                "required": False, 
                "type": "boolean_or_string"
            }, 
            {
                "choices": [
                    "nan"
                ], 
                "default": "nan", 
                "help_text": "The ``out_of_bounds`` parameter handles how x-values outside of the\ntraining domain are handled. When set to \"nan\", predicted y-values\nwill be NaN. When set to \"clip\", predicted y-values will be\nset to the value corresponding to the nearest train interval endpoint.\nWhen set to \"raise\", allow ``interp1d`` to throw ValueError.\n\n\nAttributes\n----------", 
                "name": "out_of_bounds", 
                "required": False, 
                "type": "string"
            }
        ]
    }, 
    "KNeighborsClassifier": {
        "cls": "sklearn.neighbors.classification.KNeighborsClassifier", 
        "defaults": {
            "algorithm": "auto", 
            "leaf_size": 30, 
            "metric": "minkowski", 
            "n_neighbors": 5, 
            "p": 2, 
            "weights": "uniform"
        }, 
        "help_text": "Classifier implementing the k-nearest neighbors vote.", 
        "parameters": [
            {
                "default": 5, 
                "help_text": "Number of neighbors to use by default for :meth:`k_neighbors` queries.\n", 
                "name": "n_neighbors", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "uniform", 
                "help_text": "weight function used in prediction. Possible values:\n", 
                "name": "weights", 
                "type": "string"
            }, 
            {
                "choices": [
                    "auto", 
                    "ball_tree", 
                    "kd_tree", 
                    "brute"
                ], 
                "default": "auto", 
                "help_text": "Algorithm used to compute the nearest neighbors:\n\n- 'ball_tree' will use :class:`BallTree`\n- 'kd_tree' will use :class:`KDTree`\n- 'brute' will use a brute-force search.\n- 'auto' will attempt to decide the most appropriate algorithm\nbased on the values passed to :meth:`fit` method.\n", 
                "name": "algorithm", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 30, 
                "help_text": "Leaf size passed to BallTree or KDTree. This can affect the\nspeed of the construction and query, as well as the memory\nrequired to store the tree. The optimal value depends on the\nnature of the problem.\n", 
                "name": "leaf_size", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "minkowski", 
                "help_text": "the distance metric to use for the tree. The default metric is\nminkowski, and with p=2 is equivalent to the standard Euclidean\nmetric. See the documentation of the DistanceMetric class for a\nlist of available metrics.\n", 
                "name": "metric", 
                "type": "string"
            }, 
            {
                "default": 2, 
                "help_text": "Power parameter for the Minkowski metric. When p = 1, this is\nequivalent to using manhattan_distance (l1), and euclidean_distance\n(l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.\n", 
                "name": "p", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "additional keyword arguments for the metric function.\n\nExamples\n--------\n>>> X = [[0], [1], [2], [3]]\n>>> y = [0, 0, 1, 1]\n>>> from sklearn.neighbors import KNeighborsClassifier\n>>> neigh = KNeighborsClassifier(n_neighbors=3)", 
                "name": "metric_params", 
                "required": False, 
                "type": "dict"
            }
        ]
    }, 
    "KNeighborsRegressor": {
        "cls": "sklearn.neighbors.regression.KNeighborsRegressor", 
        "defaults": {
            "algorithm": "auto", 
            "leaf_size": 30, 
            "metric": "minkowski", 
            "n_neighbors": 5, 
            "p": 2, 
            "weights": "uniform"
        }, 
        "help_text": "Regression based on k-nearest neighbors.  The target is predicted by local interpolation of the targets  associated of the nearest neighbors in the training set.", 
        "parameters": [
            {
                "default": 5, 
                "help_text": "Number of neighbors to use by default for :meth:`k_neighbors` queries.\n", 
                "name": "n_neighbors", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "uniform", 
                "help_text": "weight function used in prediction. Possible values:\n", 
                "name": "weights", 
                "type": "string"
            }, 
            {
                "choices": [
                    "auto", 
                    "ball_tree", 
                    "kd_tree", 
                    "brute"
                ], 
                "default": "auto", 
                "help_text": "Algorithm used to compute the nearest neighbors:\n\n- 'ball_tree' will use :class:`BallTree`\n- 'kd_tree' will use :class:`KDtree`\n- 'brute' will use a brute-force search.\n- 'auto' will attempt to decide the most appropriate algorithm\nbased on the values passed to :meth:`fit` method.\n", 
                "name": "algorithm", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 30, 
                "help_text": "Leaf size passed to BallTree or KDTree. This can affect the\nspeed of the construction and query, as well as the memory\nrequired to store the tree. The optimal value depends on the\nnature of the problem.\n", 
                "name": "leaf_size", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "minkowski", 
                "help_text": "the distance metric to use for the tree. The default metric is\nminkowski, and with p=2 is equivalent to the standard Euclidean\nmetric. See the documentation of the DistanceMetric class for a\nlist of available metrics.\n", 
                "name": "metric", 
                "type": "string"
            }, 
            {
                "default": 2, 
                "help_text": "Power parameter for the Minkowski metric. When p = 1, this is\nequivalent to using manhattan_distance (l1), and euclidean_distance\n(l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.\n", 
                "name": "p", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "additional keyword arguments for the metric function.\n\nExamples\n--------\n>>> X = [[0], [1], [2], [3]]\n>>> y = [0, 0, 1, 1]\n>>> from sklearn.neighbors import KNeighborsRegressor\n>>> neigh = KNeighborsRegressor(n_neighbors=2)", 
                "name": "metric_params", 
                "required": False, 
                "type": "dict"
            }
        ]
    }, 
    "KernelRidge": {
        "cls": "sklearn.kernel_ridge.KernelRidge", 
        "defaults": {
            "alpha": 1, 
            "coef0": 1, 
            "degree": 3, 
            "kernel": "linear"
        }, 
        "help_text": "Kernel ridge regression.  Kernel ridge regression (KRR) combines ridge regression (linear least  squares with l2-norm regularization) with the kernel trick. It thus  learns a linear function in the space induced by the respective kernel and  the data. For non-linear kernels, this corresponds to a non-linear  function in the original space.  The form of the model learned by KRR is identical to support vector  regression (SVR). However, different loss functions are used: KRR uses  squared error loss while support vector regression uses epsilon-insensitive  loss, both combined with l2 regularization. In contrast to SVR, fitting a  KRR model can be done in closed-form and is typically faster for  medium-sized datasets. On the other hand, the learned model is non-sparse  and thus slower than SVR, which learns a sparse model for epsilon > 0, at  prediction-time.  This estimator has built-in support for multi-variate regression  (i.e., when y is a 2d-array of shape [n_samples, n_targets]).", 
        "parameters": [
            {
                "default": 1, 
                "help_text": "Small positive values of alpha improve the conditioning of the problem\nand reduce the variance of the estimates. Alpha corresponds to\n``(2*C)^-1`` in other linear models such as LogisticRegression or\nLinearSVC. If an array is passed, penalties are assumed to be specific\nto the targets. Hence they must correspond in number.\n", 
                "name": "alpha", 
                "type": "list"
            }, 
            {
                "default": "linear", 
                "help_text": "Kernel mapping used internally. A callable should accept two arguments\nand the keyword arguments passed to this object as kernel_params, and\nshould return a floating point number.\n", 
                "name": "kernel", 
                "type": "string"
            }, 
            {
                "help_text": "Gamma parameter for the RBF, polynomial, exponential chi2 and\nsigmoid kernels. Interpretation of the default value is left to\nthe kernel; see the documentation for sklearn.metrics.pairwise.\nIgnored by other kernels.\n", 
                "name": "gamma", 
                "type": "float"
            }, 
            {
                "default": 3, 
                "help_text": "Degree of the polynomial kernel. Ignored by other kernels.\n", 
                "name": "degree", 
                "type": "float"
            }, 
            {
                "default": 1, 
                "help_text": "Zero coefficient for polynomial and sigmoid kernels.\nIgnored by other kernels.\n", 
                "name": "coef0", 
                "type": "float"
            }, 
            {
                "help_text": "Additional parameters (keyword arguments) for kernel function passed\nas callable object.\n\nAttributes\n----------", 
                "name": "kernel_params", 
                "required": False, 
                "type": "string"
            }
        ]
    }, 
    "LDA": {
        "cls": "sklearn.lda.LDA", 
        "defaults": {
            "solver": "svd", 
            "store_covariance": False, 
            "tol": 0.0001
        }, 
        "help_text": "Linear Discriminant Analysis (LDA).  A classifier with a linear decision boundary, generated by fitting class  conditional densities to the data and using Bayes' rule.  The model fits a Gaussian density to each class, assuming that all classes  share the same covariance matrix.  The fitted model can also be used to reduce the dimensionality of the input  by projecting it to the most discriminative directions.", 
        "parameters": [
            {
                "default": "svd", 
                "help_text": "Solver to use, possible values:", 
                "name": "solver", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "Shrinkage parameter, possible values:", 
                "name": "shrinkage", 
                "required": False, 
                "type": "string_or_float"
            }, 
            {
                "help_text": "Class priors.\n", 
                "name": "priors", 
                "required": False, 
                "type": "list"
            }, 
            {
                "help_text": "Number of components (< n_classes - 1) for dimensionality reduction.\n", 
                "name": "n_components", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Additionally compute class covariance matrix (default False).\n", 
                "name": "store_covariance", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 0.0001, 
                "help_text": "Threshold used for rank estimation in SVD solver.\n\nAttributes\n----------", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }
        ]
    }, 
    "LabelPropagation": {
        "cls": "sklearn.semi_supervised.label_propagation.LabelPropagation", 
        "defaults": {
            "alpha": 1, 
            "gamma": 20, 
            "kernel": "rbf", 
            "max_iter": 30, 
            "n_neighbors": 7, 
            "tol": 0.001
        }, 
        "help_text": "Label Propagation classifier", 
        "parameters": [
            {
                "choices": [
                    "knn", 
                    "rbf"
                ], 
                "default": "rbf", 
                "help_text": "String identifier for kernel function to use.\nOnly 'rbf' and 'knn' kernels are currently supported..", 
                "name": "kernel", 
                "type": "string"
            }, 
            {
                "default": 20, 
                "help_text": "Parameter for rbf kernel", 
                "name": "gamma", 
                "type": "float"
            }, 
            {
                "default": 7, 
                "help_text": "Parameter for knn kernel", 
                "name": "n_neighbors", 
                "type": "positive_integer"
            }, 
            {
                "default": 1, 
                "help_text": "Clamping factor", 
                "name": "alpha", 
                "type": "float"
            }, 
            {
                "default": 30, 
                "help_text": "Change maximum number of iterations allowed", 
                "name": "max_iter", 
                "type": "float"
            }, 
            {
                "default": 0.001, 
                "name": "tol", 
                "type": "float"
            }
        ]
    }, 
    "LabelSpreading": {
        "cls": "sklearn.semi_supervised.label_propagation.LabelSpreading", 
        "defaults": {
            "alpha": 0.2, 
            "gamma": 20, 
            "kernel": "rbf", 
            "max_iter": 30, 
            "n_neighbors": 7, 
            "tol": 0.001
        }, 
        "help_text": "LabelSpreading model for semi-supervised learning  This model is similar to the basic Label Propgation algorithm,  but uses affinity matrix based on the normalized graph Laplacian  and soft clamping across the labels.", 
        "parameters": [
            {
                "choices": [
                    "knn", 
                    "rbf"
                ], 
                "default": "rbf", 
                "help_text": "String identifier for kernel function to use.\nOnly 'rbf' and 'knn' kernels are currently supported.", 
                "name": "kernel", 
                "type": "string"
            }, 
            {
                "default": 20, 
                "help_text": "parameter for rbf kernel", 
                "name": "gamma", 
                "type": "float"
            }, 
            {
                "default": 7, 
                "help_text": "parameter for knn kernel", 
                "name": "n_neighbors", 
                "type": "positive_integer"
            }, 
            {
                "default": 0.2, 
                "help_text": "clamping factor", 
                "name": "alpha", 
                "type": "float"
            }, 
            {
                "default": 30, 
                "help_text": "maximum number of iterations allowed", 
                "name": "max_iter", 
                "type": "float"
            }, 
            {
                "default": 0.001, 
                "name": "tol", 
                "type": "float"
            }
        ]
    }, 
    "Lars": {
        "cls": "sklearn.linear_model.least_angle.Lars", 
        "defaults": {
            "copy_X": True, 
            "eps": 2.2204460492503131e-16, 
            "fit_intercept": True, 
            "fit_path": True, 
            "n_nonzero_coefs": 500, 
            "normalize": True, 
            "precompute": "auto", 
            "verbose": False
        }, 
        "help_text": "Least Angle Regression model a.k.a. LAR", 
        "parameters": [
            {
                "default": 500, 
                "help_text": "Target number of non-zero coefficients. Use ``np.inf`` for no limit.\n", 
                "name": "n_nonzero_coefs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": True, 
                "help_text": "Whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Sets the verbosity amount\n", 
                "name": "verbose", 
                "required": False, 
                "type": "boolean_or_integer"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "default": "auto", 
                "help_text": "Whether to use a precomputed Gram matrix to speed up\ncalculations. If set to ``'auto'`` let us decide. The Gram\nmatrix can also be passed as argument.\n", 
                "name": "precompute", 
                "type": "list"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 2.2204460492503131e-16, 
                "help_text": "The machine-precision regularization in the computation of the\nCholesky diagonal factors. Increase this for very ill-conditioned\nsystems. Unlike the ``tol`` parameter in some iterative\noptimization-based algorithms, this parameter does not control\nthe tolerance of the optimization.\n", 
                "name": "eps", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "If True the full path is stored in the ``coef_path_`` attribute.\nIf you compute the solution for a large problem or many targets,\nsetting ``fit_path`` to ``False`` will lead to a speedup, especially\nwith a small alpha.\n\nAttributes\n----------", 
                "name": "fit_path", 
                "type": "boolean"
            }
        ]
    }, 
    "LarsCV": {
        "cls": "sklearn.linear_model.least_angle.LarsCV", 
        "defaults": {
            "copy_X": True, 
            "eps": 2.2204460492503131e-16, 
            "fit_intercept": True, 
            "max_iter": 500, 
            "max_n_alphas": 1000, 
            "n_jobs": 1, 
            "normalize": True, 
            "precompute": "auto", 
            "verbose": False
        }, 
        "help_text": "Cross-validated Least Angle Regression model", 
        "parameters": [
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Sets the verbosity amount\n", 
                "name": "verbose", 
                "required": False, 
                "type": "boolean_or_integer"
            }, 
            {
                "default": True, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "default": "auto", 
                "help_text": "Whether to use a precomputed Gram matrix to speed up\ncalculations. If set to ``'auto'`` let us decide. The Gram\nmatrix can also be passed as argument.\n", 
                "name": "precompute", 
                "type": "list"
            }, 
            {
                "default": 500, 
                "help_text": "Maximum number of iterations to perform.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "see :mod:`sklearn.cross_validation`. If ``None`` is passed, default to\na 5-fold strategy\n", 
                "name": "cv", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of points on the path used to compute the\nresiduals in the cross-validation\n", 
                "name": "max_n_alphas", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "Number of CPUs to use during the cross validation. If ``-1``, use\nall the CPUs\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2.2204460492503131e-16, 
                "help_text": "The machine-precision regularization in the computation of the\nCholesky diagonal factors. Increase this for very ill-conditioned\nsystems.\n\n\nAttributes\n----------", 
                "name": "eps", 
                "required": False, 
                "type": "float"
            }
        ]
    }, 
    "Lasso": {
        "cls": "sklearn.linear_model.coordinate_descent.Lasso", 
        "defaults": {
            "alpha": 1.0, 
            "copy_X": True, 
            "fit_intercept": True, 
            "max_iter": 1000, 
            "normalize": False, 
            "positive": False, 
            "precompute": False, 
            "selection": "cyclic", 
            "tol": 0.0001, 
            "warm_start": False
        }, 
        "help_text": "Linear Model trained with L1 prior as regularizer (aka the Lasso)  The optimization objective for Lasso is::    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1  Technically the Lasso model is optimizing the same objective function as  the Elastic Net with ``l1_ratio=1.0`` (no L2 penalty).", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Constant that multiplies the L1 term. Defaults to 1.0.\n``alpha = 0`` is equivalent to an ordinary least square, solved\nby the :class:`LinearRegression` object. For numerical\nreasons, using ``alpha = 0`` is with the Lasso object is not advised\nand you should prefer the LinearRegression object.\n", 
                "name": "alpha", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If ``True``, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "default": False, 
                "help_text": "Whether to use a precomputed Gram matrix to speed up\ncalculations. If set to ``'auto'`` let us decide. The Gram\nmatrix can also be passed as argument. For sparse input\nthis option is always ``True`` to preserve sparsity.", 
                "name": "precompute", 
                "type": "list"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of iterations\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0001, 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "When set to True, reuse the solution of the previous call to fit as\ninitialization, otherwise, just erase the previous solution.\n", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, forces the coefficients to be positive.\n", 
                "name": "positive", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": "cyclic", 
                "help_text": "If set to 'random', a random coefficient is updated every iteration\nrather than looping over features sequentially by default. This\n(setting to 'random') often leads to significantly faster convergence\nespecially when tol is higher than 1e-4.\n", 
                "name": "selection", 
                "type": "string"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator that selects\na random feature to update. Useful only when selection is set to\n'random'.\n\nAttributes\n----------", 
                "name": "random_state", 
                "type": "integer"
            }
        ]
    }, 
    "LassoCV": {
        "cls": "sklearn.linear_model.coordinate_descent.LassoCV", 
        "defaults": {
            "copy_X": True, 
            "eps": 0.001, 
            "fit_intercept": True, 
            "max_iter": 1000, 
            "n_alphas": 100, 
            "n_jobs": 1, 
            "normalize": False, 
            "positive": False, 
            "precompute": "auto", 
            "selection": "cyclic", 
            "tol": 0.0001, 
            "verbose": False
        }, 
        "help_text": "Lasso linear model with iterative fitting along a regularization path  The best model is selected by cross-validation.  The optimization objective for Lasso is::    (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1", 
        "parameters": [
            {
                "default": 0.001, 
                "help_text": "Length of the path. ``eps=1e-3`` means that\n``alpha_min / alpha_max = 1e-3``.\n", 
                "name": "eps", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 100, 
                "help_text": "Number of alphas along the regularization path\n", 
                "name": "n_alphas", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "List of alphas where to compute the models.\nIf ``None`` alphas are set automatically\n", 
                "name": "alphas", 
                "required": False, 
                "type": "list"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "default": "auto", 
                "help_text": "Whether to use a precomputed Gram matrix to speed up\ncalculations. If set to ``'auto'`` let us decide. The Gram\nmatrix can also be passed as argument.\n", 
                "name": "precompute", 
                "type": "list"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of iterations\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0001, 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "If an integer is passed, it is the number of fold (default 3).\nSpecific cross-validation objects can be passed, see the\n:mod:`sklearn.cross_validation` module for the list of possible\nobjects.\n", 
                "name": "cv", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Amount of verbosity.\n", 
                "name": "verbose", 
                "type": "boolean_or_integer"
            }, 
            {
                "default": 1, 
                "help_text": "Number of CPUs to use during the cross validation. If ``-1``, use\nall the CPUs.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "If positive, restrict regression coefficients to be positive\n", 
                "name": "positive", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": "cyclic", 
                "help_text": "If set to 'random', a random coefficient is updated every iteration\nrather than looping over features sequentially by default. This\n(setting to 'random') often leads to significantly faster convergence\nespecially when tol is higher than 1e-4.\n", 
                "name": "selection", 
                "type": "string"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator that selects\na random feature to update. Useful only when selection is set to\n'random'.\n", 
                "name": "random_state", 
                "type": "integer"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If ``True``, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, X will be copied; else, it may be overwritten.\n\nAttributes\n----------", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "LassoLars": {
        "cls": "sklearn.linear_model.least_angle.LassoLars", 
        "defaults": {
            "alpha": 1.0, 
            "copy_X": True, 
            "eps": 2.2204460492503131e-16, 
            "fit_intercept": True, 
            "fit_path": True, 
            "max_iter": 500, 
            "normalize": True, 
            "precompute": "auto", 
            "verbose": False
        }, 
        "help_text": "Lasso model fit with Least Angle Regression a.k.a. Lars  It is a Linear Model trained with an L1 prior as regularizer.  The optimization objective for Lasso is::  (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Constant that multiplies the penalty term. Defaults to 1.0.\n``alpha = 0`` is equivalent to an ordinary least square, solved\nby :class:`LinearRegression`. For numerical reasons, using\n``alpha = 0`` with the LassoLars object is not advised and you\nshould prefer the LinearRegression object.\n", 
                "name": "alpha", 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Sets the verbosity amount\n", 
                "name": "verbose", 
                "required": False, 
                "type": "boolean_or_integer"
            }, 
            {
                "default": True, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If True, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "default": "auto", 
                "help_text": "Whether to use a precomputed Gram matrix to speed up\ncalculations. If set to ``'auto'`` let us decide. The Gram\nmatrix can also be passed as argument.\n", 
                "name": "precompute", 
                "type": "list"
            }, 
            {
                "default": 500, 
                "help_text": "Maximum number of iterations to perform.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2.2204460492503131e-16, 
                "help_text": "The machine-precision regularization in the computation of the\nCholesky diagonal factors. Increase this for very ill-conditioned\nsystems. Unlike the ``tol`` parameter in some iterative\noptimization-based algorithms, this parameter does not control\nthe tolerance of the optimization.\n", 
                "name": "eps", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "If ``True`` the full path is stored in the ``coef_path_`` attribute.\nIf you compute the solution for a large problem or many targets,\nsetting ``fit_path`` to ``False`` will lead to a speedup, especially\nwith a small alpha.\n\nAttributes\n----------", 
                "name": "fit_path", 
                "type": "boolean"
            }
        ]
    }, 
    "LassoLarsCV": {
        "cls": "sklearn.linear_model.least_angle.LassoLarsCV", 
        "defaults": {
            "copy_X": True, 
            "eps": 2.2204460492503131e-16, 
            "fit_intercept": True, 
            "max_iter": 500, 
            "max_n_alphas": 1000, 
            "n_jobs": 1, 
            "normalize": True, 
            "precompute": "auto", 
            "verbose": False
        }, 
        "help_text": "Cross-validated Lasso, using the LARS algorithm  The optimization objective for Lasso is::  (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1", 
        "parameters": [
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Sets the verbosity amount\n", 
                "name": "verbose", 
                "required": False, 
                "type": "boolean_or_integer"
            }, 
            {
                "default": True, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "default": "auto", 
                "help_text": "Whether to use a precomputed Gram matrix to speed up\ncalculations. If set to ``'auto'`` let us decide. The Gram\nmatrix can also be passed as argument.\n", 
                "name": "precompute", 
                "type": "list"
            }, 
            {
                "default": 500, 
                "help_text": "Maximum number of iterations to perform.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "see sklearn.cross_validation module. If None is passed, default to\na 5-fold strategy\n", 
                "name": "cv", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of points on the path used to compute the\nresiduals in the cross-validation\n", 
                "name": "max_n_alphas", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "Number of CPUs to use during the cross validation. If ``-1``, use\nall the CPUs\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2.2204460492503131e-16, 
                "help_text": "The machine-precision regularization in the computation of the\nCholesky diagonal factors. Increase this for very ill-conditioned\nsystems.\n", 
                "name": "eps", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "If True, X will be copied; else, it may be overwritten.\n\nAttributes\n----------", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "LassoLarsIC": {
        "cls": "sklearn.linear_model.least_angle.LassoLarsIC", 
        "defaults": {
            "copy_X": True, 
            "criterion": "aic", 
            "eps": 2.2204460492503131e-16, 
            "fit_intercept": True, 
            "max_iter": 500, 
            "normalize": True, 
            "precompute": "auto", 
            "verbose": False
        }, 
        "help_text": "Lasso model fit with Lars using BIC or AIC for model selection  The optimization objective for Lasso is::  (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1  AIC is the Akaike information criterion and BIC is the Bayes  Information criterion. Such criteria are useful to select the value  of the regularization parameter by making a trade-off between the  goodness of fit and the complexity of the model. A good model should  explain well the data while being simple.", 
        "parameters": [
            {
                "choices": [
                    "bic", 
                    "aic"
                ], 
                "default": "aic", 
                "help_text": "The type of criterion to use.\n", 
                "name": "criterion", 
                "type": "string"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Sets the verbosity amount\n", 
                "name": "verbose", 
                "required": False, 
                "type": "boolean_or_integer"
            }, 
            {
                "default": True, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If True, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "default": "auto", 
                "help_text": "Whether to use a precomputed Gram matrix to speed up\ncalculations. If set to ``'auto'`` let us decide. The Gram\nmatrix can also be passed as argument.\n", 
                "name": "precompute", 
                "type": "list"
            }, 
            {
                "default": 500, 
                "help_text": "Maximum number of iterations to perform. Can be used for\nearly stopping.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2.2204460492503131e-16, 
                "help_text": "The machine-precision regularization in the computation of the\nCholesky diagonal factors. Increase this for very ill-conditioned\nsystems. Unlike the ``tol`` parameter in some iterative\noptimization-based algorithms, this parameter does not control\nthe tolerance of the optimization.\n\n\nAttributes\n----------", 
                "name": "eps", 
                "required": False, 
                "type": "float"
            }
        ]
    }, 
    "LinearRegression": {
        "cls": "sklearn.linear_model.base.LinearRegression", 
        "defaults": {
            "copy_X": True, 
            "fit_intercept": True, 
            "n_jobs": 1, 
            "normalize": False
        }, 
        "help_text": "Ordinary least squares Linear Regression.", 
        "parameters": [
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If True, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 1, 
                "help_text": "The number of jobs to use for the computation.\nIf -1 all CPUs are used. This will only provide speedup for\nn_targets > 1 and sufficient large problems.\n\nAttributes\n----------", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "LinearSVC": {
        "cls": "sklearn.svm.classes.LinearSVC", 
        "defaults": {
            "C": 1.0, 
            "dual": True, 
            "fit_intercept": True, 
            "intercept_scaling": 1, 
            "loss": "l2", 
            "max_iter": 1000, 
            "multi_class": "ovr", 
            "penalty": "l2", 
            "tol": 0.0001, 
            "verbose": 0
        }, 
        "help_text": "Linear Support Vector Classification.  Similar to SVC with parameter kernel='linear', but implemented in terms of  liblinear rather than libsvm, so it has more flexibility in the choice of  penalties and loss functions and should scale better (to large numbers of  samples).  This class supports both dense and sparse input and the multiclass support  is handled according to a one-vs-the-rest scheme.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Penalty parameter C of the error term.\n", 
                "name": "C", 
                "required": False, 
                "type": "float"
            }, 
            {
                "choices": [
                    "l1", 
                    "l2"
                ], 
                "default": "l2", 
                "help_text": "Specifies the loss function. 'l1' is the hinge loss (standard SVM)\nwhile 'l2' is the squared hinge loss.\n", 
                "name": "loss", 
                "type": "string"
            }, 
            {
                "choices": [
                    "l1", 
                    "l2"
                ], 
                "default": "l2", 
                "help_text": "Specifies the norm used in the penalization. The 'l2'\npenalty is the standard used in SVC. The 'l1' leads to `coef_`\nvectors that are sparse.\n", 
                "name": "penalty", 
                "type": "string"
            }, 
            {
                "default": True, 
                "help_text": "Select the algorithm to either solve the dual or primal\noptimization problem. Prefer dual=False when n_samples > n_features.\n", 
                "name": "dual", 
                "type": "boolean"
            }, 
            {
                "default": 0.0001, 
                "help_text": "Tolerance for stopping criteria\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "choices": [
                    "ovr", 
                    "crammer_singer"
                ], 
                "default": "ovr", 
                "help_text": "Determines the multi-class strategy if `y` contains more than\ntwo classes.\n`ovr` trains n_classes one-vs-rest classifiers, while `crammer_singer`\noptimizes a joint objective over all classes.\nWhile `crammer_singer` is interesting from an theoretical perspective\nas it is consistent it is seldom used in practice and rarely leads to\nbetter accuracy and is more expensive to compute.\nIf `crammer_singer` is chosen, the options loss, penalty and dual will\nbe ignored.\n", 
                "name": "multi_class", 
                "type": "string"
            }, 
            {
                "default": True, 
                "help_text": "Whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 1, 
                "help_text": "when self.fit_intercept is True, instance vector x becomes\n[x, self.intercept_scaling],\ni.e. a \"synthetic\" feature with constant value equals to\nintercept_scaling is appended to the instance vector.\nThe intercept becomes intercept_scaling * synthetic feature weight\nNote! the synthetic feature weight is subject to l1/l2 regularization\nas all other features.\nTo lessen the effect of regularization on synthetic feature weight\n(and therefore on the intercept) intercept_scaling has to be increased\n", 
                "name": "intercept_scaling", 
                "required": False, 
                "type": "float"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "help_text": "Set the parameter C of class i to class_weight[i]*C for\nSVC. If not given, all classes are supposed to have\nweight one. The 'auto' mode uses the values of y to\nautomatically adjust weights inversely proportional to\nclass frequencies.\n", 
                "name": "class_weight", 
                "required": False, 
                "type": "auto_dict"
            }, 
            {
                "default": 0, 
                "help_text": "Enable verbose output. Note that this setting takes advantage of a\nper-process runtime setting in liblinear that, if enabled, may not work\nproperly in a multithreaded context.\n", 
                "name": "verbose", 
                "type": "integer"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator to use when\nshuffling the data.\n", 
                "name": "random_state", 
                "type": "integer"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of iterations to be run.\n\nAttributes\n----------", 
                "name": "max_iter", 
                "type": "integer"
            }
        ]
    }, 
    "LinearSVR": {
        "cls": "sklearn.svm.classes.LinearSVR", 
        "defaults": {
            "C": 1.0, 
            "dual": True, 
            "epsilon": 0.0, 
            "fit_intercept": True, 
            "intercept_scaling": 1.0, 
            "loss": "l1", 
            "max_iter": 1000, 
            "tol": 0.0001, 
            "verbose": 0
        }, 
        "help_text": "Linear Support Vector Regression.  Similar to SVR with parameter kernel='linear', but implemented in terms of  liblinear rather than libsvm, so it has more flexibility in the choice of  penalties and loss functions and should scale better (to large numbers of  samples).  This class supports both dense and sparse input.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Penalty parameter C of the error term. The penalty is a squared\nl2 penalty. The bigger this parameter, the less regularization is used.\n", 
                "name": "C", 
                "required": False, 
                "type": "float"
            }, 
            {
                "choices": [
                    "l1", 
                    "l2"
                ], 
                "default": "l1", 
                "help_text": "Specifies the loss function. 'l1' is the epsilon-insensitive loss\n(standard SVR) while 'l2' is the squared epsilon-insensitive loss.\n", 
                "name": "loss", 
                "type": "string"
            }, 
            {
                "default": 0.0, 
                "help_text": "Epsilon parameter in the epsilon-insensitive loss function. Note\nthat the value of this parameter depends on the scale of the target\nvariable y. If unsure, set epsilon=0.\n", 
                "name": "epsilon", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Select the algorithm to either solve the dual or primal\noptimization problem. Prefer dual=False when n_samples > n_features.\n", 
                "name": "dual", 
                "type": "boolean"
            }, 
            {
                "default": 0.0001, 
                "help_text": "Tolerance for stopping criteria\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 1.0, 
                "help_text": "when self.fit_intercept is True, instance vector x becomes\n[x, self.intercept_scaling],\ni.e. a \"synthetic\" feature with constant value equals to\nintercept_scaling is appended to the instance vector.\nThe intercept becomes intercept_scaling * synthetic feature weight\nNote! the synthetic feature weight is subject to l1/l2 regularization\nas all other features.\nTo lessen the effect of regularization on synthetic feature weight\n(and therefore on the intercept) intercept_scaling has to be increased\n", 
                "name": "intercept_scaling", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0, 
                "help_text": "Enable verbose output. Note that this setting takes advantage of a\nper-process runtime setting in liblinear that, if enabled, may not work\nproperly in a multithreaded context.\n", 
                "name": "verbose", 
                "type": "integer"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator to use when\nshuffling the data.\n", 
                "name": "random_state", 
                "type": "integer"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of iterations to be run.\n\nAttributes\n----------", 
                "name": "max_iter", 
                "type": "integer"
            }
        ]
    }, 
    "LogisticRegressionCV": {
        "cls": "sklearn.linear_model.logistic.LogisticRegressionCV", 
        "defaults": {
            "Cs": 10, 
            "dual": False, 
            "fit_intercept": True, 
            "intercept_scaling": 1.0, 
            "max_iter": 100, 
            "multi_class": "ovr", 
            "n_jobs": 1, 
            "penalty": "l2", 
            "refit": True, 
            "solver": "lbfgs", 
            "tol": 0.0001, 
            "verbose": 0
        }, 
        "help_text": "Logistic Regression CV (aka logit, MaxEnt) classifier.  This class implements logistic regression using liblinear, newton-cg or  LBFGS optimizer. The newton-cg and lbfgs solvers support only L2  regularization with primal formulation. The liblinear solver supports both  L1 and L2 regularization, with a dual formulation only for the L2 penalty.  For the grid of Cs values (that are set by default to be ten values in  a logarithmic scale between 1e-4 and 1e4), the best hyperparameter is  selected by the cross-validator StratifiedKFold, but it can be changed  using the cv parameter. In the case of newton-cg and lbfgs solvers,  we warm start along the path i.e guess the initial coefficients of the  present fit to be the coefficients got after convergence in the previous  fit, so in general it is supposed to be faster.  For a multiclass problem, the hyperparameters for each class are computed  using the best scores got by doing a one-vs-rest in parallel across all  folds and classes. Hence this is not the True multinomial loss.", 
        "parameters": [
            {
                "default": 10, 
                "help_text": "Each of the values in Cs describes the inverse of regularization\nstrength. If Cs is as an int, then a grid of Cs values are chosen\nin a logarithmic scale between 1e-4 and 1e4.\nLike in support vector machines, smaller values specify stronger\nregularization.\n", 
                "name": "Cs", 
                "type": "list"
            }, 
            {
                "default": True, 
                "help_text": "Specifies if a constant (a.k.a. bias or intercept) should be\nadded the decision function.\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "help_text": "Over-/undersamples the samples of each class according to the given\nweights. If not given, all classes are supposed to have weight one.\nThe 'auto' mode selects weights inversely proportional to class\nfrequencies in the training set.\n", 
                "name": "class_weight", 
                "required": False, 
                "type": "auto_dict"
            }, 
            {
                "help_text": "The default cross-validation generator used is Stratified K-Folds.\nIf an integer is provided, then it is the number of folds used.\nSee the module :mod:`sklearn.cross_validation` module for the\nlist of possible cross-validation objects.\n", 
                "name": "cv", 
                "type": "integer"
            }, 
            {
                "choices": [
                    "l1", 
                    "l2"
                ], 
                "default": "l2", 
                "help_text": "Used to specify the norm used in the penalization. The newton-cg and\nlbfgs solvers support only l2 penalties.\n", 
                "name": "penalty", 
                "type": "string"
            }, 
            {
                "default": False, 
                "help_text": "Dual or primal formulation. Dual formulation is only implemented for\nl2 penalty with liblinear solver. Prefer dual=False when\nn_samples > n_features.\n", 
                "name": "dual", 
                "type": "boolean"
            }, 
            {
                "help_text": "Scoring function to use as cross-validation criteria. For a list of\nscoring functions that can be used, look at :mod:`sklearn.metrics`.\nThe default scoring option used is accuracy_score.\n", 
                "name": "scoring", 
                "type": "object"
            }, 
            {
                "choices": [
                    "newton-cg", 
                    "lbfgs", 
                    "liblinear"
                ], 
                "default": "lbfgs", 
                "help_text": "Algorithm to use in the optimization problem.\n", 
                "name": "solver", 
                "type": "string"
            }, 
            {
                "default": 0.0001, 
                "help_text": "Tolerance for stopping criteria.\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 100, 
                "help_text": "Maximum number of iterations of the optimization algorithm.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "help_text": "Over-/undersamples the samples of each class according to the given\nweights. If not given, all classes are supposed to have weight one.\nThe 'auto' mode selects weights inversely proportional to class\nfrequencies in the training set.\n", 
                "name": "class_weight", 
                "required": False, 
                "type": "auto_dict"
            }, 
            {
                "default": 1, 
                "help_text": "Number of CPU cores used during the cross-validation loop. If given\na value of -1, all cores are used.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "For the liblinear and lbfgs solvers set verbose to any positive\nnumber for verbosity.\n", 
                "name": "verbose", 
                "type": "integer"
            }, 
            {
                "default": True, 
                "help_text": "If set to True, the scores are averaged across all folds, and the\ncoefs and the C that corresponds to the best score is taken, and a\nfinal refit is done using these parameters.\nOtherwise the coefs, intercepts and C that correspond to the\nbest scores across folds are averaged.\n", 
                "name": "refit", 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "ovr", 
                    "multinomial"
                ], 
                "default": "ovr", 
                "help_text": "Multiclass option can be either 'ovr' or 'multinomial'. If the option\nchosen is 'ovr', then a binary problem is fit for each label. Else\nthe loss minimised is the multinomial loss fit across\nthe entire probability distribution. Works only for the 'lbfgs'\nsolver.\n", 
                "name": "multi_class", 
                "type": "string"
            }, 
            {
                "default": 1.0, 
                "help_text": "Useful only if solver is liblinear.\nThis parameter is useful only when the solver 'liblinear' is used\nand self.fit_intercept is set to True. In this case, x becomes\n[x, self.intercept_scaling],\ni.e. a \"synthetic\" feature with constant value equals to\nintercept_scaling is appended to the instance vector.\nThe intercept becomes intercept_scaling * synthetic feature weight\nNote! the synthetic feature weight is subject to l1/l2 regularization\nas all other features.\nTo lessen the effect of regularization on synthetic feature weight\n(and therefore on the intercept) intercept_scaling has to be increased.\n\nAttributes\n----------", 
                "name": "intercept_scaling", 
                "type": "float"
            }
        ]
    }, 
    "MultiTaskElasticNet": {
        "cls": "sklearn.linear_model.coordinate_descent.MultiTaskElasticNet", 
        "defaults": {
            "alpha": 1.0, 
            "copy_X": True, 
            "fit_intercept": True, 
            "l1_ratio": 0.5, 
            "max_iter": 1000, 
            "normalize": False, 
            "selection": "cyclic", 
            "tol": 0.0001, 
            "warm_start": False
        }, 
        "help_text": "Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer  The optimization objective for MultiTaskElasticNet is::    (1 / (2 * n_samples)) * ||Y - XW||^Fro_2    + alpha * l1_ratio * ||W||_21    + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2  Where::    ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}  i.e. the sum of norm of each row.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Constant that multiplies the L1/L2 term. Defaults to 1.0\n", 
                "name": "alpha", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.5, 
                "help_text": "The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.\nFor l1_ratio = 0 the penalty is an L1/L2 penalty. For l1_ratio = 1 it\nis an L1 penalty.\nFor ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.\n", 
                "name": "l1_ratio", 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If ``True``, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of iterations\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0001, 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, reuse the solution of the previous call to fit as\ninitialization, otherwise, just erase the previous solution.\n", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": "cyclic", 
                "help_text": "If set to 'random', a random coefficient is updated every iteration\nrather than looping over features sequentially by default. This\n(setting to 'random') often leads to significantly faster convergence\nespecially when tol is higher than 1e-4.\n", 
                "name": "selection", 
                "type": "string"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator that selects\na random feature to update. Useful only when selection is set to\n'random'.\n\nAttributes\n----------", 
                "name": "random_state", 
                "type": "integer"
            }
        ]
    }, 
    "MultiTaskElasticNetCV": {
        "cls": "sklearn.linear_model.coordinate_descent.MultiTaskElasticNetCV", 
        "defaults": {
            "copy_X": True, 
            "eps": 0.001, 
            "fit_intercept": True, 
            "l1_ratio": 0.5, 
            "max_iter": 1000, 
            "n_alphas": 100, 
            "n_jobs": 1, 
            "normalize": False, 
            "selection": "cyclic", 
            "tol": 0.0001, 
            "verbose": 0
        }, 
        "help_text": "Multi-task L1/L2 ElasticNet with built-in cross-validation.  The optimization objective for MultiTaskElasticNet is::    (1 / (2 * n_samples)) * ||Y - XW||^Fro_2    + alpha * l1_ratio * ||W||_21    + 0.5 * alpha * (1 - l1_ratio) * ||W||_Fro^2  Where::    ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}  i.e. the sum of norm of each row.", 
        "parameters": [
            {
                "default": 0.001, 
                "help_text": "Length of the path. ``eps=1e-3`` means that\n``alpha_min / alpha_max = 1e-3``.\n", 
                "name": "eps", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "List of alphas where to compute the models.\nIf not provided, set automatically.\n", 
                "name": "alphas", 
                "required": False, 
                "type": "list"
            }, 
            {
                "default": 100, 
                "help_text": "Number of alphas along the regularization path\n", 
                "name": "n_alphas", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.5, 
                "help_text": "The ElasticNet mixing parameter, with 0 < l1_ratio <= 1.\nFor l1_ratio = 0 the penalty is an L1/L2 penalty. For l1_ratio = 1 it\nis an L1 penalty.\nFor ``0 < l1_ratio < 1``, the penalty is a combination of L1/L2 and L2.\n", 
                "name": "l1_ratio", 
                "type": "list"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If ``True``, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of iterations\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0001, 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "If an integer is passed, it is the number of fold (default 3).\nSpecific cross-validation objects can be passed, see the\n:mod:`sklearn.cross_validation` module for the list of possible\nobjects.\n", 
                "name": "cv", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "Amount of verbosity.\n", 
                "name": "verbose", 
                "type": "boolean_or_integer"
            }, 
            {
                "default": 1, 
                "help_text": "Number of CPUs to use during the cross validation. If ``-1``, use\nall the CPUs. Note that this is used only if multiple values for\nl1_ratio are given.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "cyclic", 
                "help_text": "If set to 'random', a random coefficient is updated every iteration\nrather than looping over features sequentially by default. This\n(setting to 'random') often leads to significantly faster convergence\nespecially when tol is higher than 1e-4.\n", 
                "name": "selection", 
                "type": "string"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator that selects\na random feature to update. Useful only when selection is set to\n'random'.\n\nAttributes\n----------", 
                "name": "random_state", 
                "type": "integer"
            }
        ]
    }, 
    "MultiTaskLasso": {
        "cls": "sklearn.linear_model.coordinate_descent.MultiTaskLasso", 
        "defaults": {
            "alpha": 1.0, 
            "copy_X": True, 
            "fit_intercept": True, 
            "max_iter": 1000, 
            "normalize": False, 
            "selection": "cyclic", 
            "tol": 0.0001, 
            "warm_start": False
        }, 
        "help_text": "Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer  The optimization objective for Lasso is::    (1 / (2 * n_samples)) * ||Y - XW||^2_Fro + alpha * ||W||_21  Where::    ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}  i.e. the sum of norm of earch row.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Constant that multiplies the L1/L2 term. Defaults to 1.0\n", 
                "name": "alpha", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If ``True``, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of iterations\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0001, 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, reuse the solution of the previous call to fit as\ninitialization, otherwise, just erase the previous solution.\n", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": "cyclic", 
                "help_text": "If set to 'random', a random coefficient is updated every iteration\nrather than looping over features sequentially by default. This\n(setting to 'random') often leads to significantly faster convergence\nespecially when tol is higher than 1e-4\n", 
                "name": "selection", 
                "type": "string"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator that selects\na random feature to update. Useful only when selection is set to\n'random'.\n\nAttributes\n----------", 
                "name": "random_state", 
                "type": "integer"
            }
        ]
    }, 
    "MultiTaskLassoCV": {
        "cls": "sklearn.linear_model.coordinate_descent.MultiTaskLassoCV", 
        "defaults": {
            "copy_X": True, 
            "eps": 0.001, 
            "fit_intercept": True, 
            "max_iter": 1000, 
            "n_alphas": 100, 
            "n_jobs": 1, 
            "normalize": False, 
            "selection": "cyclic", 
            "tol": 0.0001, 
            "verbose": False
        }, 
        "help_text": "Multi-task L1/L2 Lasso with built-in cross-validation.  The optimization objective for MultiTaskLasso is::    (1 / (2 * n_samples)) * ||Y - XW||^Fro_2 + alpha * ||W||_21  Where::    ||W||_21 = \\sum_i \\sqrt{\\sum_j w_{ij}^2}  i.e. the sum of norm of each row.", 
        "parameters": [
            {
                "default": 0.001, 
                "help_text": "Length of the path. ``eps=1e-3`` means that\n``alpha_min / alpha_max = 1e-3``.\n", 
                "name": "eps", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "List of alphas where to compute the models.\nIf not provided, set automaticlly.\n", 
                "name": "alphas", 
                "required": False, 
                "type": "list"
            }, 
            {
                "default": 100, 
                "help_text": "Number of alphas along the regularization path\n", 
                "name": "n_alphas", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If ``True``, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If ``True``, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 1000, 
                "help_text": "The maximum number of iterations.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0001, 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "If an integer is passed, it is the number of fold (default 3).\nSpecific cross-validation objects can be passed, see the\n:mod:`sklearn.cross_validation` module for the list of possible\nobjects.\n", 
                "name": "cv", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Amount of verbosity.\n", 
                "name": "verbose", 
                "type": "boolean_or_integer"
            }, 
            {
                "default": 1, 
                "help_text": "Number of CPUs to use during the cross validation. If ``-1``, use\nall the CPUs. Note that this is used only if multiple values for\nl1_ratio are given.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "cyclic", 
                "help_text": "If set to 'random', a random coefficient is updated every iteration\nrather than looping over features sequentially by default. This\n(setting to 'random') often leads to significantly faster convergence\nespecially when tol is higher than 1e-4.\n", 
                "name": "selection", 
                "type": "string"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator that selects\na random feature to update. Useful only when selection is set to\n'random'.\n\nAttributes\n----------", 
                "name": "random_state", 
                "type": "integer"
            }
        ]
    }, 
    "MultinomialNB": {
        "cls": "sklearn.naive_bayes.MultinomialNB", 
        "defaults": {
            "alpha": 1.0, 
            "fit_prior": True
        }, 
        "help_text": "Naive Bayes classifier for multinomial models  The multinomial Naive Bayes classifier is suitable for classification with  discrete features (e.g., word counts for text classification). The  multinomial distribution normally requires integer feature counts. However,  in practice, fractional counts such as tf-idf may also work.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Additive (Laplace/Lidstone) smoothing parameter\n(0 for no smoothing).\n", 
                "name": "alpha", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Whether to learn class prior probabilities or not.\nIf False, a uniform prior will be used.\n", 
                "name": "fit_prior", 
                "type": "boolean"
            }, 
            {
                "help_text": "Prior probabilities of the classes. If specified the priors are not\nadjusted according to the data.\n\nAttributes\n----------", 
                "name": "class_prior", 
                "type": "list"
            }
        ]
    }, 
    "NearestCentroid": {
        "cls": "sklearn.neighbors.nearest_centroid.NearestCentroid", 
        "defaults": {
            "metric": "euclidean"
        }, 
        "help_text": "Nearest centroid classifier.  Each class is represented by its centroid, with test samples classified to  the class with the nearest centroid.", 
        "parameters": [
            {
                "default": "euclidean", 
                "help_text": "The metric to use when calculating distance between instances in a\nfeature array. If metric is a string or callable, it must be one of\nthe options allowed by metrics.pairwise.pairwise_distances for its\nmetric parameter.\nThe centroids for the samples corresponding to each class is the point\nfrom which the sum of the distances (according to the metric) of all\nsamples that belong to that particular class are minimized.\nIf the \"manhattan\" metric is provided, this centroid is the median and\nfor all other metrics, the centroid is now set to be the mean.\n", 
                "name": "metric", 
                "type": "string"
            }, 
            {
                "help_text": "Threshold for shrinking centroids to remove features.\n\nAttributes\n----------", 
                "name": "shrink_threshold", 
                "required": False, 
                "type": "float"
            }
        ]
    }, 
    "NuSVC": {
        "cls": "sklearn.svm.classes.NuSVC", 
        "defaults": {
            "cache_size": 200, 
            "coef0": 0.0, 
            "degree": 3, 
            "gamma": 0.0, 
            "kernel": "rbf", 
            "max_iter": -1, 
            "nu": 0.5, 
            "probability": False, 
            "shrinking": True, 
            "tol": 0.001, 
            "verbose": False
        }, 
        "help_text": "Nu-Support Vector Classification.  Similar to SVC but uses a parameter to control the number of support  vectors.  The implementation is based on libsvm.", 
        "parameters": [
            {
                "default": 0.5, 
                "help_text": "An upper bound on the fraction of training errors and a lower\nbound of the fraction of support vectors. Should be in the\ninterval (0, 1].\n", 
                "name": "nu", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": "rbf", 
                "help_text": "Specifies the kernel type to be used in the algorithm.\nIt must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or\na callable.\nIf none is given, 'rbf' will be used. If a callable is given it is\nused to precompute the kernel matrix.\n", 
                "name": "kernel", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 3, 
                "help_text": "degree of kernel function\nis significant only in poly, rbf, sigmoid\n", 
                "name": "degree", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "kernel coefficient for rbf and poly, if gamma is 0.0 then 1/n_features\nwill be taken.\n", 
                "name": "gamma", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.0, 
                "help_text": "independent term in kernel function. It is only significant\nin poly/sigmoid.\n", 
                "name": "coef0", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "Whether to enable probability estimates. This must be enabled prior\nto calling `fit`, and will slow down that method.\n", 
                "name": "probability", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "Whether to use the shrinking heuristic.\n", 
                "name": "shrinking", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 0.001, 
                "help_text": "Tolerance for stopping criterion.\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 200, 
                "help_text": "Specify the size of the kernel cache (in MB)\n", 
                "name": "cache_size", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "Enable verbose output. Note that this setting takes advantage of a\nper-process runtime setting in libsvm that, if enabled, may not work\nproperly in a multithreaded context.\n", 
                "name": "verbose", 
                "type": "boolean"
            }, 
            {
                "default": -1, 
                "help_text": "Hard limit on iterations within solver, or -1 for no limit.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator to use when\nshuffling the data for probability estimation.\n\nAttributes\n----------", 
                "name": "random_state", 
                "type": "integer"
            }
        ]
    }, 
    "NuSVR": {
        "cls": "sklearn.svm.classes.NuSVR", 
        "defaults": {
            "C": 1.0, 
            "cache_size": 200, 
            "coef0": 0.0, 
            "degree": 3, 
            "gamma": 0.0, 
            "kernel": "rbf", 
            "max_iter": -1, 
            "nu": 0.5, 
            "shrinking": True, 
            "tol": 0.001, 
            "verbose": False
        }, 
        "help_text": "Nu Support Vector Regression.  Similar to NuSVC, for regression, uses a parameter nu to control  the number of support vectors. However, unlike NuSVC, where nu  replaces C, here nu replaces with the parameter epsilon of SVR.  The implementation is based on libsvm.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "penalty parameter C of the error term.\n", 
                "name": "C", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.5, 
                "help_text": "An upper bound on the fraction of training errors and a lower bound of\nthe fraction of support vectors. Should be in the interval (0, 1]. By\ndefault 0.5 will be taken. Only available if impl='nu_svc'.\n", 
                "name": "nu", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": "rbf", 
                "help_text": "Specifies the kernel type to be used in the algorithm.\nIt must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or\na callable.\nIf none is given, 'rbf' will be used. If a callable is given it is\nused to precompute the kernel matrix.\n", 
                "name": "kernel", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 3, 
                "help_text": "degree of kernel function\nis significant only in poly, rbf, sigmoid\n", 
                "name": "degree", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "kernel coefficient for rbf and poly, if gamma is 0.0 then 1/n_features\nwill be taken.\n", 
                "name": "gamma", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.0, 
                "help_text": "independent term in kernel function. It is only significant\nin poly/sigmoid.\n", 
                "name": "coef0", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Whether to use the shrinking heuristic.\n", 
                "name": "shrinking", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 0.001, 
                "help_text": "Tolerance for stopping criterion.\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 200, 
                "help_text": "Specify the size of the kernel cache (in MB)\n", 
                "name": "cache_size", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "Enable verbose output. Note that this setting takes advantage of a\nper-process runtime setting in libsvm that, if enabled, may not work\nproperly in a multithreaded context.\n", 
                "name": "verbose", 
                "type": "boolean"
            }, 
            {
                "default": -1, 
                "help_text": "Hard limit on iterations within solver, or -1 for no limit.\n\nAttributes\n----------", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "OneVsOneClassifier": {
        "cls": "sklearn.multiclass.OneVsOneClassifier", 
        "defaults": {
            "estimator": 1
        }, 
        "help_text": "One-vs-one multiclass strategy  This strategy consists in fitting one classifier per class pair.  At prediction time, the class which received the most votes is selected.  Since it requires to fit `n_classes * (n_classes - 1) / 2` classifiers,  this method is usually slower than one-vs-the-rest, due to its  O(n_classes^2) complexity. However, this method may be advantageous for  algorithms such as kernel algorithms which don't scale well with  `n_samples`. This is because each individual learning problem only involves  a small subset of the data whereas, with one-vs-the-rest, the complete  dataset is used `n_classes` times.", 
        "parameters": [
            {
                "default": 1, 
                "help_text": "An estimator object implementing `fit` and one of `decision_function`\nor `predict_proba`.\n", 
                "name": "estimator", 
                "type": "object"
            }, 
            {
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "OneVsRestClassifier": {
        "cls": "sklearn.multiclass.OneVsRestClassifier", 
        "defaults": {
            "estimator": 1
        }, 
        "help_text": "One-vs-the-rest (OvR) multiclass/multilabel strategy  Also known as one-vs-all, this strategy consists in fitting one classifier  per class. For each classifier, the class is fitted against all the other  classes. In addition to its computational efficiency (only `n_classes`  classifiers are needed), one advantage of this approach is its  interpretability. Since each class is represented by one and one classifier  only, it is possible to gain knowledge about the class by inspecting its  corresponding classifier. This is the most commonly used strategy for  multiclass classification and is a fair default choice.  This strategy can also be used for multilabel learning, where a classifier  is used to predict multiple labels for instance, by fitting on a 2-d matrix  in which cell [i, j] is 1 if sample i has label j and 0 otherwise.  In the multilabel learning literature, OvR is also known as the binary  relevance method.", 
        "parameters": [
            {
                "default": 1, 
                "help_text": "An estimator object implementing `fit` and one of `decision_function`\nor `predict_proba`.\n", 
                "name": "estimator", 
                "type": "object"
            }, 
            {
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "OrthogonalMatchingPursuit": {
        "cls": "sklearn.linear_model.omp.OrthogonalMatchingPursuit", 
        "defaults": {
            "fit_intercept": True, 
            "normalize": True, 
            "precompute": "auto"
        }, 
        "help_text": "Orthogonal Matching Pursuit model (OMP)", 
        "parameters": [
            {
                "help_text": "Desired number of non-zero entries in the solution. If None (by\ndefault) this value is set to 10% of n_features.\n", 
                "name": "n_nonzero_coefs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "Maximum norm of the residual. If not None, overrides n_nonzero_coefs.\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If False, the regressors X are assumed to be already normalized.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "default": "auto", 
                "help_text": "Whether to use a precomputed Gram and Xy matrix to speed up\ncalculations. Improves performance when `n_targets` or `n_samples` is\nvery large. Note that if you already have such matrices, you can pass\nthem directly to the fit method.\n\nAttributes\n----------", 
                "name": "precompute", 
                "type": "string"
            }
        ]
    }, 
    "OrthogonalMatchingPursuitCV": {
        "cls": "sklearn.linear_model.omp.OrthogonalMatchingPursuitCV", 
        "defaults": {
            "copy": True, 
            "fit_intercept": True, 
            "n_jobs": 1, 
            "normalize": True, 
            "verbose": False
        }, 
        "help_text": "Cross-validated Orthogonal Matching Pursuit model (OMP)", 
        "parameters": [
            {
                "default": True, 
                "help_text": "Whether the design matrix X must be copied by the algorithm. A False\nvalue is only helpful if X is already Fortran-ordered, otherwise a\ncopy is made anyway.\n", 
                "name": "copy", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If False, the regressors X are assumed to be already normalized.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "help_text": "Maximum numbers of iterations to perform, therefore maximum features\nto include. 10% of ``n_features`` but at least 5 if available.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "see :mod:`sklearn.cross_validation`. If ``None`` is passed, default to\na 5-fold strategy\n", 
                "name": "cv", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 1, 
                "help_text": "Number of CPUs to use during the cross validation. If ``-1``, use\nall the CPUs\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Sets the verbosity amount\n\nAttributes\n----------", 
                "name": "verbose", 
                "required": False, 
                "type": "boolean_or_integer"
            }
        ]
    }, 
    "OutputCodeClassifier": {
        "cls": "sklearn.multiclass.OutputCodeClassifier", 
        "defaults": {
            "estimator": 1.5, 
            "random_state": 1
        }, 
        "help_text": "(Error-Correcting) Output-Code multiclass strategy  Output-code based strategies consist in representing each class with a  binary code (an array of 0s and 1s). At fitting time, one binary  classifier per bit in the code book is fitted. At prediction time, the  classifiers are used to project new points in the class space and the class  closest to the points is chosen. The main advantage of these strategies is  that the number of classifiers used can be controlled by the user, either  for compressing the model (0 < code_size < 1) or for making the model more  robust to errors (code_size > 1). See the documentation for more details.", 
        "parameters": [
            {
                "default": 1.5, 
                "help_text": "An estimator object implementing `fit` and one of `decision_function`\nor `predict_proba`.\n", 
                "name": "estimator", 
                "type": "object"
            }, 
            {
                "help_text": "Percentage of the number of classes to be used to create the code book.\nA number between 0 and 1 will require fewer classifiers than\none-vs-the-rest. A number greater than 1 will require more classifiers\nthan one-vs-the-rest.\n", 
                "name": "code_size", 
                "type": "float"
            }, 
            {
                "default": 1, 
                "help_text": "The generator used to initialize the codebook. Defaults to\nnumpy.random.\n", 
                "name": "random_state", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "The number of jobs to use for the computation. If -1 all CPUs are used.\nIf 1 is given, no parallel computing code is used at all, which is\nuseful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are\nused. Thus for n_jobs = -2, all CPUs but one are used.\n\nAttributes\n----------", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "PLSCanonical": {
        "cls": "sklearn.cross_decomposition.pls_.PLSCanonical", 
        "defaults": {
            "algorithm": "nipals", 
            "copy": True, 
            "max_iter": 500, 
            "n_components": 2, 
            "scale": True, 
            "tol": 1e-06
        }, 
        "help_text": "PLSCanonical implements the 2 blocks canonical PLS of the original Wold  algorithm [Tenenhaus 1998] p.204, referred as PLS-C2A in [Wegelin 2000].  This class inherits from PLS with mode=\"A\" and deflation_mode=\"canonical\",  norm_y_weights=True and algorithm=\"nipals\", but svd should provide similar  results up to numerical errors.", 
        "parameters": [
            {
                "default": True, 
                "help_text": "", 
                "name": "scale", 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "nipals", 
                    "svd"
                ], 
                "default": "nipals", 
                "help_text": "The algorithm used to estimate the weights. It will be called\nn_components times, i.e. once for each iteration of the outer loop.\n", 
                "name": "algorithm", 
                "type": "string"
            }, 
            {
                "default": 500, 
                "help_text": "the maximum number of iterations of the NIPALS inner loop (used\nonly if algorithm=\"nipals\")\n", 
                "name": "max_iter", 
                "type": "integer"
            }, 
            {
                "default": 1e-06, 
                "help_text": "the tolerance used in the iterative algorithm\n", 
                "name": "tol", 
                "type": "string"
            }, 
            {
                "default": True, 
                "help_text": "Whether the deflation should be done on a copy. Let the default\nvalue to True unless you don't care about side effect\n", 
                "name": "copy", 
                "type": "boolean"
            }, 
            {
                "default": 2, 
                "help_text": "\nAttributes\n----------", 
                "name": "n_components", 
                "type": "integer"
            }
        ]
    }, 
    "PLSRegression": {
        "cls": "sklearn.cross_decomposition.pls_.PLSRegression", 
        "defaults": {
            "copy": True, 
            "max_iter": 500, 
            "n_components": 2, 
            "scale": True, 
            "tol": 1e-06
        }, 
        "help_text": "PLS regression  PLSRegression implements the PLS 2 blocks regression known as PLS2 or PLS1  in case of one dimensional response.  This class inherits from _PLS with mode=\"A\", deflation_mode=\"regression\",  norm_y_weights=False and algorithm=\"nipals\".", 
        "parameters": [
            {
                "default": 2, 
                "help_text": "Number of components to keep.\n", 
                "name": "n_components", 
                "type": "integer"
            }, 
            {
                "default": True, 
                "help_text": "whether to scale the data\n", 
                "name": "scale", 
                "type": "boolean"
            }, 
            {
                "default": 500, 
                "help_text": "the maximum number of iterations of the NIPALS inner loop (used\nonly if algorithm=\"nipals\")\n", 
                "name": "max_iter", 
                "type": "integer"
            }, 
            {
                "default": 1e-06, 
                "help_text": "Tolerance used in the iterative algorithm default 1e-06.\n", 
                "name": "tol", 
                "type": "string"
            }, 
            {
                "default": True, 
                "help_text": "Whether the deflation should be done on a copy. Let the default\nvalue to True unless you don't care about side effect\n\nAttributes\n----------", 
                "name": "copy", 
                "type": "boolean"
            }
        ]
    }, 
    "PassiveAggressiveClassifier": {
        "cls": "sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier", 
        "defaults": {
            "C": 1.0, 
            "fit_intercept": True, 
            "loss": "hinge", 
            "n_iter": 5, 
            "n_jobs": 1, 
            "shuffle": False, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "Passive Aggressive Classifier", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Maximum step size (regularization). Defaults to 1.0.\n", 
                "name": "C", 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Whether the intercept should be estimated or not. If False, the\ndata is assumed to be already centered. Defaults to True.\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": 5, 
                "help_text": "The number of passes over the training data (aka epochs).\nDefaults to 5.\n", 
                "name": "n_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Whether or not the training data should be shuffled after each epoch.\nDefaults to False.\n", 
                "name": "shuffle", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator to use when\nshuffling the data.\n", 
                "name": "random_state", 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "The verbosity level\n", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "The number of CPUs to use to do the OVA (One Versus All, for\nmulti-class problems) computation. -1 means 'all CPUs'. Defaults\nto 1.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "hinge", 
                "help_text": "The loss function to be used:", 
                "name": "loss", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": False, 
                "help_text": "When set to True, reuse the solution of the previous call to fit as\ninitialization, otherwise, just erase the previous solution.\n\nAttributes\n----------", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "PassiveAggressiveRegressor": {
        "cls": "sklearn.linear_model.passive_aggressive.PassiveAggressiveRegressor", 
        "defaults": {
            "C": 1.0, 
            "epsilon": 0.1, 
            "fit_intercept": True, 
            "loss": "epsilon_insensitive", 
            "n_iter": 5, 
            "shuffle": False, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "Passive Aggressive Regressor", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Maximum step size (regularization). Defaults to 1.0.\n", 
                "name": "C", 
                "type": "float"
            }, 
            {
                "default": 0.1, 
                "help_text": "If the difference between the current prediction and the correct label\nis below this threshold, the model is not updated.\n", 
                "name": "epsilon", 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Whether the intercept should be estimated or not. If False, the\ndata is assumed to be already centered. Defaults to True.\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": 5, 
                "help_text": "The number of passes over the training data (aka epochs).\nDefaults to 5.\n", 
                "name": "n_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Whether or not the training data should be shuffled after each epoch.\nDefaults to False.\n", 
                "name": "shuffle", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator to use when\nshuffling the data.\n", 
                "name": "random_state", 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "The verbosity level\n", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "epsilon_insensitive", 
                "help_text": "The loss function to be used:", 
                "name": "loss", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": False, 
                "help_text": "When set to True, reuse the solution of the previous call to fit as\ninitialization, otherwise, just erase the previous solution.\n\nAttributes\n----------", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "Perceptron": {
        "cls": "sklearn.linear_model.perceptron.Perceptron", 
        "defaults": {
            "alpha": 0.0001, 
            "eta0": 1.0, 
            "fit_intercept": True, 
            "n_iter": 5, 
            "n_jobs": 1, 
            "random_state": 0, 
            "shuffle": False, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "Perceptron", 
        "parameters": [
            {
                "choices": [
                    "l2", 
                    "l1", 
                    "elasticnet"
                ], 
                "help_text": "The penalty (aka regularization term) to be used. Defaults to None.\n", 
                "name": "penalty", 
                "type": "string"
            }, 
            {
                "default": 0.0001, 
                "help_text": "Constant that multiplies the regularization term if regularization is\nused. Defaults to 0.0001\n", 
                "name": "alpha", 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Whether the intercept should be estimated or not. If False, the\ndata is assumed to be already centered. Defaults to True.\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": 5, 
                "help_text": "The number of passes over the training data (aka epochs).\nDefaults to 5.\n", 
                "name": "n_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Whether or not the training data should be shuffled after each epoch.\nDefaults to False.\n", 
                "name": "shuffle", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 0, 
                "help_text": "The seed of the pseudo random number generator to use when\nshuffling the data.\n", 
                "name": "random_state", 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "The verbosity level\n", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "The number of CPUs to use to do the OVA (One Versus All, for\nmulti-class problems) computation. -1 means 'all CPUs'. Defaults\nto 1.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1.0, 
                "help_text": "Constant by which the updates are multiplied. Defaults to 1.\n", 
                "name": "eta0", 
                "type": "float"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "help_text": "Preset for the class_weight fit parameter.\n\nWeights associated with classes. If not given, all classes\nare supposed to have weight one.\n\nThe \"auto\" mode uses the values of y to automatically adjust\nweights inversely proportional to class frequencies.\n", 
                "name": "class_weight", 
                "required": False, 
                "type": "auto_dict"
            }, 
            {
                "default": False, 
                "help_text": "When set to True, reuse the solution of the previous call to fit as\ninitialization, otherwise, just erase the previous solution.\n\nAttributes\n----------", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "QDA": {
        "cls": "sklearn.qda.QDA", 
        "defaults": {
            "reg_param": 0.0
        }, 
        "help_text": "Quadratic Discriminant Analysis (QDA)  A classifier with a quadratic decision boundary, generated  by fitting class conditional densities to the data  and using Bayes' rule.  The model fits a Gaussian density to each class.", 
        "parameters": [
            {
                "help_text": "Priors on classes\n", 
                "name": "priors", 
                "required": False, 
                "type": "list"
            }, 
            {
                "default": 0.0, 
                "help_text": "Regularizes the covariance estimate as\n``(1-reg_param)*Sigma + reg_param*np.eye(n_features)``\n\nAttributes\n----------", 
                "name": "reg_param", 
                "required": False, 
                "type": "float"
            }
        ]
    }, 
    "RANSACRegressor": {
        "cls": "sklearn.linear_model.ransac.RANSACRegressor", 
        "defaults": {
            "max_trials": 100, 
            "stop_n_inliers": Infinity, 
            "stop_probability": 0.99, 
            "stop_score": Infinity
        }, 
        "help_text": "RANSAC (RANdom SAmple Consensus) algorithm.  RANSAC is an iterative algorithm for the robust estimation of parameters  from a subset of inliers from the complete data set. More information can  be found in the general documentation of linear models.  A detailed description of the algorithm can be found in the documentation  of the ``linear_model`` sub-package.", 
        "parameters": [
            {
                "help_text": "Base estimator object which implements the following methods:\n", 
                "name": "base_estimator", 
                "required": False, 
                "type": "object"
            }, 
            {
                "help_text": "Minimum number of samples chosen randomly from original data. Treated\nas an absolute number of samples for `min_samples >= 1`, treated as a\nrelative number `ceil(min_samples * X.shape[0]`) for\n`min_samples < 1`. This is typically chosen as the minimal number of\nsamples necessary to estimate the given `base_estimator`. By default a\n``sklearn.linear_model.LinearRegression()`` estimator is assumed and\n`min_samples` is chosen as ``X.shape[1] + 1``.\n", 
                "name": "min_samples", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "Maximum residual for a data sample to be classified as an inlier.\nBy default the threshold is chosen as the MAD (median absolute\ndeviation) of the target values `y`.\n", 
                "name": "residual_threshold", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "This function is called with the randomly selected data before the", 
                "name": "is_data_valid", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "This function is called with the estimated model and the randomly", 
                "name": "is_model_valid", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 100, 
                "help_text": "Maximum number of iterations for random sample selection.\n", 
                "name": "max_trials", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": Infinity, 
                "help_text": "Stop iteration if at least this number of inliers are found.\n", 
                "name": "stop_n_inliers", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": Infinity, 
                "help_text": "Stop iteration if score is greater equal than this threshold.\n", 
                "name": "stop_score", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.99, 
                "help_text": "RANSAC iteration stops if at least one outlier-free set of the training\ndata is sampled in RANSAC. This requires to generate at least N\nsamples (iterations)::\n\nN >= log(1 - probability) / log(1 - e**m)\n\nwhere the probability (confidence) is typically set to high value such\nas 0.99 (the default) and e is the current fraction of inliers w.r.t.\nthe total number of samples.\n", 
                "name": "stop_probability", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "Metric to reduce the dimensionality of the residuals to 1 for\nmulti-dimensional target values ``y.shape[1] > 1``. By default the sum\nof absolute differences is used::\n", 
                "name": "residual_metric", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "The generator used to initialize the centers. If an integer is\ngiven, it fixes the seed. Defaults to the global numpy random\nnumber generator.\n\nAttributes\n----------", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "RadiusNeighborsClassifier": {
        "cls": "sklearn.neighbors.classification.RadiusNeighborsClassifier", 
        "defaults": {
            "algorithm": "auto", 
            "leaf_size": 30, 
            "metric": "minkowski", 
            "p": 2, 
            "radius": 1.0, 
            "weights": "uniform"
        }, 
        "help_text": "Classifier implementing a vote among neighbors within a given radius", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Range of parameter space to use by default for :meth`radius_neighbors`\nqueries.\n", 
                "name": "radius", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": "uniform", 
                "help_text": "weight function used in prediction. Possible values:\n", 
                "name": "weights", 
                "type": "string"
            }, 
            {
                "choices": [
                    "auto", 
                    "ball_tree", 
                    "kd_tree", 
                    "brute"
                ], 
                "default": "auto", 
                "help_text": "Algorithm used to compute the nearest neighbors:\n\n- 'ball_tree' will use :class:`BallTree`\n- 'kd_tree' will use :class:`KDtree`\n- 'brute' will use a brute-force search.\n- 'auto' will attempt to decide the most appropriate algorithm\nbased on the values passed to :meth:`fit` method.\n", 
                "name": "algorithm", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 30, 
                "help_text": "Leaf size passed to BallTree or KDTree. This can affect the\nspeed of the construction and query, as well as the memory\nrequired to store the tree. The optimal value depends on the\nnature of the problem.\n", 
                "name": "leaf_size", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "minkowski", 
                "help_text": "the distance metric to use for the tree. The default metric is\nminkowski, and with p=2 is equivalent to the standard Euclidean\nmetric. See the documentation of the DistanceMetric class for a\nlist of available metrics.\n", 
                "name": "metric", 
                "type": "string"
            }, 
            {
                "default": 2, 
                "help_text": "Power parameter for the Minkowski metric. When p = 1, this is\nequivalent to using manhattan_distance (l1), and euclidean_distance\n(l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.\n", 
                "name": "p", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "Label, which is given for outlier samples (samples with no\nneighbors on given radius).\nIf set to None, ValueError is raised, when outlier is detected.\n", 
                "name": "outlier_label", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "additional keyword arguments for the metric function.\n\nExamples\n--------\n>>> X = [[0], [1], [2], [3]]\n>>> y = [0, 0, 1, 1]\n>>> from sklearn.neighbors import RadiusNeighborsClassifier\n>>> neigh = RadiusNeighborsClassifier(radius=1.0)", 
                "name": "metric_params", 
                "required": False, 
                "type": "dict"
            }
        ]
    }, 
    "RadiusNeighborsRegressor": {
        "cls": "sklearn.neighbors.regression.RadiusNeighborsRegressor", 
        "defaults": {
            "algorithm": "auto", 
            "leaf_size": 30, 
            "metric": "minkowski", 
            "p": 2, 
            "radius": 1.0, 
            "weights": "uniform"
        }, 
        "help_text": "Regression based on neighbors within a fixed radius.  The target is predicted by local interpolation of the targets  associated of the nearest neighbors in the training set.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Range of parameter space to use by default for :meth`radius_neighbors`\nqueries.\n", 
                "name": "radius", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": "uniform", 
                "help_text": "weight function used in prediction. Possible values:\n", 
                "name": "weights", 
                "type": "string"
            }, 
            {
                "choices": [
                    "auto", 
                    "ball_tree", 
                    "kd_tree", 
                    "brute"
                ], 
                "default": "auto", 
                "help_text": "Algorithm used to compute the nearest neighbors:\n\n- 'ball_tree' will use :class:`BallTree`\n- 'kd_tree' will use :class:`KDtree`\n- 'brute' will use a brute-force search.\n- 'auto' will attempt to decide the most appropriate algorithm\nbased on the values passed to :meth:`fit` method.\n", 
                "name": "algorithm", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 30, 
                "help_text": "Leaf size passed to BallTree or KDTree. This can affect the\nspeed of the construction and query, as well as the memory\nrequired to store the tree. The optimal value depends on the\nnature of the problem.\n", 
                "name": "leaf_size", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "minkowski", 
                "help_text": "the distance metric to use for the tree. The default metric is\nminkowski, and with p=2 is equivalent to the standard Euclidean\nmetric. See the documentation of the DistanceMetric class for a\nlist of available metrics.\n", 
                "name": "metric", 
                "type": "string"
            }, 
            {
                "default": 2, 
                "help_text": "Power parameter for the Minkowski metric. When p = 1, this is\nequivalent to using manhattan_distance (l1), and euclidean_distance\n(l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.\n", 
                "name": "p", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "additional keyword arguments for the metric function.\n\nExamples\n--------\n>>> X = [[0], [1], [2], [3]]\n>>> y = [0, 0, 1, 1]\n>>> from sklearn.neighbors import RadiusNeighborsRegressor\n>>> neigh = RadiusNeighborsRegressor(radius=1.0)", 
                "name": "metric_params", 
                "required": False, 
                "type": "dict"
            }
        ]
    }, 
    "Ridge": {
        "cls": "sklearn.linear_model.ridge.Ridge", 
        "defaults": {
            "alpha": 1.0, 
            "copy_X": True, 
            "fit_intercept": True, 
            "normalize": False, 
            "solver": "auto", 
            "tol": 0.001
        }, 
        "help_text": "Linear least squares with l2 regularization.  This model solves a regression model where the loss function is  the linear least squares function and regularization is given by  the l2-norm. Also known as Ridge Regression or Tikhonov regularization.  This estimator has built-in support for multi-variate regression  (i.e., when y is a 2d-array of shape [n_samples, n_targets]).", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "shape = [n_targets]\nSmall positive values of alpha improve the conditioning of the problem\nand reduce the variance of the estimates. Alpha corresponds to\n``(2*C)^-1`` in other linear models such as LogisticRegression or\nLinearSVC. If an array is passed, penalties are assumed to be specific\nto the targets. Hence they must correspond in number.\n", 
                "name": "alpha", 
                "type": "list"
            }, 
            {
                "default": True, 
                "help_text": "If True, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "Whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "help_text": "Maximum number of iterations for conjugate gradient solver.\nThe default value is determined by scipy.sparse.linalg.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto", 
                    "svd", 
                    "cholesky", 
                    "lsqr", 
                    "sparse_cg"
                ], 
                "default": "auto", 
                "help_text": "Solver to use in the computational routines:\n\n- 'auto' chooses the solver automatically based on the type of data.\n\n- 'svd' uses a Singular Value Decomposition of X to compute the Ridge\ncoefficients. More stable for singular matrices than\n'cholesky'.\n\n- 'cholesky' uses the standard scipy.linalg.solve function to\nobtain a closed-form solution.\n\n- 'sparse_cg' uses the conjugate gradient solver as found in\nscipy.sparse.linalg.cg. As an iterative algorithm, this solver is\nmore appropriate than 'cholesky' for large-scale data\n(possibility to set `tol` and `max_iter`).\n\n- 'lsqr' uses the dedicated regularized least-squares routine\nscipy.sparse.linalg.lsqr. It is the fatest but may not be available\nin old scipy versions. It also uses an iterative procedure.\n\nAll three solvers support both dense and sparse data.\n", 
                "name": "solver", 
                "type": "string"
            }, 
            {
                "default": 0.001, 
                "help_text": "Precision of the solution.\n\nAttributes\n----------", 
                "name": "tol", 
                "type": "float"
            }
        ]
    }, 
    "RidgeCV": {
        "cls": "sklearn.linear_model.ridge.RidgeCV", 
        "defaults": {
            "alphas": [
                0.1, 
                1.0, 
                10.0
            ], 
            "fit_intercept": True, 
            "normalize": False, 
            "store_cv_values": False
        }, 
        "help_text": "Ridge regression with built-in cross-validation.  By default, it performs Generalized Cross-Validation, which is a form of  efficient Leave-One-Out cross-validation.", 
        "parameters": [
            {
                "default": [
                    0.1, 
                    1.0, 
                    10.0
                ], 
                "help_text": "Array of alpha values to try.\nSmall positive values of alpha improve the conditioning of the\nproblem and reduce the variance of the estimates.\nAlpha corresponds to ``(2*C)^-1`` in other linear models such as\nLogisticRegression or LinearSVC.\n", 
                "name": "alphas", 
                "type": "list"
            }, 
            {
                "default": True, 
                "help_text": "Whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "help_text": "A string (see model evaluation documentation) or\na scorer callable object / function with signature\n``scorer(estimator, X, y)``.\n", 
                "name": "scoring", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "If None, Generalized Cross-Validation (efficient Leave-One-Out)\nwill be used.\nIf an integer is passed, it is the number of folds for KFold cross\nvalidation. Specific cross-validation objects can be passed, see\nsklearn.cross_validation module for the list of possible objects\n", 
                "name": "cv", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "choices": [
                    "auto", 
                    "svd"
                ], 
                "help_text": "Flag indicating which strategy to use when performing\nGeneralized Cross-Validation. Options are::\n", 
                "name": "gcv_mode", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": False, 
                "help_text": "Flag indicating if the cross-validation values corresponding to\neach alpha should be stored in the `cv_values_` attribute (see\nbelow). This flag is only compatible with `cv=None` (i.e. using\nGeneralized Cross-Validation).\n\nAttributes\n----------", 
                "name": "store_cv_values", 
                "type": "boolean"
            }
        ]
    }, 
    "RidgeClassifier": {
        "cls": "sklearn.linear_model.ridge.RidgeClassifier", 
        "defaults": {
            "alpha": 1.0, 
            "copy_X": True, 
            "fit_intercept": True, 
            "normalize": False, 
            "solver": "auto", 
            "tol": 0.001
        }, 
        "help_text": "Classifier using Ridge regression.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Small positive values of alpha improve the conditioning of the problem\nand reduce the variance of the estimates. Alpha corresponds to\n``(2*C)^-1`` in other linear models such as LogisticRegression or\nLinearSVC.\n", 
                "name": "alpha", 
                "type": "float"
            }, 
            {
                "help_text": "Weights associated with classes in the form", 
                "name": "class_weight", 
                "required": False, 
                "type": "dict"
            }, 
            {
                "default": True, 
                "help_text": "If True, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "Whether to calculate the intercept for this model. If set to False, no\nintercept will be used in calculations (e.g. data is expected to be\nalready centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "help_text": "Maximum number of iterations for conjugate gradient solver.\nThe default value is determined by scipy.sparse.linalg.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto", 
                    "svd", 
                    "cholesky", 
                    "lsqr", 
                    "sparse_cg"
                ], 
                "default": "auto", 
                "help_text": "Solver to use in the computational\nroutines. 'svd' will use a Singular value decomposition to obtain\nthe solution, 'cholesky' will use the standard\nscipy.linalg.solve function, 'sparse_cg' will use the\nconjugate gradient solver as found in\nscipy.sparse.linalg.cg while 'auto' will chose the most\nappropriate depending on the matrix X. 'lsqr' uses\na direct regularized least-squares routine provided by scipy.\n", 
                "name": "solver", 
                "type": "string"
            }, 
            {
                "default": 0.001, 
                "help_text": "Precision of the solution.\n\nAttributes\n----------", 
                "name": "tol", 
                "type": "float"
            }
        ]
    }, 
    "RidgeClassifierCV": {
        "cls": "sklearn.linear_model.ridge.RidgeClassifierCV", 
        "defaults": {
            "alphas": [
                0.1, 
                1.0, 
                10.0
            ], 
            "fit_intercept": True, 
            "normalize": False
        }, 
        "help_text": "Ridge classifier with built-in cross-validation.  By default, it performs Generalized Cross-Validation, which is a form of  efficient Leave-One-Out cross-validation. Currently, only the n_features >  n_samples case is handled efficiently.", 
        "parameters": [
            {
                "default": [
                    0.1, 
                    1.0, 
                    10.0
                ], 
                "help_text": "Array of alpha values to try.\nSmall positive values of alpha improve the conditioning of the\nproblem and reduce the variance of the estimates.\nAlpha corresponds to ``(2*C)^-1`` in other linear models such as\nLogisticRegression or LinearSVC.\n", 
                "name": "alphas", 
                "type": "list"
            }, 
            {
                "default": True, 
                "help_text": "Whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations\n(e.g. data is expected to be already centered).\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "If True, the regressors X will be normalized before regression.\n", 
                "name": "normalize", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "help_text": "A string (see model evaluation documentation) or\na scorer callable object / function with signature\n``scorer(estimator, X, y)``.\n", 
                "name": "scoring", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "If None, Generalized Cross-Validation (efficient Leave-One-Out)\nwill be used.\n", 
                "name": "cv", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "Weights associated with classes in the form", 
                "name": "class_weight", 
                "required": False, 
                "type": "dict"
            }
        ]
    }, 
    "SGDRegressor": {
        "cls": "sklearn.linear_model.stochastic_gradient.SGDRegressor", 
        "defaults": {
            "alpha": 0.0001, 
            "average": False, 
            "epsilon": 0.1, 
            "eta0": 0.01, 
            "fit_intercept": True, 
            "l1_ratio": 0.15, 
            "learning_rate": "invscaling", 
            "loss": "squared_loss", 
            "n_iter": 5, 
            "penalty": "l2", 
            "power_t": 0.25, 
            "shuffle": False, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "Linear model fitted by minimizing a regularized empirical loss with SGD  SGD stands for Stochastic Gradient Descent: the gradient of the loss is  estimated each sample at a time and the model is updated along the way with  a decreasing strength schedule (aka learning rate).  The regularizer is a penalty added to the loss function that shrinks model  parameters towards the zero vector using either the squared euclidean norm  L2 or the absolute norm L1 or a combination of both (Elastic Net). If the  parameter update crosses the 0.0 value because of the regularizer, the  update is truncated to 0.0 to allow for learning sparse models and achieve  online feature selection.  This implementation works with data represented as dense numpy arrays of  floating point values for the features.", 
        "parameters": [
            {
                "choices": [
                    "squared_loss", 
                    "huber", 
                    "epsilon_insensitive", 
                    "squared_epsilon_insensitive"
                ], 
                "default": "squared_loss", 
                "help_text": "The loss function to be used. Defaults to 'squared_loss' which refers\nto the ordinary least squares fit. 'huber' modifies 'squared_loss' to\nfocus less on getting outliers correct by switching from squared to\nlinear loss past a distance of epsilon. 'epsilon_insensitive' ignores\nerrors less than epsilon and is linear past that; this is the loss\nfunction used in SVR. 'squared_epsilon_insensitive' is the same but\nbecomes squared loss past a tolerance of epsilon.\n", 
                "name": "loss", 
                "type": "string"
            }, 
            {
                "choices": [
                    "none", 
                    "l2", 
                    "l1", 
                    "elasticnet"
                ], 
                "default": "l2", 
                "help_text": "The penalty (aka regularization term) to be used. Defaults to 'l2'\nwhich is the standard regularizer for linear SVM models. 'l1' and\n'elasticnet' might bring sparsity to the model (feature selection)\nnot achievable with 'l2'.\n", 
                "name": "penalty", 
                "type": "string"
            }, 
            {
                "default": 0.0001, 
                "help_text": "Constant that multiplies the regularization term. Defaults to 0.0001\n", 
                "name": "alpha", 
                "type": "float"
            }, 
            {
                "default": 0.15, 
                "help_text": "The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.\nl1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.\nDefaults to 0.15.\n", 
                "name": "l1_ratio", 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Whether the intercept should be estimated or not. If False, the\ndata is assumed to be already centered. Defaults to True.\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": 5, 
                "help_text": "The number of passes over the training data (aka epochs). The number\nof iterations is set to 1 if using partial_fit.\nDefaults to 5.\n", 
                "name": "n_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Whether or not the training data should be shuffled after each epoch.\nDefaults to False.\n", 
                "name": "shuffle", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator to use when\nshuffling the data.\n", 
                "name": "random_state", 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "The verbosity level.\n", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.1, 
                "help_text": "Epsilon in the epsilon-insensitive loss functions; only if `loss` is\n'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.\nFor 'huber', determines the threshold at which it becomes less\nimportant to get the prediction exactly right.\nFor epsilon-insensitive, any differences between the current prediction\nand the correct label are ignored if they are less than this threshold.\n", 
                "name": "epsilon", 
                "type": "float"
            }, 
            {
                "default": "invscaling", 
                "help_text": "The learning rate:", 
                "name": "learning_rate", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 0.01, 
                "help_text": "The initial learning rate [default 0.01].\n", 
                "name": "eta0", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.25, 
                "help_text": "The exponent for inverse scaling learning rate [default 0.25].\n", 
                "name": "power_t", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "When set to True, reuse the solution of the previous call to fit as\ninitialization, otherwise, just erase the previous solution.\n", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "When set to True, computes the averaged SGD weights and stores the\nresult in the coef_ attribute. If set to an int greater than 1,\naveraging will begin once the total number of samples seen reaches\naverage. So average=10 will begin averaging after seeing 10 samples.\n\nAttributes\n----------", 
                "name": "average", 
                "required": False, 
                "type": "boolean_or_integer"
            }
        ]
    }, 
    "SVC": {
        "cls": "sklearn.svm.classes.SVC", 
        "defaults": {
            "C": 1.0, 
            "cache_size": 200, 
            "coef0": 0.0, 
            "degree": 3, 
            "gamma": 0.0, 
            "kernel": "rbf", 
            "max_iter": -1, 
            "probability": False, 
            "shrinking": True, 
            "tol": 0.001, 
            "verbose": False
        }, 
        "help_text": "C-Support Vector Classification.  The implementation is based on libsvm. The fit time complexity  is more than quadratic with the number of samples which makes it hard  to scale to dataset with more than a couple of 10000 samples.  The multiclass support is handled according to a one-vs-one scheme.  For details on the precise mathematical formulation of the provided  kernel functions and how `gamma`, `coef0` and `degree` affect each,  see the corresponding section in the narrative documentation:  :ref:`svm_kernels`.  .. The narrative documentation is available at http://scikit-learn.org/", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "Penalty parameter C of the error term.\n", 
                "name": "C", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": "rbf", 
                "help_text": "Specifies the kernel type to be used in the algorithm.\nIt must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or\na callable.\nIf none is given, 'rbf' will be used. If a callable is given it is\nused to precompute the kernel matrix.\n", 
                "name": "kernel", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 3, 
                "help_text": "Degree of the polynomial kernel function ('poly').\nIgnored by all other kernels.\n", 
                "name": "degree", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.\nIf gamma is 0.0 then 1/n_features will be used instead.\n", 
                "name": "gamma", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.0, 
                "help_text": "Independent term in kernel function.\nIt is only significant in 'poly' and 'sigmoid'.\n", 
                "name": "coef0", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "Whether to enable probability estimates. This must be enabled prior\nto calling `fit`, and will slow down that method.\n", 
                "name": "probability", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "Whether to use the shrinking heuristic.\n", 
                "name": "shrinking", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 0.001, 
                "help_text": "Tolerance for stopping criterion.\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 200, 
                "help_text": "Specify the size of the kernel cache (in MB)\n", 
                "name": "cache_size", 
                "required": False, 
                "type": "float"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "help_text": "Set the parameter C of class i to class_weight[i]*C for\nSVC. If not given, all classes are supposed to have\nweight one. The 'auto' mode uses the values of y to\nautomatically adjust weights inversely proportional to\nclass frequencies.\n", 
                "name": "class_weight", 
                "required": False, 
                "type": "auto_dict"
            }, 
            {
                "default": False, 
                "help_text": "Enable verbose output. Note that this setting takes advantage of a\nper-process runtime setting in libsvm that, if enabled, may not work\nproperly in a multithreaded context.\n", 
                "name": "verbose", 
                "type": "boolean"
            }, 
            {
                "default": -1, 
                "help_text": "Hard limit on iterations within solver, or -1 for no limit.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator to use when\nshuffling the data for probability estimation.\n\nAttributes\n----------", 
                "name": "random_state", 
                "type": "integer"
            }
        ]
    }, 
    "TheilSenRegressor": {
        "cls": "sklearn.linear_model.theil_sen.TheilSenRegressor", 
        "defaults": {
            "copy_X": True, 
            "fit_intercept": True, 
            "max_iter": 300, 
            "max_subpopulation": 10000.0, 
            "n_jobs": 1, 
            "tol": 0.001, 
            "verbose": False
        }, 
        "help_text": "Theil-Sen Estimator: robust multivariate regression model.  The algorithm calculates least square solutions on subsets with size  n_subsamples of the samples in X. Any value of n_subsamples between the  number of features and samples leads to an estimator with a compromise  between robustness and efficiency. Since the number of least square  solutions is \"n_samples choose n_subsamples\", it can be extremely large  and can therefore be limited with max_subpopulation. If this limit is  reached, the subsets are chosen randomly. In a final step, the spatial  median (or L1 median) is calculated of all least square solutions.", 
        "parameters": [
            {
                "default": True, 
                "help_text": "Whether to calculate the intercept for this model. If set\nto False, no intercept will be used in calculations.\n", 
                "name": "fit_intercept", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": True, 
                "help_text": "If True, X will be copied; else, it may be overwritten.\n", 
                "name": "copy_X", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 10000.0, 
                "help_text": "Instead of computing with a set of cardinality 'n choose k', where n is\nthe number of samples and k is the number of subsamples (at least\nnumber of features), consider only a stochastic subpopulation of a\ngiven maximal size if 'n choose k' is larger than max_subpopulation.\nFor other than small problem sizes this parameter will determine\nmemory usage and runtime if n_subsamples is not changed.\n", 
                "name": "max_subpopulation", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "Number of samples to calculate the parameters. This is at least the\nnumber of features (plus 1 if fit_intercept=True) and the number of\nsamples as a maximum. A lower number leads to a higher breakdown\npoint and a low efficiency while a high number leads to a low\nbreakdown point and a high efficiency. If None, take the\nminimum number of subsamples leading to maximal robustness.\nIf n_subsamples is set to n_samples, Theil-Sen is identical to least\nsquares.\n", 
                "name": "n_subsamples", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 300, 
                "help_text": "Maximum number of iterations for the calculation of spatial median.\n", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.001, 
                "help_text": "Tolerance when calculating spatial median.\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "A random number generator instance to define the state of the\nrandom permutations generator.\n", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "Number of CPUs to use during the cross validation. If ``-1``, use\nall the CPUs.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Verbose mode when fitting the model.\n\nAttributes\n----------", 
                "name": "verbose", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "decision tree classifier": {
        "cls": "sklearn.tree.tree.DecisionTreeClassifier", 
        "defaults": {
            "criterion": "gini", 
            "min_samples_leaf": 1, 
            "min_samples_split": 2, 
            "min_weight_fraction_leaf": 0.0, 
            "splitter": "best"
        }, 
        "help_text": "A decision tree classifier.", 
        "parameters": [
            {
                "default": "gini", 
                "help_text": "The function to measure the quality of a split. Supported criteria are\n\"gini\" for the Gini impurity and \"entropy\" for the information gain.\n", 
                "name": "criterion", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": "best", 
                "help_text": "The strategy used to choose the split at each node. Supported\nstrategies are \"best\" to choose the best split and \"random\" to choose\nthe best random split.\n", 
                "name": "splitter", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "The number of features to consider when looking for the best split:\n- If int, then consider `max_features` features at each split.\n- If float, then `max_features` is a percentage and\n`int(max_features * n_features)` features are considered at each\nsplit.\n- If \"auto\", then `max_features=sqrt(n_features)`.\n- If \"sqrt\", then `max_features=sqrt(n_features)`.\n- If \"log2\", then `max_features=log2(n_features)`.\n- If None, then `max_features=n_features`.\n", 
                "name": "max_features", 
                "required": False, 
                "type": "int_float_string_none"
            }, 
            {
                "help_text": "The maximum depth of the tree. If None, then nodes are expanded until\nall leaves are pure or until all leaves contain less than\nmin_samples_split samples.\nIgnored if ``max_leaf_nodes`` is not None.\n", 
                "name": "max_depth", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2, 
                "help_text": "The minimum number of samples required to split an internal node.\n", 
                "name": "min_samples_split", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "The minimum number of samples required to be at a leaf node.\n", 
                "name": "min_samples_leaf", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "The minimum weighted fraction of the input samples required to be at a\nleaf node.\n", 
                "name": "min_weight_fraction_leaf", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "Grow a tree with ``max_leaf_nodes`` in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.\nIf not None then ``max_depth`` will be ignored.\n", 
                "name": "max_leaf_nodes", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "help_text": "(default=None)", 
                "name": "class_weight", 
                "required": False, 
                "type": "list"
            }, 
            {
                "help_text": "If int, random_state is the seed used by the random number generator;\nIf RandomState instance, random_state is the random number generator;\nIf None, the random number generator is the RandomState instance used\nby `np.random`.\n\nAttributes\n----------", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "decision tree regressor": {
        "cls": "sklearn.tree.tree.DecisionTreeRegressor", 
        "defaults": {
            "criterion": "mse", 
            "min_samples_leaf": 1, 
            "min_samples_split": 2, 
            "min_weight_fraction_leaf": 0.0, 
            "splitter": "best"
        }, 
        "help_text": "A decision tree regressor.", 
        "parameters": [
            {
                "default": "mse", 
                "help_text": "The function to measure the quality of a split. The only supported\ncriterion is \"mse\" for the mean squared error.\n", 
                "name": "criterion", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": "best", 
                "help_text": "The strategy used to choose the split at each node. Supported\nstrategies are \"best\" to choose the best split and \"random\" to choose\nthe best random split.\n", 
                "name": "splitter", 
                "required": False, 
                "type": "string"
            }, 
            {
                "help_text": "The number of features to consider when looking for the best split:\n- If int, then consider `max_features` features at each split.\n- If float, then `max_features` is a percentage and\n`int(max_features * n_features)` features are considered at each\nsplit.\n- If \"auto\", then `max_features=n_features`.\n- If \"sqrt\", then `max_features=sqrt(n_features)`.\n- If \"log2\", then `max_features=log2(n_features)`.\n- If None, then `max_features=n_features`.\n", 
                "name": "max_features", 
                "required": False, 
                "type": "int_float_string_none"
            }, 
            {
                "help_text": "The maximum depth of the tree. If None, then nodes are expanded until\nall leaves are pure or until all leaves contain less than\nmin_samples_split samples.\nIgnored if ``max_leaf_nodes`` is not None.\n", 
                "name": "max_depth", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2, 
                "help_text": "The minimum number of samples required to split an internal node.\n", 
                "name": "min_samples_split", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "The minimum number of samples required to be at a leaf node.\n", 
                "name": "min_samples_leaf", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "The minimum weighted fraction of the input samples required to be at a\nleaf node.\n", 
                "name": "min_weight_fraction_leaf", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "Grow a tree with ``max_leaf_nodes`` in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.\nIf not None then ``max_depth`` will be ignored.\n", 
                "name": "max_leaf_nodes", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "If int, random_state is the seed used by the random number generator;\nIf RandomState instance, random_state is the random number generator;\nIf None, the random number generator is the RandomState instance used\nby `np.random`.\n\nAttributes\n----------", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }
        ]
    }, 
    "extra trees classifier": {
        "cls": "sklearn.ensemble.forest.ExtraTreesClassifier", 
        "defaults": {
            "bootstrap": False, 
            "criterion": "gini", 
            "max_features": "auto", 
            "min_samples_leaf": 1, 
            "min_samples_split": 2, 
            "min_weight_fraction_leaf": 0.0, 
            "n_estimators": 10, 
            "n_jobs": 1, 
            "oob_score": False, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "An extra-trees classifier.  This class implements a meta estimator that fits a number of  randomized decision trees (a.k.a. extra-trees) on various sub-samples  of the dataset and use averaging to improve the predictive accuracy  and control over-fitting.", 
        "parameters": [
            {
                "default": 10, 
                "help_text": "The number of trees in the forest.\n", 
                "name": "n_estimators", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "gini", 
                "help_text": "The function to measure the quality of a split. Supported criteria are\n\"gini\" for the Gini impurity and \"entropy\" for the information gain.", 
                "name": "criterion", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": "auto", 
                "help_text": "The number of features to consider when looking for the best split:\n\n- If int, then consider `max_features` features at each split.\n- If float, then `max_features` is a percentage and\n`int(max_features * n_features)` features are considered at each\nsplit.\n- If \"auto\", then `max_features=sqrt(n_features)`.\n- If \"sqrt\", then `max_features=sqrt(n_features)`.\n- If \"log2\", then `max_features=log2(n_features)`.\n- If None, then `max_features=n_features`.\n", 
                "name": "max_features", 
                "required": False, 
                "type": "int_float_string_none"
            }, 
            {
                "help_text": "The maximum depth of the tree. If None, then nodes are expanded until\nall leaves are pure or until all leaves contain less than\nmin_samples_split samples.\nIgnored if ``max_leaf_nodes`` is not None.", 
                "name": "max_depth", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2, 
                "help_text": "The minimum number of samples required to split an internal node.", 
                "name": "min_samples_split", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "The minimum number of samples in newly created leaves. A split is\ndiscarded if after the split, one of the leaves would contain less then\n``min_samples_leaf`` samples.", 
                "name": "min_samples_leaf", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "The minimum weighted fraction of the input samples required to be at a\nleaf node.", 
                "name": "min_weight_fraction_leaf", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "Grow trees with ``max_leaf_nodes`` in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.\nIf not None then ``max_depth`` will be ignored.", 
                "name": "max_leaf_nodes", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Whether bootstrap samples are used when building trees.\n", 
                "name": "bootstrap", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Whether to use out-of-bag samples to estimate\nthe generalization error.\n", 
                "name": "oob_score", 
                "type": "boolean"
            }, 
            {
                "default": 1, 
                "help_text": "The number of jobs to run in parallel for both `fit` and `predict`.\nIf -1, then the number of jobs is set to the number of cores.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "If int, random_state is the seed used by the random number generator;\nIf RandomState instance, random_state is the random number generator;\nIf None, the random number generator is the RandomState instance used\nby `np.random`.\n", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "Controls the verbosity of the tree building process.\n", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, reuse the solution of the previous call to fit\nand add more estimators to the ensemble, otherwise, just fit a whole\nnew forest.\n", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto", 
                    "subsample"
                ], 
                "help_text": "", 
                "name": "class_weight", 
                "required": False, 
                "type": "list"
            }
        ]
    }, 
    "gradient boosting classifier": {
        "cls": "sklearn.ensemble.gradient_boosting.GradientBoostingClassifier", 
        "defaults": {
            "learning_rate": 0.1, 
            "loss": "deviance", 
            "max_depth": 3, 
            "min_samples_leaf": 1, 
            "min_samples_split": 2, 
            "min_weight_fraction_leaf": 0.0, 
            "n_estimators": 100, 
            "subsample": 1.0, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "Gradient Boosting for classification.  GB builds an additive model in a  forward stage-wise fashion; it allows for the optimization of  arbitrary differentiable loss functions. In each stage ``n_classes_``  regression trees are fit on the negative gradient of the  binomial or multinomial deviance loss function. Binary classification  is a special case where only a single regression tree is induced.", 
        "parameters": [
            {
                "choices": [
                    "deviance", 
                    "exponential"
                ], 
                "default": "deviance", 
                "help_text": "loss function to be optimized. 'deviance' refers to\ndeviance (= logistic regression) for classification\nwith probabilistic outputs. For loss 'exponential' gradient\nboosting recovers the AdaBoost algorithm.\n", 
                "name": "loss", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 0.1, 
                "help_text": "learning rate shrinks the contribution of each tree by `learning_rate`.\nThere is a trade-off between learning_rate and n_estimators.\n", 
                "name": "learning_rate", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 100, 
                "help_text": "The number of boosting stages to perform. Gradient boosting\nis fairly robust to over-fitting so a large number usually\nresults in better performance.\n", 
                "name": "n_estimators", 
                "type": "integer"
            }, 
            {
                "default": 3, 
                "help_text": "maximum depth of the individual regression estimators. The maximum\ndepth limits the number of nodes in the tree. Tune this parameter\nfor best performance; the best value depends on the interaction\nof the input variables.\nIgnored if ``max_leaf_nodes`` is not None.\n", 
                "name": "max_depth", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2, 
                "help_text": "The minimum number of samples required to split an internal node.\n", 
                "name": "min_samples_split", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "The minimum number of samples required to be at a leaf node.\n", 
                "name": "min_samples_leaf", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "The minimum weighted fraction of the input samples required to be at a\nleaf node.\n", 
                "name": "min_weight_fraction_leaf", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 1.0, 
                "help_text": "The fraction of samples to be used for fitting the individual base\nlearners. If smaller than 1.0 this results in Stochastic Gradient\nBoosting. `subsample` interacts with the parameter `n_estimators`.\nChoosing `subsample < 1.0` leads to a reduction of variance\nand an increase in bias.\n", 
                "name": "subsample", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "The number of features to consider when looking for the best split:\n- If int, then consider `max_features` features at each split.\n- If float, then `max_features` is a percentage and\n`int(max_features * n_features)` features are considered at each\nsplit.\n- If \"auto\", then `max_features=sqrt(n_features)`.\n- If \"sqrt\", then `max_features=sqrt(n_features)`.\n- If \"log2\", then `max_features=log2(n_features)`.\n- If None, then `max_features=n_features`.\n\nChoosing `max_features < n_features` leads to a reduction of variance\nand an increase in bias.\n", 
                "name": "max_features", 
                "required": False, 
                "type": "int_float_string_none"
            }, 
            {
                "help_text": "Grow trees with ``max_leaf_nodes`` in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.\nIf not None then ``max_depth`` will be ignored.\n", 
                "name": "max_leaf_nodes", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "An estimator object that is used to compute the initial\npredictions. ``init`` has to provide ``fit`` and ``predict``.\nIf None it uses ``loss.init_estimator``.\n", 
                "name": "init", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 0, 
                "help_text": "Enable verbose output. If 1 then it prints progress and performance\nonce in a while (the more trees the lower the frequency). If greater\nthan 1 then it prints progress and performance for every tree.\n", 
                "name": "verbose", 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, reuse the solution of the previous call to fit\nand add more estimators to the ensemble, otherwise, just erase the\nprevious solution.\n\nAttributes\n----------", 
                "name": "warm_start", 
                "type": "boolean"
            }
        ]
    }, 
    "logistic regression": {
        "cls": "sklearn.linear_model.logistic.LogisticRegression", 
        "defaults": {
            "C": 1.0, 
            "dual": False, 
            "fit_intercept": True, 
            "intercept_scaling": 1, 
            "max_iter": 100, 
            "multi_class": "ovr", 
            "penalty": "l2", 
            "solver": "liblinear", 
            "tol": 0.0001, 
            "verbose": 0
        }, 
        "help_text": "Logistic Regression (aka logit, MaxEnt) classifier.  In the multiclass case, the training algorithm uses the one-vs-rest (OvR)  scheme if the 'multi_class' option is set to 'ovr' and uses the  cross-entropy loss, if the 'multi_class' option is set to 'multinomial'.  (Currently the 'multinomial' option is supported only by the 'lbfgs' and  'newton-cg' solvers.)  This class implements regularized logistic regression using the  `liblinear` library, newton-cg and lbfgs solvers. It can handle both  dense and sparse input. Use C-ordered arrays or CSR matrices containing  64-bit floats for optimal performance; any other input format will be  converted (and copied).  The newton-cg and lbfgs solvers support only L2 regularization with primal  formulation. The liblinear solver supports both L1 and L2 regularization,  with a dual formulation only for the L2 penalty.", 
        "parameters": [
            {
                "choices": [
                    "l1", 
                    "l2"
                ], 
                "default": "l2", 
                "help_text": "Used to specify the norm used in the penalization. The newton-cg and\nlbfgs solvers support only l2 penalties.\n", 
                "name": "penalty", 
                "type": "string"
            }, 
            {
                "default": False, 
                "help_text": "Dual or primal formulation. Dual formulation is only implemented for\nl2 penalty with liblinear solver. Prefer dual=False when\nn_samples > n_features.\n", 
                "name": "dual", 
                "type": "boolean"
            }, 
            {
                "default": 1.0, 
                "help_text": "Inverse of regularization strength; must be a positive float.\nLike in support vector machines, smaller values specify stronger\nregularization.\n", 
                "name": "C", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Specifies if a constant (a.k.a. bias or intercept) should be\nadded the decision function.\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": 1, 
                "help_text": "Useful only if solver is liblinear.\nwhen self.fit_intercept is True, instance vector x becomes\n[x, self.intercept_scaling],\ni.e. a \"synthetic\" feature with constant value equals to\nintercept_scaling is appended to the instance vector.\nThe intercept becomes intercept_scaling * synthetic feature weight\nNote! the synthetic feature weight is subject to l1/l2 regularization\nas all other features.\nTo lessen the effect of regularization on synthetic feature weight\n(and therefore on the intercept) intercept_scaling has to be increased.\n", 
                "name": "intercept_scaling", 
                "type": "float"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "help_text": "Over-/undersamples the samples of each class according to the given\nweights. If not given, all classes are supposed to have weight one.\nThe 'auto' mode selects weights inversely proportional to class\nfrequencies in the training set.\n", 
                "name": "class_weight", 
                "required": False, 
                "type": "auto_dict"
            }, 
            {
                "default": 100, 
                "help_text": "Useful only for the newton-cg and lbfgs solvers. Maximum number of\niterations taken for the solvers to converge.\n", 
                "name": "max_iter", 
                "type": "integer"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator to use when\nshuffling the data.\n", 
                "name": "random_state", 
                "type": "integer"
            }, 
            {
                "choices": [
                    "newton-cg", 
                    "lbfgs", 
                    "liblinear"
                ], 
                "default": "liblinear", 
                "help_text": "Algorithm to use in the optimization problem.\n", 
                "name": "solver", 
                "type": "string"
            }, 
            {
                "default": 0.0001, 
                "help_text": "Tolerance for stopping criteria.\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "choices": [
                    "ovr", 
                    "multinomial"
                ], 
                "default": "ovr", 
                "help_text": "Multiclass option can be either 'ovr' or 'multinomial'. If the option\nchosen is 'ovr', then a binary problem is fit for each label. Else\nthe loss minimised is the multinomial loss fit across\nthe entire probability distribution. Works only for the 'lbfgs'\nsolver.\n", 
                "name": "multi_class", 
                "type": "string"
            }, 
            {
                "default": 0, 
                "help_text": "For the liblinear and lbfgs solvers set verbose to any positive\nnumber for verbosity.\n\nAttributes\n----------", 
                "name": "verbose", 
                "type": "integer"
            }
        ]
    }, 
    "random forest classifier": {
        "cls": "sklearn.ensemble.forest.RandomForestClassifier", 
        "defaults": {
            "bootstrap": True, 
            "criterion": "gini", 
            "max_features": "auto", 
            "min_samples_leaf": 1, 
            "min_samples_split": 2, 
            "min_weight_fraction_leaf": 0.0, 
            "n_estimators": 10, 
            "n_jobs": 1, 
            "oob_score": False, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "A random forest classifier.  A random forest is a meta estimator that fits a number of decision tree  classifiers on various sub-samples of the dataset and use averaging to  improve the predictive accuracy and control over-fitting.", 
        "parameters": [
            {
                "default": 10, 
                "help_text": "The number of trees in the forest.\n", 
                "name": "n_estimators", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "gini", 
                "help_text": "The function to measure the quality of a split. Supported criteria are\n\"gini\" for the Gini impurity and \"entropy\" for the information gain.", 
                "name": "criterion", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": "auto", 
                "help_text": "The number of features to consider when looking for the best split:\n\n- If int, then consider `max_features` features at each split.\n- If float, then `max_features` is a percentage and\n`int(max_features * n_features)` features are considered at each\nsplit.\n- If \"auto\", then `max_features=sqrt(n_features)`.\n- If \"sqrt\", then `max_features=sqrt(n_features)`.\n- If \"log2\", then `max_features=log2(n_features)`.\n- If None, then `max_features=n_features`.\n", 
                "name": "max_features", 
                "required": False, 
                "type": "int_float_string_none"
            }, 
            {
                "help_text": "The maximum depth of the tree. If None, then nodes are expanded until\nall leaves are pure or until all leaves contain less than\nmin_samples_split samples.\nIgnored if ``max_leaf_nodes`` is not None.", 
                "name": "max_depth", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2, 
                "help_text": "The minimum number of samples required to split an internal node.", 
                "name": "min_samples_split", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "The minimum number of samples in newly created leaves. A split is\ndiscarded if after the split, one of the leaves would contain less then\n``min_samples_leaf`` samples.", 
                "name": "min_samples_leaf", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "The minimum weighted fraction of the input samples required to be at a\nleaf node.", 
                "name": "min_weight_fraction_leaf", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "Grow trees with ``max_leaf_nodes`` in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.\nIf not None then ``max_depth`` will be ignored.", 
                "name": "max_leaf_nodes", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": True, 
                "help_text": "Whether bootstrap samples are used when building trees.\n", 
                "name": "bootstrap", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "Whether to use out-of-bag samples to estimate\nthe generalization error.\n", 
                "name": "oob_score", 
                "type": "boolean"
            }, 
            {
                "default": 1, 
                "help_text": "The number of jobs to run in parallel for both `fit` and `predict`.\nIf -1, then the number of jobs is set to the number of cores.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "If int, random_state is the seed used by the random number generator;\nIf RandomState instance, random_state is the random number generator;\nIf None, the random number generator is the RandomState instance used\nby `np.random`.\n", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "Controls the verbosity of the tree building process.\n", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, reuse the solution of the previous call to fit\nand add more estimators to the ensemble, otherwise, just fit a whole\nnew forest.\n", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "choices": [
                    "auto", 
                    "subsample"
                ], 
                "help_text": "", 
                "name": "class_weight", 
                "required": False, 
                "type": "list"
            }
        ]
    }, 
    "random forest regressor": {
        "cls": "sklearn.ensemble.forest.RandomForestRegressor", 
        "defaults": {
            "bootstrap": True, 
            "criterion": "mse", 
            "max_features": "auto", 
            "min_samples_leaf": 1, 
            "min_samples_split": 2, 
            "min_weight_fraction_leaf": 0.0, 
            "n_estimators": 10, 
            "n_jobs": 1, 
            "oob_score": False, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "A random forest regressor.  A random forest is a meta estimator that fits a number of classifying  decision trees on various sub-samples of the dataset and use averaging  to improve the predictive accuracy and control over-fitting.", 
        "parameters": [
            {
                "default": 10, 
                "help_text": "The number of trees in the forest.\n", 
                "name": "n_estimators", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "mse", 
                "help_text": "The function to measure the quality of a split. The only supported\ncriterion is \"mse\" for the mean squared error.", 
                "name": "criterion", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": "auto", 
                "help_text": "The number of features to consider when looking for the best split:\n\n- If int, then consider `max_features` features at each split.\n- If float, then `max_features` is a percentage and\n`int(max_features * n_features)` features are considered at each\nsplit.\n- If \"auto\", then `max_features=n_features`.\n- If \"sqrt\", then `max_features=sqrt(n_features)`.\n- If \"log2\", then `max_features=log2(n_features)`.\n- If None, then `max_features=n_features`.\n", 
                "name": "max_features", 
                "required": False, 
                "type": "int_float_string_none"
            }, 
            {
                "help_text": "The maximum depth of the tree. If None, then nodes are expanded until\nall leaves are pure or until all leaves contain less than\nmin_samples_split samples.\nIgnored if ``max_leaf_nodes`` is not None.", 
                "name": "max_depth", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 2, 
                "help_text": "The minimum number of samples required to split an internal node.", 
                "name": "min_samples_split", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 1, 
                "help_text": "The minimum number of samples in newly created leaves. A split is\ndiscarded if after the split, one of the leaves would contain less then\n``min_samples_leaf`` samples.", 
                "name": "min_samples_leaf", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "The minimum weighted fraction of the input samples required to be at a\nleaf node.", 
                "name": "min_weight_fraction_leaf", 
                "required": False, 
                "type": "float"
            }, 
            {
                "help_text": "Grow trees with ``max_leaf_nodes`` in best-first fashion.\nBest nodes are defined as relative reduction in impurity.\nIf None then unlimited number of leaf nodes.\nIf not None then ``max_depth`` will be ignored.", 
                "name": "max_leaf_nodes", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": True, 
                "help_text": "Whether bootstrap samples are used when building trees.\n", 
                "name": "bootstrap", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "whether to use out-of-bag samples to estimate\nthe generalization error.\n", 
                "name": "oob_score", 
                "type": "boolean"
            }, 
            {
                "default": 1, 
                "help_text": "The number of jobs to run in parallel for both `fit` and `predict`.\nIf -1, then the number of jobs is set to the number of cores.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "help_text": "If int, random_state is the seed used by the random number generator;\nIf RandomState instance, random_state is the random number generator;\nIf None, the random number generator is the RandomState instance used\nby `np.random`.\n", 
                "name": "random_state", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "Controls the verbosity of the tree building process.\n", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "When set to ``True``, reuse the solution of the previous call to fit\nand add more estimators to the ensemble, otherwise, just fit a whole\nnew forest.\n\nAttributes\n----------", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }
        ]
    }, 
    "stochastic gradient descent classifier": {
        "cls": "sklearn.linear_model.stochastic_gradient.SGDClassifier", 
        "defaults": {
            "alpha": 0.0001, 
            "average": False, 
            "epsilon": 0.1, 
            "eta0": 0.0, 
            "fit_intercept": True, 
            "l1_ratio": 0.15, 
            "learning_rate": "optimal", 
            "loss": "hinge", 
            "n_iter": 5, 
            "n_jobs": 1, 
            "penalty": "l2", 
            "power_t": 0.5, 
            "shuffle": False, 
            "verbose": 0, 
            "warm_start": False
        }, 
        "help_text": "Linear classifiers (SVM, logistic regression, a.o.) with SGD training.  This estimator implements regularized linear models with stochastic  gradient descent (SGD) learning: the gradient of the loss is estimated  each sample at a time and the model is updated along the way with a  decreasing strength schedule (aka learning rate). SGD allows minibatch  (online/out-of-core) learning, see the partial_fit method.  For best results using the default learning rate schedule, the data should  have zero mean and unit variance.  This implementation works with data represented as dense or sparse arrays  of floating point values for the features. The model it fits can be  controlled with the loss parameter; by default, it fits a linear support  vector machine (SVM).  The regularizer is a penalty added to the loss function that shrinks model  parameters towards the zero vector using either the squared euclidean norm  L2 or the absolute norm L1 or a combination of both (Elastic Net). If the  parameter update crosses the 0.0 value because of the regularizer, the  update is truncated to 0.0 to allow for learning sparse models and achieve  online feature selection.", 
        "parameters": [
            {
                "choices": [
                    "hinge", 
                    "log", 
                    "modified_huber", 
                    "squared_hinge", 
                    "perceptron", 
                    "squared_loss", 
                    "huber", 
                    "epsilon_insensitive", 
                    "squared_epsilon_insensitive"
                ], 
                "default": "hinge", 
                "help_text": "The loss function to be used. Defaults to 'hinge', which gives a\nlinear SVM.\nThe 'log' loss gives logistic regression, a probabilistic classifier.\n'modified_huber' is another smooth loss that brings tolerance to\noutliers as well as probability estimates.\n'squared_hinge' is like hinge but is quadratically penalized.\n'perceptron' is the linear loss used by the perceptron algorithm.\nThe other losses are designed for regression but can be useful in\nclassification as well; see SGDRegressor for a description.\n", 
                "name": "loss", 
                "type": "string"
            }, 
            {
                "choices": [
                    "none", 
                    "l2", 
                    "l1", 
                    "elasticnet"
                ], 
                "default": "l2", 
                "help_text": "The penalty (aka regularization term) to be used. Defaults to 'l2'\nwhich is the standard regularizer for linear SVM models. 'l1' and\n'elasticnet' might bring sparsity to the model (feature selection)\nnot achievable with 'l2'.\n", 
                "name": "penalty", 
                "type": "string"
            }, 
            {
                "default": 0.0001, 
                "help_text": "Constant that multiplies the regularization term. Defaults to 0.0001\n", 
                "name": "alpha", 
                "type": "float"
            }, 
            {
                "default": 0.15, 
                "help_text": "The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.\nl1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.\nDefaults to 0.15.\n", 
                "name": "l1_ratio", 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Whether the intercept should be estimated or not. If False, the\ndata is assumed to be already centered. Defaults to True.\n", 
                "name": "fit_intercept", 
                "type": "boolean"
            }, 
            {
                "default": 5, 
                "help_text": "The number of passes over the training data (aka epochs). The number\nof iterations is set to 1 if using partial_fit.\nDefaults to 5.\n", 
                "name": "n_iter", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": False, 
                "help_text": "Whether or not the training data should be shuffled after each epoch.\nDefaults to False.\n", 
                "name": "shuffle", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "help_text": "The seed of the pseudo random number generator to use when\nshuffling the data.\n", 
                "name": "random_state", 
                "type": "integer"
            }, 
            {
                "default": 0, 
                "help_text": "The verbosity level\n", 
                "name": "verbose", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.1, 
                "help_text": "Epsilon in the epsilon-insensitive loss functions; only if `loss` is\n'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.\nFor 'huber', determines the threshold at which it becomes less\nimportant to get the prediction exactly right.\nFor epsilon-insensitive, any differences between the current prediction\nand the correct label are ignored if they are less than this threshold.\n", 
                "name": "epsilon", 
                "type": "float"
            }, 
            {
                "default": 1, 
                "help_text": "The number of CPUs to use to do the OVA (One Versus All, for\nmulti-class problems) computation. -1 means 'all CPUs'. Defaults\nto 1.\n", 
                "name": "n_jobs", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": "optimal", 
                "help_text": "The learning rate schedule:", 
                "name": "learning_rate", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 0.0, 
                "help_text": "The initial learning rate for the 'constant' or 'invscaling'\nschedules. The default value is 0.0 as eta0 is not used by the\ndefault schedule 'optimal'.\n", 
                "name": "eta0", 
                "type": "float"
            }, 
            {
                "default": 0.5, 
                "help_text": "The exponent for inverse scaling learning rate [default 0.5].\n", 
                "name": "power_t", 
                "type": "float"
            }, 
            {
                "choices": [
                    "auto"
                ], 
                "help_text": "Preset for the class_weight fit parameter.\n\nWeights associated with classes. If not given, all classes\nare supposed to have weight one.\n\nThe \"auto\" mode uses the values of y to automatically adjust\nweights inversely proportional to class frequencies.\n", 
                "name": "class_weight", 
                "required": False, 
                "type": "auto_dict"
            }, 
            {
                "default": False, 
                "help_text": "When set to True, reuse the solution of the previous call to fit as\ninitialization, otherwise, just erase the previous solution.\n", 
                "name": "warm_start", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": False, 
                "help_text": "When set to True, computes the averaged SGD weights and stores the\nresult in the coef_ attribute. If set to an int greater than 1,\naveraging will begin once the total number of samples seen reaches\naverage. So average=10 will begin averaging after seeing 10 samples.\n\nAttributes\n----------", 
                "name": "average", 
                "required": False, 
                "type": "boolean_or_integer"
            }
        ]
    }, 
    "support vector regression": {
        "cls": "sklearn.svm.classes.SVR", 
        "defaults": {
            "C": 1.0, 
            "cache_size": 200, 
            "coef0": 0.0, 
            "degree": 3, 
            "epsilon": 0.1, 
            "gamma": 0.0, 
            "kernel": "rbf", 
            "max_iter": -1, 
            "shrinking": True, 
            "tol": 0.001, 
            "verbose": False
        }, 
        "help_text": "epsilon-Support Vector Regression.  The free parameters in the model are C and epsilon.  The implementation is based on libsvm.", 
        "parameters": [
            {
                "default": 1.0, 
                "help_text": "penalty parameter C of the error term.\n", 
                "name": "C", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.1, 
                "help_text": "epsilon in the epsilon-SVR model. It specifies the epsilon-tube\nwithin which no penalty is associated in the training loss function\nwith points predicted within a distance epsilon from the actual\nvalue.\n", 
                "name": "epsilon", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": "rbf", 
                "help_text": "Specifies the kernel type to be used in the algorithm.\nIt must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or\na callable.\nIf none is given, 'rbf' will be used. If a callable is given it is\nused to precompute the kernel matrix.\n", 
                "name": "kernel", 
                "required": False, 
                "type": "string"
            }, 
            {
                "default": 3, 
                "help_text": "degree of kernel function\nis significant only in poly, rbf, sigmoid\n", 
                "name": "degree", 
                "required": False, 
                "type": "integer"
            }, 
            {
                "default": 0.0, 
                "help_text": "kernel coefficient for rbf and poly, if gamma is 0.0 then 1/n_features\nwill be taken.\n", 
                "name": "gamma", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 0.0, 
                "help_text": "independent term in kernel function. It is only significant\nin poly/sigmoid.\n", 
                "name": "coef0", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": True, 
                "help_text": "Whether to use the shrinking heuristic.\n", 
                "name": "shrinking", 
                "required": False, 
                "type": "boolean"
            }, 
            {
                "default": 0.001, 
                "help_text": "Tolerance for stopping criterion.\n", 
                "name": "tol", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": 200, 
                "help_text": "Specify the size of the kernel cache (in MB)\n", 
                "name": "cache_size", 
                "required": False, 
                "type": "float"
            }, 
            {
                "default": False, 
                "help_text": "Enable verbose output. Note that this setting takes advantage of a\nper-process runtime setting in libsvm that, if enabled, may not work\nproperly in a multithreaded context.\n", 
                "name": "verbose", 
                "type": "boolean"
            }, 
            {
                "default": -1, 
                "help_text": "Hard limit on iterations within solver, or -1 for no limit.\n\nAttributes\n----------", 
                "name": "max_iter", 
                "required": False, 
                "type": "integer"
            }
        ]
    }
}

#### generated classifiers config end


for name, config in CLASSIFIERS.iteritems():
    if name in FULL_SUPPORT:
        config['support'] = True


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
