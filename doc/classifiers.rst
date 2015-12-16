=====================
Supported Classifiers
=====================

.. contents:: 
   :depth: 2

.. _classifier-logistic-regression:

Logistic Regression
-------------------

`Scikit Learn LogisticRegression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression>`_ will be used as the underlying implementation.

Sample configuration in features.json file:

.. code-block:: json

   "classifier": {
	    "type": "logistic regression",
	    "params": {"penalty": "l2"}
   }


This classifier has the following parameters:

* `penalty` : string, {l1, l2}
        Used to specify the norm used in the penalization. The newton-cg and lbfgs solvers only support l2 penalties.
* `dual` : boolean
        Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.
* `C` : float, default=1
        Inverse of regularization strength; must be a positive float. As in the case of support vector machines, smaller values specify stronger regularization.
* `fit_intercept` : boolean, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.
* `intercept_scaling` : float, default=1
        Useful only if the solver is liblinear, when self.fit_intercept is True, instance vector x becomes[x, self.intercept_scaling],i.e. a "synthetic" feature with constant value equals to intercept_scaling is appended to the instance vector. The intercept becomes intercept_scaling * synthetic feature weight. Note! the synthetic feature weight is subject to l1/l2 regularization as all other features. To lessen the effect of regularization on synthetic feature weight(and therefore on the intercept) intercept_scaling has to be increased.
* `class_weight` : 'auto' or a dictionary
        Over-/undersamples the samples of each class according to the given weights. If not specified, all classes are assumed to have a weight of one. The 'auto' mode selects weights which are inversely proportional to class frequencies in the training set.
* `max_iter` : integer
        Useful only for the newton-cg and lbfgs solvers. Maximum number of iterations taken for the solvers to converge.
* `random_state` : integer
        The seed of the pseudo random number generator to use when shuffling the data.
* `solver` : string, {newton-cg, lbfgs, liblinear}
        Algorithm for use in the optimization problem.
* `tol` : float
        Tolerance for stopping criteria.
* `multi_class` : string, {ovr, multinomial}
        Multiclass option can be either 'ovr' or 'multinomial'. If the option chosen is 'ovr', then a binary problem is fit for each label, otherwise the loss minimised is the multinomial loss fitting across the entire probability distribution. Works only for the 'lbfgs' solver.
* `verbose` : integer
        For the liblinear and lbfgs solvers set verbose to any positive number for verbosity.


.. _classifier-stochastic-gradient-descent-classifier:

Stochastic Gradient Descent Classifier
--------------------------------------

`Scikit Learn SGDClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn-linear-model-sgdclassifier>`_ will be used as the underlying implementation.

Sample configuration in features.json file:

.. code-block:: json

   "classifier": {
	    "type": "stochastic gradient descent classifier",
	    "params": {"loss": "log"}
   }


This classifier has following parameters:

* `loss` : string, {hinge, log, modified_huber, squared_hinge, perceptron, squared_loss, huber, epsilon_insensitive, squared_epsilon_insensitive}
        The loss function to be used. Defaults to 'hinge', which gives a linear SVM.The 'log' loss gives logistic regression, a probabilistic classifier.'modified_huber' is another smooth loss that brings tolerance to outliers as well as probability estimates.'squared_hinge' is like hinge but is quadratically penalized.'perceptron' is the linear loss used by the perceptron algorithm.The other losses are designed for regression but can be useful in classification as well; see SGDRegressor for a description.
* `penalty` : string, {none, l2, l1, elasticnet}
        The penalty (aka regularization term) to be used. Defaults to 'l2'which is the standard regularizer for linear SVM models. 'l1' and'elasticnet' might bring sparsity to the model (feature selection)not achievable with 'l2'.
* `alpha` : float
        Constant that multiplies the regularization term. Defaults to 0.0001
* `l1_ratio` : float
        The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1.l1_ratio=0 corresponds to L2 penalty, l1_ratio=1 to L1.Defaults to 0.15.
* `fit_intercept` : boolean
        Whether the intercept should be estimated or not. If False, the data is assumed to be already centered. Defaults to True.
* `n_iter` : integer
        The number of passes over the training data (aka epochs). The number of iterations is set to 1 if using partial_fit.Defaults to 5.
* `shuffle` : boolean
        Whether or not the training data should be shuffled after each epoch.Defaults to False.
* `random_state` : integer
        The seed of the pseudo random number generator to use when shuffling the data.
* `verbose` : string
        The verbosity level
* `epsilon` : float
        Epsilon in the epsilon-insensitive loss functions; only if`loss` is'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'.For 'huber', determines the threshold at which it becomes less important to get the prediction exactly right.For epsilon-insensitive, any differences between the current prediction and the correct label are ignored if they are less than this threshold.
* `n_jobs` : string
        The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation. -1 means 'all CPUs'. Defaults to 1.
* `learning_rate` : string
        The learning rate schedule:constant: eta = eta0optimal: eta = 1.0 / (t + t0) [default]invscaling: eta = eta0 / pow(t, power_t)where t0 is chosen by a heuristic proposed by Leon Bottou.
* `eta0` : double
        The initial learning rate for the 'constant' or 'invscaling'schedules. The default value is 0.0 as eta0 is not used by the default schedule 'optimal'.
* `power_t` : double
        The exponent for inverse scaling learning rate [default 0.5].
* `class_weight` : string
        Preset for the class_weight fit parameter.Weights associated with classes. If not given, all classes are supposed to have weight one.The "auto" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies.
* `warm_start` : boolean
        When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
* `average` : string
        When set to True, computes the averaged SGD weights and stores the result in the coef_ attribute. If set to an int greater than 1,averaging will begin once the total number of samples seen reaches average. So average=10 will begin averaging after seeing 10 samples.

.. _classifier-support-vector-regression:

Support Vector Regression
-------------------------

`Scikit Learn SVR <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn-svm-svr>`_ will be used as the underlying implementation.

Sample configuration in features.json file:

.. code-block:: json

   "classifier": {
	    "type": "support vector regression",
	    "params": {"loss": "log"}
   }


This classifier has following parameters:

* `C` : float, default=1
        penalty parameter C of the error term.
* `epsilon` : float
        epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.
* `kernel` : string, default='rbf'
        Specifies the kernel type to be used in the algorithm.It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' ora callable.If none is given, 'rbf' will be used. If a callable is given it is used to precompute the kernel matrix.
* `degree` : integer, default=3
        degree of kernel function is significant only in poly, rbf, sigmoid
* `gamma` : float
        kernel coefficient for rbf and poly, if gamma is 0.0 then 1/n_features will be taken.
* `coef0` : float
        independent term in kernel function. It is only significant in poly/sigmoid.
* `shrinking` : string, default=True
        Whether to use the shrinking heuristic.
* `tol` : float
        Tolerance for stopping criterion.
* `cache_size` : float
        Specify the size of the kernel cache (in MB)
* `verbose` : boolean
        Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work 1properly in a multithreaded context.
* `max_iter` : integer, default=-1
        Hard limit on iterations within solver, or -1 for no limit.

.. _decision_tree:

Decision Tree
-------------

`Scikit Learn Decision Tree Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html>`_ will be used as the underlying implementation.

Sample configuration in features.json file:

.. code-block:: json

   "classifier": {
	    "type": "decision tree classifier",
	    "params": {"loss": "log"}
   }


This classifier has following parameters:

* `criterion` : string, default="gini"
        The function to measure the quality of a split. Supported criteria are"gini" for the Gini impurity and "entropy" for the information gain.
* `splitter` : string, default="best"
        The strategy used to choose the split at each node. Supported strategies are "best" to choose the best split and "random" to choose the best random split.
* `max_features` : integer, default=None
        The number of features to consider when looking for the best split:- If int, then consider`max_features` features at each split.- If float, then`max_features` is a percentage and`int(max_features * n_features)` features are considered at each split.- If "auto", then`max_features=sqrt(n_features)`.- If "sqrt", then`max_features=sqrt(n_features)`.- If "log2", then`max_features=log2(n_features)`.- If None, then`max_features=n_features`.Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than``max_features`` features.
* `max_depth` : string, default=None
        The maximum depth of the tree. If None, then nodes are expanded untilall leaves are pure or until all leaves contain less than min_samples_split samples.Ignored if``max_leaf_nodes`` is not None.
* `min_samples_split` : integer, default=2
        The minimum number of samples required to split an internal node.
* `min_samples_leaf` : integer, default=1
        The minimum number of samples required to be at a leaf node.
* `min_weight_fraction_leaf` : float
        The minimum weighted fraction of the input samples required to be at aleaf node.
* `max_leaf_nodes` : string, default=None
        Grow a tree with``max_leaf_nodes`` in best-first fashion.Best nodes are defined as relative reduction in impurity.If None then unlimited number of leaf nodes.If not None then``max_depth`` will be ignored.
* `class_weight` : string
        (default=None)Weights associated with classes in the form``{class_label: weight}``.If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.The "auto" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data.For multi-output, the weights of each column of y will be multiplied.Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
* `random_state` : integer, default=None
        If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator;If None, the random number generator is the RandomState instance used by `np.random`.

.. _extra_tree:

Extra Tree
----------

`Scikit Learn ExtraTreesClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html>`_ will be used as the underlying implementation.

Sample configuration in features.json file:

.. code-block:: json

   "classifier": {
	    "type": "extra trees classifier",
	    "params": {"loss": "log"}
   }


This classifier has the following parameters:

* `n_estimators` : string, default=10
        The number of trees in the forest.
* `criterion` : string, default="gini"
        The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain. Note: this parameter is tree-specific.
* `max_features` : integer, default="auto"
        The number of features to consider when looking for the best split:- If int, then consider `max_features` features at each split.- If float, then `max_features` is a percentage and `int(max_features * n_features)` features are considered at each split. If "auto", then `max_features=sqrt(n_features)`.- If "sqrt", then `max_features=sqrt(n_features)`.- If "log2", then `max_features=log2(n_features)`.- If None, then `max_features=n_features`. Note: the search for a split does not end until, at least, one valid partition of the node samples is found, even if it requires to effectively inspect more than``max_features`` features. Note: this parameter is tree-specific.
* `max_depth` : string, default=None
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure, or until all leaves contain less than min_samples_split samples. Ignored if``max_leaf_nodes`` is not None.Note: this parameter is tree-specific.
* `min_samples_split` : string, default=2
        The minimum number of samples required to split an internal node. Note: this parameter is tree-specific.
* `min_samples_leaf` : string, default=1
        The minimum number of samples in newly created leaves.  A split is discarded if after the split, one of the leaves would contain less then``min_samples_leaf`` samples. Note: this parameter is tree-specific.
* `min_weight_fraction_leaf` : float
        The minimum weighted fraction of the input samples required to be at a leaf node. Note: this parameter is tree-specific.
* `max_leaf_nodes` : string, default=None
        Grow trees with``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None found then an unlimited number of leaf nodes. If not None then``max_depth`` will be ignored. Note: this parameter is tree-specific.
* `bootstrap` : string
        Whether bootstrap samples are used when building trees.
* `oob_score` : boolean
        Whether to use out-of-bag samples in order to estimate the generalization error.
* `n_jobs` : string, default=1
        The number of jobs running in parallel for both `fit` and `predict`. If -1, then the number of jobs are set to the number of cores.
* `random_state` : integer, default=None
        If int, random_state is the seed used by the random number generator; If Random State instance, random_state is the random number generator; If None, the random number generator is the Random State instance used by `np.random`.
* `verbose` : integer
        Controls the verbosity of the tree building process.
* `warm_start` : boolean
        When set to ``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, simply fit a whole new forest.
* `class_weight` : string
        Weights associated with classes in the form ``{class_label: weight}``. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as of the columns of y. The "auto" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data. The "subsample" mode is the same as "auto", except that weights are computed based on the bootstrap sample for every tree grown. For multi-output, the weights of each y column will be multiplied. Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

.. _random_forest:

Random Forest
-------------

`Scikit Learn RandomForestClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_ will be used as the underlying implementation.

Sample configuration in features.json file:

.. code-block:: json

   "classifier": {
	    "type": "random forest classifier",
	    "params": {"loss": "log"}
   }


This classifier has following parameters:

* `n_estimators` : string, default=10
        The number of trees in the forest.
* `criterion` : string, default="gini"
        The function to measure the quality of a split. Supported criteria are"gini" for the Gini impurity and "entropy" for the information gain.Note: this parameter is tree-specific.
* `max_features` : integer, default="auto"
        The number of features to consider when looking for the best split:- If int, then consider`max_features` features at each split.- If float, then`max_features` is a percentage and`int(max_features * n_features)` features are considered at each split.- If "auto", then`max_features=sqrt(n_features)`.- If "sqrt", then`max_features=sqrt(n_features)`.- If "log2", then`max_features=log2(n_features)`.- If None, then`max_features=n_features`.Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than``max_features`` features.Note: this parameter is tree-specific.
* `max_depth` : string, default=None
        The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.Ignored if``max_leaf_nodes`` is not None.Note: this parameter is tree-specific.
* `min_samples_split` : string, default=2
        The minimum number of samples required to split an internal node.Note: this parameter is tree-specific.
* `min_samples_leaf` : string, default=1
        The minimum number of samples in newly created leaves.  A split is discarded if after the split, one of the leaves would contain less then``min_samples_leaf`` samples.Note: this parameter is tree-specific.
* `min_weight_fraction_leaf` : float
        The minimum weighted fraction of the input samples required to be at a leaf node.Note: this parameter is tree-specific.
* `max_leaf_nodes` : string, default=None
        Grow trees with``max_leaf_nodes`` in best-first fashion.Best nodes are defined as relative reduction in impurity.If None then unlimited number of leaf nodes.If not None then``max_depth`` will be ignored.Note: this parameter is tree-specific.
* `bootstrap` : string, default=True
        Whether bootstrap samples are used when building trees.
* `oob_score` : boolean
        Whether to use out-of-bag samples to estimate the generalization error.
* `n_jobs` : string, default=1
        The number of jobs to run in parallel for both`fit` and`predict`.If -1, then the number of jobs is set to the number of cores.
* `random_state` : integer, default=None
        If int, random_state is the seed used by the random number generator;If RandomState instance, random_state is the random number generator;If None, the random number generator is the RandomState instance used by`np.random`.
* `verbose` : integer
        Controls the verbosity of the tree building process.
* `warm_start` : boolean
        When set to``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
* `class_weight` : string
        Weights associated with classes in the form``{class_label: weight}``.If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.The "auto" mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data.The "subsample" mode is the same as "auto" except that weights are computed based on the bootstrap sample for every tree grown.For multi-output, the weights of each column of y will be multiplied.Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

.. _gradient_boosting:

Gradient Boosting
-----------------

`Scikit Learn GradientBoostingClassifier <http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_ will be used as the underlying implementation.

Sample configuration in features.json file:

.. code-block:: json

   "classifier": {
	    "type": "gradient boosting classifier",
	    "params": {"loss": "log"}
   }


This classifier has the following parameters:

* `loss` : string, default='deviance'
        Loss function to be optimized. 'deviance' refers to deviance (= logistic regression) for classification with probabilistic outputs. For loss 'exponential' gradient boosting recovers the AdaBoost algorithm.
* `learning_rate` : float
        Learning rate shrinks the contribution of each tree by `learning_rate`. There is a trade-off between learning_rate and n_estimators.
* `n_estimators` : string, default=100, {int ()}
        The number of boosting stages to perform. Gradient boosting is fairly robust to over-fitting, therefore a large number usually results in better performance.
* `max_depth` : string, default=3
        Maximum depth of the individual regression estimators. The maximum depth limits the number of nodes in the tree. Tune this parameter for best performance; the best value depends on the interaction of the input variables. Ignored if``max_leaf_nodes`` is not None.
* `min_samples_split` : string, default=2
        The minimum number of samples required to split an internal node.
* `min_samples_leaf` : string, default=1
        The minimum number of samples required to be at a leaf node.
* `min_weight_fraction_leaf` : float
        The minimum weighted fraction of the input samples required to be at a leaf node.
* `subsample` : float, default=1
        The fraction of samples to be used for fitting the individual base learners. If smaller than 1.0 this results in Stochastic Gradient Boosting.`subsample` interacts with the parameter `n_estimators`.Choosing`subsample < 1.0` leads to a reduction of variance and an increase in bias.
* `max_features` : integer, default=None
        The number of features to consider when looking for the best split:- If int, then consider `max_features` features at each split.- If float, then `max_features` is a percentage and `int(max_features * n_features)` features are considered at each split.- If "auto", then`max_features=sqrt(n_features)`.- If "sqrt", then`max_features=sqrt(n_features)`.- If "log2", then`max_features=log2(n_features)`.- If None, then`max_features=n_features`. Choosing `max_features < n_features` leads to a reduction of variance and an increase in bias. Note: the search for a split does not end until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than``max_features`` features.
* `max_leaf_nodes` : string, default=None
        Grow trees with``max_leaf_nodes`` in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes. If not None then``max_depth`` will be ignored.
* `init` : string, default=None
        An estimator object that is used to compute the initial predictions.``init`` has to provide``fit`` and``predict``. If None, ``loss.init_estimator`` is used.
* `verbose` : integer
        Enable verbose output. If 1 then it prints progress and performance once in a while (the more trees the lower the frequency). If greater than 1 then it prints progress and performance for every tree.
* `warm_start` : boolean
        When set to``True``, reuse the solution of the previous call to fit and add further estimators to the ensemble, otherwise, simply erase the previous solution.

.. note::

    For the moment, it is not possible to use Gradient Boosting classifier, because it does not support sparse matrix.
