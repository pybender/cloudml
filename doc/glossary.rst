Glossary
========

As an interdisciplinary field, machine learning takes terminology from statistics, computer science and the various fields in which it is commonly employed. Sometimes this results in multiple terms being used to refer to the same concept, or very similar concepts.

The following is a list of terms used in the CloudML and documentation:

.. glossary::

    The model
        classifier + list of features

    Train model
        building the classifier: fitting the data, imported by import handler.

    Test model
        evaluating the classifier

    Predict
        predicting the target variable on user data.

    Feature (field, variable, attribute)
        A feature is a property of an instance that may be used to determine its classification.

    Example (instance, case, record)
        A single row of data is called an example.

    Target Variable
        Target variable is the variable(feature) that is or should be the output.


    Data Set
        Data set is a data in csv or json format could be used for train/test model.

    Extraction Plan
        Extraction Plan is describes how to import the dataset from various datasources and howto that data should be converted.

    Classifier
        `estimator <http://en.wikipedia.org/wiki/Estimator>`_  for classification. A Python object that implements the methods ``fit(X, y)`` and ``predict(T)`` and builds using classifier config section in features.json file.

    Accuracy (error rate)
        The rate of correct (incorrect) predictions made by the model over a data set (cf. covera

    Confusion matrix
        A matrix showing the predicted and actual classifications. A confusion matrix is of size LxL, where L is the number of different label values. The following confusion matrix is for L=2:

		+-------------------+----------+----------+
		|actual \\ predicted| Negative | Positive |
		+-------------------+----------+----------+
		| negative          | a        | c        |
		+-------------------+----------+----------+
		| positive          | b        | d        |
		+-------------------+----------+----------+
