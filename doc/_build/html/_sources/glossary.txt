Glossary
========

Being an interdisciplinary field, machine learning borrows terminology from statistics, computer science and various fields in which it is commonly employed. Occasionally, this results in multiple terms being referred to as the same concept, or similar concepts.

The following is a list of terms used which pertain to CloudML and in the documentation:

.. glossary::

    The model
        Classifier + list of features

    Train model
        Building the classifier: fitting the data, imported by import handler.

    Test model
        Evaluating the classifier

    Predict
        Predicting the target variable on user data.

    Feature (field, variable, attribute)
        A feature is a property of an instance that may be used to determine its classification.

    Example (instance, case, record)
        A single row of data is called an example.

    Target Variable
        Target variable is the variable (feature) that is or ought to be the output.


    Data Set
        Data set is a data in csv or json format which could be used to train/test the model.

    Extraction Plan
        Extraction Plan describes how the dataset is imported from various datasources and how it should be converted.

    Classifier
        `Estimator <http://en.wikipedia.org/wiki/Estimator>`_  for classification. A Python object that implements the methods ``fit(X, y)`` and ``predict(T)`` and builds using classifier config section in features.json file.

    Accuracy (error rate)
        The rate of correct (incorrect) predictions made by the model for a data set (cf. covera

    Confusion matrix
        A matrix displaying the predicted and actual classifications. A confusion matrix is of size LxL, where L is the number of different label values. The following confusion matrix is for L=2:

		+-------------------+----------+----------+
		|actual \\ predicted| Negative | Positive |
		+-------------------+----------+----------+
		| negative          | a        | c        |
		+-------------------+----------+----------+
		| positive          | b        | d        |
		+-------------------+----------+----------+
