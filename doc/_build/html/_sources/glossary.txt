Glossary
========

As an interdisciplinary field, machine learning takes terminology from statistics, computer science and the various fields in which it is commonly employed. Sometimes this results in multiple terms being used to refer to the same concept, or very similar concepts.

The following is a list of terms used in the CloudML and documentation:

* `The model`  -- classifier + list of features
* `Train model` -- building the classifier: fitting the data, imported by import handler.
* `Test model`  -- evaluating the classifier
* `Predict model` -- predicting the target variable on user data.
* `Feature` -- model variable opis
* `Target Variable`  -- depended variable (y)
* `DataSet`  -- data in csv or json format could be used for train/test model. 
* `Extraction Plan`  -- describes how to import the dataset from various datasources and howto that data should be converted.
* `Classifier` -- `estimator <http://en.wikipedia.org/wiki/Estimator>`_  for classification. A Python object that implements the methods ``fit(X, y)`` and ``predict(T)`` and builds using classifier config section in features.json file.