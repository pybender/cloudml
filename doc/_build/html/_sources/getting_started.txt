.. _introduction:

=======================================================
An introduction to building the classifier with CloudML
=======================================================


.. topic:: Section contents

    In this section, we provide a very short example how to
    learn and evaluate an classifier using CloudML. Also basic concepts and conventions would be introduced.


CloudML aims to provide a set of tools that allow building a classifier on the cloud. It consists of three components:

1. Import handler, a utility module that is responsible for feeding the trainer and the predictor with data.
2. Trainer, which receives data from the import handler and trains a classifier to produce a classification model.
3. Predictor, that uses a model produced by the trainer to predict the class of incoming requests.

Importing data
==============

.. _loading_example_dataset:

Loading an example dataset
--------------------------

In this example we will use standart `Abalone dataset <https://archive.ics.uci.edu/ml/datasets/Abalone>`_ in CSV format from the `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml>`_.

For example, we can do it using wget command::

    $wget http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data

.. _define_extraction_plan:

Defining extraction plan
------------------------

But first we have to describe the structure of the dataset, make an :ref:`xml extraction plan file <import_handlers>`.

The extraction plan for a dataset to load from a csv file is as follows:
    .. literalinclude:: _static/extract_csv.xml
        :language: xml
        :lines: 1-17

.. note::

	Full example of extraction plan could be find here: :download:`extract_csv.xml <_static/extract_csv.xml>`

.. _importing_data:

Importing the dataset
---------------------

Run import data and store dataset to abalano.json file::

	$importhandler.py extract_csv.xml -o abalone.json

	2015-07-03 06:30:16,951 - root
	                            - INFO - User-defined parameters:
	2015-07-03 06:30:16,952 - root
	                            - DEBUG - Validating schema...
	2015-07-03 06:30:16,953 - root
	                            - DEBUG - No input parameters declared
	2015-07-03 06:30:16,953 - root
	                            - INFO - Validate input parameters.
	2015-07-03 06:30:16,953 - root
	                            - INFO - Storing data to abalone.json...
	2015-07-03 06:30:16,954 - root
	                            - INFO - Processed 0 rows so far
	2015-07-03 06:30:17,019 - root
	                            - INFO - Processed 1000 rows so far
	2015-07-03 06:30:17,083 - root
	                            - INFO - Processed 2000 rows so far
	2015-07-03 06:30:17,148 - root
	                            - INFO - Processed 3000 rows so far
	2015-07-03 06:30:17,211 - root
	                            - INFO - Processed 4000 rows so far
	2015-07-03 06:30:17,222 - root
	                            - INFO - Total 4177 lines
	2015-07-03 06:30:17,223 - root
	                            - INFO - Ignored 0 lines


File :download:`abalone.json <_static/abalone.json>` contain json for each row. We should get folowing results::

	{"shucked_weight": 0.2245, "diameter": 0.365, "length": 0.455, "square": 0.043225, "whole_weight": 0.514, "sex": "M", "rings": 15, "height": 0.095}
	{"shucked_weight": 0.0995, "diameter": 0.265, "length": 0.35, "square": 0.0315, "whole_weight": 0.2255, "sex": "M", "rings": 7, "height": 0.09}
	......


Creating the model
==================

Describing features
-------------------

For create new model you should describe model features json file. You can found info about format in :ref:`Feature JSON file format<features>` chapter of this documentaion.

Firts we should define the classifier:
    .. literalinclude:: _static/features.json
        :language: json
        :lines: 3-6

The feateres are as follows:
    .. literalinclude:: _static/features.json
        :language: json
        :lines: 7-39


.. note::

	Full example we can found in :download:`feature.json <_static/features.json>`.


Training the model
------------------

Train model using command::

	trainer.py features.json -i abalone.json

Output::

	2015-07-03 18:33:46,317 - root
	                            - INFO - Using "logistic regression"
	2015-07-03 18:33:46,318 - root
	                            - WARNING - Could not determine input data file format.'json' would be used.
	2015-07-03 18:33:46,318 - root
	                            - INFO - Training the model using input file dataset.
	2015-07-03 18:33:46,527 - root
	                            - INFO - Processed 3342 lines, ignored 0 lines
	2015-07-03 18:33:46,550 - root
	                            - INFO - Extracting features for segment default ...
	2015-07-03 18:33:46,552 - root
	                            - INFO - Preparing feature square for train
	2015-07-03 18:33:46,557 - root
	                            - INFO - Preparing feature diameter for train
	2015-07-03 18:33:46,561 - root
	                            - INFO - Preparing feature whole_weight for train
	2015-07-03 18:33:46,568 - root
	                            - INFO - Preparing feature shucked_weight for train
	2015-07-03 18:33:46,571 - root
	                            - INFO - Preparing feature rings for train
	2015-07-03 18:33:46,590 - root
	                            - INFO - Training model...
	2015-07-03 18:33:46,634 - root
	                            - INFO - Number of features: 5
	2015-07-03 18:33:46,690 - root
	                            - INFO - Genarate trained model visualization
	2015-07-03 18:33:46,690 - root
	                            - INFO - Calculate feature weights for default segment
	2015-07-03 18:33:46,691 - root
	                            - INFO - Get weights for label 0
	2015-07-03 18:33:46,691 - root
	                            - INFO - Get weights for label 1
	2015-07-03 18:33:46,691 - root
	                            - INFO - Get weights for label 2
	2015-07-03 18:33:46,721 - root
	                            - INFO - Training completed...


For store trained model to file need specify -o option with file name. For example::

    $trainer.py features.json -i abalone.json -o model.dat


Testing the model
-----------------

For evaluating model we can use part of input dataset. For this we should set percent of data which will be use for test to `-tp` param. For example, if we want to use 20% of dataset for test::

	trainer.py features.json -i abalone.json -tp 20

After run this command we will get folowing info::

	......

	2015-07-03 18:33:46,921 - root
	                            - INFO - Processed 835 lines, ignored 0 lines
	2015-07-03 18:33:46,921 - root
	                            - INFO - Starting test "default" segment
	2015-07-03 18:33:46,921 - root
	                            - INFO - Extracting features for segment default ...
	2015-07-03 18:33:46,949 - root
	                            - INFO - Evaluating model...
	2015-07-03 18:33:46,989 - root
	                            - INFO - Area under ROC curve: {0: 0.87069287725025435, 1: 0.6702269847952107, 2: 0.71342276511627289}
	2015-07-03 18:33:46,996 - root
	                            - INFO - Confusion Matrix: [[1081  210   51]
	 [ 367  897  264]
	 [ 240  753  314]]
	2015-07-03 18:33:46,998 - root
	                            - INFO - ROC curve: {0: [array([ 0.        ,  0.        ,  0.        , ...,  0.99929453,
	        0.99964727,  1.        ]), array([  7.45156483e-04,   1.49031297e-03,   2.23546945e-03, ...,
	         1.00000000e+00,   1.00000000e+00,   1.00000000e+00])], 1: [array([  0.00000000e+00,   3.77500944e-04,   3.77500944e-04, ...,
	         9.99244998e-01,   9.99622499e-01,   1.00000000e+00]), array([  0.00000000e+00,   0.00000000e+00,   6.54450262e-04, ...,
	         1.00000000e+00,   1.00000000e+00,   1.00000000e+00])], 2: [array([ 0.        ,  0.        ,  0.        , ...,  0.99930314,
	        0.99965157,  1.        ]), array([  7.65110941e-04,   1.53022188e-03,   2.29533282e-03, ...,
	         1.00000000e+00,   1.00000000e+00,   1.00000000e+00])]}
	2015-07-03 18:33:47,001 - root
	                            - INFO - Accuracy: 0.548719176442 

.. note::

	When we trying to use other classifiers, for example Decision Tree classifier or Random Forest Classifier, we got match better accuracy, which is very close to 1.
	To try play with them simply use :download:`feature.json <_static/decision tree/features.json>` or :download:`feature.json <_static/random forest/features.json>`.

Predicting
==========

For predict data using existing trained model and store results to csv file::

    $predictor.py model.dat -i abalone.json -m csv

Results will be stored to result.csv file. First lines from it::

	label,0,1,2
	1,0.28701459000432328,0.40396444257495651,0.30902096742072022
	0,0.69853735998655109,0.19688865629972377,0.10457398371372523

Label column is a predicted label. Other columns are probabilities for each class.

We can build rest api service for predict using cloudml-predict.