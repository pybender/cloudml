===============
Getting started
===============

CloudML gives you a set of tools to build a classifier in a cloud. It consists of three components:

1. Import handler, a utility module that is responsible for feeding the trainer and the predictor with data.
2. Trainer, which receives data from the import handler and trains a classifier to produce a classification model.
3. Predictor, that uses a model produced by the trainer to predict the class of incoming requests.


Minimal example
===============

In this example we are going to measure the impact of a number of factors on the evaluation of TA performance.
First we download a dataset from the UCI Machine Learning Repository: http://archive.ics.uci.edu/ml and feed it to the import handler
    .. literalinclude:: minimal_example.py
        :language: python
        :lines: 1-20

The config file for a dataset is as follows:
    .. literalinclude:: TA_dataset_config
        :language: xml
        :lines: 1-20

For more details see :ref:`features`.

Next we need to describe features of our model.

The last two things we need to do are training and testing our model::

    # train a model


    # test a model


Finally, we are ready to make a prediction::

    # make a prediction



Describe features
=================

To create a new model you should describe model features using json. You can found info about format in :ref:`Feature JSON file format<features>` chapter of this documentaion.


Import data
===========



Train model
===========


Test model
==========

Predict
=======
