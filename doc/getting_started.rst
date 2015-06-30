===============
Getting started
===============

CloudML gives you a set of tools to build a classifier in a cloud. It consists of three components:

1. Import handler, a utility module that is responsible for feeding the trainer and the predictor with data.
2. Trainer, which receives data from the import handler and trains a classifier to produce a classification model.
3. Predictor, that uses a model produced by the trainer to predict the class of incoming requests.


Minimal example
===============

In this example we are going to measure the impact of a number of factors on the evaluation of the TA performance.

First we download a dataset from the `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml>`_.
For example, we can do it using Python interpreter:
    .. literalinclude:: minimal_example.py
        :language: none
        :lines: 5, 8-9, 12-13

Next we are going to load our data to PostgreSQL database we are to create.
We will use the Postgres interactive terminal called **psql**:
    .. literalinclude:: minimal_example.py
        :language: none
        :lines: 30-34

We are going to use **importhandler.py** to load the data to the model.

But first we have to describe the structure of the dataset, make an .xml *explanation plan* file.
The explanation plan for a dataset to load from a Postgres db is as follows:
    .. literalinclude:: TA_dataset_config_postgres
        :language: xml
        :lines: 1-22

Importhandler.py can be run as follows:
    .. literalinclude:: minimal_example.py
        :language: none
        :lines: 36

*(In case you want to omit a step with transferring your data to PostgreSQL)*

The explanation plan for a dataset to load from the .data file is as follows:
    .. literalinclude:: TA_dataset_config
        :language: xml
        :lines: 1-20


For more details see :ref:`import_handlers`.

Next we have to describe features relevant for our model.

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
