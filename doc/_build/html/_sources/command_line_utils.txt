==================
Command line utils
==================

Describes the command line utilities provided by CloudML.

In examples of calls used the standard `Abalone dataset <https://archive.ics.uci.edu/ml/datasets/Abalone>`_ from the `UCI Machine Learning Repository <http://archive.ics.uci.edu/ml>`_.

----------------
importhandler.py
----------------

Running the Import Handler

The import handler can be run using::

  python importhandler.py [-h] [-o output] [-d] [-U user-param]
                          [-V] path

The details of the parameters passed to importhandler.py are as follows:

+-----------------+------------------------+ 
| Parameter       | Description            | 
+=================+========================+ 
| -h, --help      | Prints help message    | 
+-----------------+------------------------+ 
| -V, --version   | Prints version message.| 
+-----------------+------------------------+ 
| -d, --debug     | Adds additional log    |
|                 | output while running   | 
+-----------------+------------------------+ 
| -o output       | Saves extracted data to| 
| --output output | file output as multiple|
|                 | JSON objects (one      |
|                 | object per row).       |
+-----------------+------------------------+ 
| -U key=value    | Allows user defined    |
|                 | parameters. The given  |
|                 | parameters will be used|
|                 | to replace parameters  |
|                 | in the SQL query. Can  |
|                 | have multiple values.  | 
+-----------------+------------------------+ 
| path            | Path pointing to the   |
|                 | file containing the    |
|                 | extraction plan.       | 
+-----------------+------------------------+ 

Example
-------

Download :download:`extract-train.xml <_static/command line/extract-train.xml>` a sample of the import handler file, which would be used for training the model.

Create the dataset to be used for training the model:

.. code-block:: console

    $ python importhandler.py extract-train.xml -o train-dataset.json

Download :download:`extract-test.xml <_static/command line/extract-test.xml>` a sample of the import handler file, which would be used for evaluating the model.

Create the dataset to be used for evaluating the model:

.. code-block:: console

    $ python importhandler.py extract-test.xml -o test-dataset.json


.. note::

  More information about building the import handlers could be found in :ref:`Getting Started <loading_example_dataset>` topic.


----------
trainer.py
----------

The trainer can be run using::

  python trainer.py [-h] [-V] [-d] [-o output] [-w weight-file]
                    [-s store-vect-file] [-v store-train-vect-file]
                    [-i input-file] [-t test-file] [-tp test-percent]
                    [-e extraction-plan-file] [-I train-param] [-T test-param]
                    [--skip-test] [--transformer-path transformer_path]
                    path

The details of the parameters passed to trainer.py are as follows:

+--------------------+---------------------------+ 
| Parameter          | Description               | 
+====================+===========================+ 
| -h, --help         | Prints help message       | 
+--------------------+---------------------------+ 
| -V, --version      | Prints version message.   | 
+--------------------+---------------------------+ 
| -d, --debug        | Adds additional log       |
|                    | output while running      | 
+--------------------+---------------------------+ 
| -o output          | Saves trained model and   |
| --output output    | related data to this      |
|                    | file.                     | 
+--------------------+---------------------------+
| -w file            | Stores feature weights    |
| --weights file     | to the specified file     |
|                    | as JSON objects. The      |
|                    | weights are stored        |
|                    | within two lists, one     |
|                    | for positive weights      |
|                    | (in descending order),    |
|                    | and one for the           |
|                    | negative weights (in      |
|                    | ascending order).         |
|                    | In case, a feature        |
|                    | results from a "parent"   |
|                    | feature (i.e. when        |
|                    | using Tfidf, count        |
|                    | etc.), the name used is   |
|                    | in the form               |
|                    | <parent feature>.<value>. | 
+--------------------+---------------------------+
| -s test-vect-file  | Store test vectorized     |
|                    | data to given file.       |
+--------------------+---------------------------+
| -v train-vect-file | Store train vectorized    |
|                    | data to given file.       |
+--------------------+---------------------------+
| -i input-data      | Read train data from file |
| --input input-data | 'input-data'. Input file  |
|                    | may contain multiple JSON |
|                    | objects, each containing  |
|                    | the feature data for each |
|                    | row data.                 |
+--------------------+---------------------------+
| -t test-data       | Read test data from file  |
| --test test-data   | 'test data'. Input file   |
|                    | may contain multiple JSON |
|                    | objects, each containing  |
|                    | the feature data for each |
|                    | row data.                 |
+--------------------+---------------------------+
| -tp test-percent   | Specify what percentage of|
|                    | the training data would be|
|                    | used for testing and this |
|                    | part of the data would be |
|                    | excluded from the training|
|                    | set and considered only in|
|                    |  the testing phase.       |
+--------------------+---------------------------+
| -e extraction-plan | Use the extraction plan   |
|                    | defined in the given path.|
|                    | If -i has been defined,   |
|                    | it will be ignored.       |
+--------------------+---------------------------+
| -I key=value       | Allows user defined       |
|                    | parameters. The given     |
|                    | parameters will be used to|
|                    | replace parameters in the |
|                    | SQL query. Can have       |
|                    | multiple values. Will be  |
|                    | used only if flag -e is   |
|                    | defined. These values will|
|                    | be used for extracting    |
|                    | train data.               |
+--------------------+---------------------------+
| -T key=value       | Same as -I, but used for  |
|                    | extracting test data.     |
+--------------------+---------------------------+
| --skip-test        | Skips testing phase.      | 
+--------------------+---------------------------+
| --transformer-path | Path to the folder with   | 
| transformer_path   | pretrained transformers.  |
+--------------------+---------------------------+
| path               | Path pointing to          |
|                    | features.json file the    |
|                    | configuration file.       | 
+--------------------+---------------------------+ 

.. note::
  
  If one is required to use csv as the input format, the csv file will be required to have a .csv extension.


Example
-------

Download :download:`feature.json <_static/command line/features.json>` file, that describes all model fields and the :ref:`classifier <supported-classifiers>` to be used.

Create the model and evaluate it using previously created train-dataset.json and test-dataset.json:

.. code-block:: console

    $ python trainer.py ./doc/_static/command\ line/features.json -i ./train-dataset.json -o model.dat -t ./test-dataset.json

The `model.dat` will be created with pickled trained model.

.. note::

  More information about create the model could be found in :ref:`Getting Started <createing-the-model>` topic.


------------
predictor.py
------------

The predictor loads a trained classifier, its configuration, and attempts to classify incoming data. Different types of input data should be allowed, including:
* file containing multiple JSON objects; and,
* import handler extraction plan (see Import Handler below).

Predictor also provides functionality for evaluating current trained classifier, allowing definition of evaluation function to be used.

Predictor's configuration is performed entirely from the command line.

The predictor can be run using::

  python predictor.py [-h] [-V] [-d] [-o output] [-m {roc,csv}]
                      [-P param] [-i] [-e extraction-plan-file]
                      [-U eval-param] path


The details of the parameters passed to predict.py are as follows:

+--------------------+---------------------------+ 
| Parameter          | Description               | 
+====================+===========================+ 
| -h, --help         | Prints help message.      | 
+--------------------+---------------------------+ 
| -V, --version      | Prints version message.   | 
+--------------------+---------------------------+ 
| -d, --debug        | Adds additional log       |
|                    | output while running      | 
+--------------------+---------------------------+ 
| -o output          | Saves each row result and |
| --output output    | data to this file as a    |
|                    | JSON object.              | 
+--------------------+---------------------------+
| -m method          | Process the results using | 
| --method method    | the given methods.        |
|                    | Supported methods:        |
|                    |  * roc - Area under the   |
|                    |    ROC curve.             |
|                    |  * csv - Dump results as  |
|                    |    CSV.                   |
+--------------------+---------------------------+
| -P key=value       | Allows passing parameters |
|                    | to evaluation method      |
|                    | defined using -m.         |
+--------------------+---------------------------+
| -i input-data      | Read the data from file   |
| --input input-data | 'input-data'. Input file  |
|                    | may contain multiple JSON |
|                    | objects, each containing  |
|                    | the feature data for each |
|                    | row data.                 |
+--------------------+---------------------------+
| -e extraction-plan | Use the extraction plan   |
|                    | defined in the given path.|
|                    | If -i has been defined,   |
|                    | it will be ignored.       |
+--------------------+---------------------------+
| -U key=value       | Allows user defined       |
|                    | parameters for the        |
|                    | extraction plan. The given|
|                    | parameters will be used to|
|                    | replace parameters in the |
|                    | SQL query. Can have       |
|                    | multiple values.          | 
+--------------------+---------------------------+
| path               | Path pointing to trained  |
|                    | classifier, as saved by   |
|                    | trainer.                  | 
+--------------------+---------------------------+

Example
-------

Download :download:`data.json <_static/command line/data.json>` with a data for making prediction.

Predict the target variable using data from `data.json` file:

.. code-block:: console

    $ python predictor.py -i data.json -m csv model.dat

The `results.csv` file would be created with probabilities for each value of the target variable.


.. _transformer_py:

--------------
transformer.py
--------------

The train transfromer can be run using::

  python transformer.py [-h] [-V] [-d] [-o output]
                    [-i input-file] [-e extraction-plan-file]
                    [-I train-param] path

The details of the parameters passed to predict.py are as follows:

+--------------------+---------------------------+ 
| Parameter          | Description               | 
+====================+===========================+ 
| -h, --help         | Prints help message.      | 
+--------------------+---------------------------+ 
| -V, --version      | Prints version message.   | 
+--------------------+---------------------------+ 
| -d, --debug        | Adds additional log       |
|                    | output while running      | 
+--------------------+---------------------------+ 
| -o output          | Saves trained transformer |
| --output output    | and  related data to this |
|                    | file.                     | 
+--------------------+---------------------------+
| -i input-data      | Read train data from file |
| --input input-data | 'input-data'. Input file  |
|                    | may contain multiple JSON |
|                    | objects, each containing  |
|                    | the feature data for each |
|                    | row data.                 |
+--------------------+---------------------------+
| -e extraction-plan | Use the extraction plan   |
|                    | defined in the given path.|
|                    | If -i has been defined,   |
|                    | it will be ignored.       |
+--------------------+---------------------------+
| -I key=value       | Allows user defined       |
|                    | parameters. The given     |
|                    | parameters will be used to|
|                    | replace parameters in the |
|                    | SQL query. Can have       |
|                    | multiple values. Will be  |
|                    | used only if flag -e is   |
|                    | defined. These values will|
|                    | be used for extracting    |
|                    | train data.               |
+--------------------+---------------------------+
| path               | Path pointing to          |
|                    | transformer.json file the |
|                    | configuration file.       | 
+--------------------+---------------------------+ 


Example
-------

Download :download:`transformer.json <_static/command line/transformer.json>` with the feature transformer specification.

Train the transformer using previously created `train-dataset.json`:

.. code-block:: console

    $ python transformer.py -i train-dataset.json -o transformers/rings-transformer.dat transformer.json

The `rings-transformer.dat` will be created with pickled transformer class.

To try the pretrained transformer in action download :download:`features-with-pretrained-transformer.json <_static/command line/features-with-pretrained-transformer.json>` file, that describes all model fields and the :ref:`classifier <supported-classifiers>` to be used. The feature `transformed-rings` use pretrained transformer `rings-transformer`.

Train the model:

.. code-block:: console

    $ python trainer.py features.json -i ./train-dataset.json -o new-model.dat -t ./test-dataset.json --transformer-path ./

The `new-model.dat` will be created with pickled trained model.