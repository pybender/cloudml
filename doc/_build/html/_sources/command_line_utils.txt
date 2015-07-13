==================
Command line utils
==================

----------------
importhandler.py
----------------

Running the Import Handler

You can run the import handler using::

  python importhandler.py [-h] [-o output] [-d] [-U user-param]
                          [-V] path

The details of the parameters passed to importhandler.py are the following:

.. raw:: html

  <table>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>-h, --help</td>
      <td>Prints help message</td>
    </tr>
    <tr>
      <td>-V, --version</td>
      <td>Prints version message</td>
    </tr>
    <tr>
      <td>-d, --debug</td>
      <td>Adds more log output while running</td>
    </tr>
    <tr>
      <td>-o output, --output output</td>
      <td>Saves extracted data to file output as multiple JSON objects (one
      object per row).</td>
    </tr>
    <tr>
      <td>-U key=value</td>
      <td>Allows user defined parameters. The given parameters will be used to
      replace parameters in the SQL query. Can have multiple values.</td>
    </tr>
    <tr>
      <td>path</td>
      <td>Path pointing to the file containing the extraction plan.</td>
    </tr>
  </table>


----------
trainer.py
----------

You can run the trainer using::

  python trainer.py [-h] [-V] [-d] [-o output] [-w weight-file]
                    [-i input-file] [-t test-file]
                    [-e extraction-plan-file] [-I train-param]
                    [-T test-param] [--skip-test] path

The details of the parameters passed to trainer.py are the following:


.. raw:: html

  <table>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>-h, --help</td>
      <td>Prints help message</td>
    </tr>
    <tr>
      <td>-V, --version</td>
      <td>Prints version message</td>
    </tr>
    <tr>
      <td>-d, --debug</td>
      <td>Adds more log output while running</td>
    </tr>
    <tr>
      <td>-o output, --output output</td>
      <td>Saves trained model and related data to this file.</td>
    </tr>
    <tr>
      <td>-w weight-file, --weights weight-file</td>
      <td>Stores feature weights to the specified file as JSON objects. The
      weights are stored in two lists, one for positive weights (in descending
      order), and one for the negative weights (in ascending order). In case a
      feature results from a "parent" feature (i.e. when using Tfidf, count
      etc.), the name used is in the form <parent feature>.<value>.
      </td>
    </tr>
    <tr>
      <td>-i input-data, --input input-data</td>
      <td>Read train data from file 'input-data'. Input file may contain
      multiple JSON objects, each one containing the feature data for each row
      data.</td>
    </tr>
    <tr>
      <td>-t test-data, --test test-data</td>
      <td>Read test data from file 'test data'. Input file may contain multiple
      JSON objects, each one containing the feature data for each row
      data.</td>
    </tr>
    <tr>
      <td>-e extraction-plan</td>
      <td>Use the extraction plan defined in the given path. If -i has been
      defined, it will be ignored.</td>
    </tr>
    <tr>
      <td>-I key=value</td>
      <td>Allows user defined parameters. The given parameters will be used to
      replace parameters in the SQL query. Can have multiple values. Will be
      used only if flag -e is defined. These values will be used for extracting
      train data.</td>
    </tr>
    <tr>
      <td>-T key=value</td>
      <td>Same as -I, but used for extracting test data.</td>
    </tr>
    <tr>
      <td>--skip-test</td>
      <td>Skips testing phase.</td>
    </tr>
    <tr>
      <td>path</td>
      <td>Path pointing to features.json configuration file.</td>
    </tr>
  </table>

.. note::
  
  If you want to use csv as input format. Your csv file need have .csv extension.

------------
predictor.py
------------

The predictor loads a trained classifier and its configuration, and attempts to classify incoming data. Different types of input data should be allowed, including:
* file containing multiple JSON objects and
* import handler extraction plan (see Import Handler below).

Predictor also provides functionality for evaluating current trained classifier, allowing definition of evaluation function to use.

Predictor's configuration is performed entirely from the command line.

Running the predictor

You can run the predictor using::

  python predictor.py [-h] [-V] [-d] [-o output] [-m {roc,csv}]
                      [-P param] [-i] [-e extraction-plan-file]
                      [-U eval-param] path


The details of the parameters passed to predict.py are the following:

.. raw:: html

  <table>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>-h, --help</td>
      <td>Prints help message</td>
    </tr>
    <tr>
      <td>-V, --version</td>
      <td>Prints version message</td>
    </tr>
    <tr>
      <td>-d, --debug</td>
      <td>Adds more log output while running</td>
    </tr>
    <tr>
      <td>-o output, --output output</td>
      <td>Saves each row result and data to this file as a JSON object.</td>
    </tr>
    <tr>
      <td>-m method, --method method</td>
      <td>Process the results using the given methods. Current methods
      supported are:
      <ul>
        <li>roc - Area under the ROC curve.</li>
        <li>csv - Dump results as CSV. </li>
      </ul>
      </td>
    </tr>
    <tr>
      <td>-P key=value</td>
      <td>Allows passing parameters to evaluation method defined using -m.</td>
    </tr>
    <tr>
      <td>-i input-data, --input input-data</td>
      <td>Read train data from file 'input-data'. Input file may contain
      multiple JSON objects, each one containing the feature data for each row data.</td>
    </tr>
    <tr>
      <td>-e extraction-plan</td>
      <td>Use the extraction plan defined in the given path. If -i has been
      defined, it will be ignored.</td>
    </tr>
    <tr>
     <td>-U key=value</td>
      <td>Allows user defined parameters for the extraction plan. The given
      parameters will be used to replace parameters in the SQL query. Can have
      multiple values.</td>
    </tr>
    <tr>
      <td>path</td>
      <td>Path pointing to trained classifier, as saved by trainer.</td>
    </tr>
  </table>

--------------
transformer.py
--------------

You can run the train train transfromer using::

  python transformer.py [-h] [-V] [-d] [-o output]
                    [-i input-file] [-e extraction-plan-file]
                    [-I train-param] path

The details of the parameters passed to predict.py are the following:

.. raw:: html


  <table>
    <tr>
      <th>Parameter</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>-h, --help</td>
      <td>Prints help message</td>
    </tr>
    <tr>
      <td>-V, --version</td>
      <td>Prints version message</td>
    </tr>
    <tr>
      <td>-d, --debug</td>
      <td>Adds more log output while running</td>
    </tr>
    <tr>
      <td>-o output, --output output</td>
      <td>Saves trained transformer and related data to this file.</td>
    </tr>
    <tr>
      <td>-i input-data, --input input-data</td>
      <td>Read train data from file 'input-data'. Input file may contain
      multiple JSON objects, each one containing the feature data for each row
      data.</td>
    </tr>
    <tr>
      <td>-e extraction-plan</td>
      <td>Use the extraction plan defined in the given path. If -i has been
      defined, it will be ignored.</td>
    </tr>
    <tr>
      <td>-I key=value</td>
      <td>Allows user defined parameters. The given parameters will be used to
      replace parameters in the SQL query. Can have multiple values. Will be
      used only if flag -e is defined. These values will be used for extracting
      train data.</td>
    </tr>
    <tr>
      <td>path</td>
      <td>Path pointing to transformer.json configuration file.</td>
    </tr>
  </table>
