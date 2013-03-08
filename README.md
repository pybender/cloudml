# Purpose

CloudML aims to provide a set of tools that allow building a classifier on the
cloud. It consists of three components:

1. Trainer, which receives data from the import handler and trains a classifier to produce a classification model,
2. Predictor, that uses a model produced by the trainer to predict the class of incoming requests,
3. Import handler, a utility module that is responsible for feeding the trainer and the predictor with data.

Next sections describe the configuration of each of these components.

# Model Description

The predictive model in CloudML is described in JSON configuration file called
features.json. It includes information like:

1. The classifier's configuration.
2. The features (and their name).
3. The type of each feature. This might imply transformation to be done on each item of data.
4. Generic feature types, in case more than one feature share the same feature type.
5. Transformers, that allow converting features to different formats (i.e. Tfidf for converting a text feature to a matrix of TF-IDF features).

Here's an example of such a file:


    {
       "schema-name":"test",
       "classifier":{
          "type":"logistic regression",
          "penalty":"l2",
          "dual":false,
          "C":1.0,
          "fit_intercept":true,
          "intercept_scaling":1.0,
          "class_weight":null,
          "tol":null
       },
       "feature-types":[
          {
             "name":"str_to_timezone",
             "type":"composite",
             "params":{
                "chain":[
                   {
                      "type":"regex",
                      "params":{
                         "pattern":"UTC([-\\+]+\\d\\d).*"
                      }
                   },
                   {
                      "type":"int"
                   }
                ]
             }
          }
       ],
       "features":[
          {
             "name":"app_class",
             "type":"int",
             "is-target-variable":true
          },
          {
             "name":"contractor.dev_adj_score_recent",
             "type":"float",
             "is-required":true
          },
          {
             "name":"contractor.dev_is_looking",
             "type":"boolean",
             "is-required":false
          },
          {
             "name":"contractor.dev_city_timezone",
             "type":"str_to_timezone",
             "is-required":false
          },
          {
             "name":"contractor.dev_title",
             "transformer":{
                "type":"Tfidf",
                "ngram_range_min":1,
                "ngram_range_max":1,
                "min_df":3
             }
          }
       ]
    }


There are four top-level elements:

1. classifier, defining the configuration of the classifier to use.
2. schema-name, a string describing the schema in the document,
3. feature-types, a list of feature type definitions and
4. features, a list of the features that the trainer will read from the data.

## Classifier

The first section of features.json is used to define the configuration of the
classifier to use. The available options are the following:

<table>
  <tr>
    <th>Name</th>
    <th>Required</th>
    <th>Description
  </tr>
  <tr>
    <td>type</td>
    <td>Yes</td>
    <td>Currently only "logistic regression" is considered a valid value. Scipy's <a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression">LogisticRegression</a> will be used as the underlying implementation.</td>
  </tr>
  <tr>
    <td>penalty</td>
    <td>No</td>
    <td>Specifies the norm used in the penalization. Can be either 'l1' or 'l2'. Default is 'l2'.</td>
  </tr>
  <tr>
    <td>dual</td>
    <td>No</td>
    <td>Dual or primal formulation. Dual formulation is only implemented for l2 penalty. Prefer dual=False when n_samples > n_features. Default is false.</td>
  </tr>
  <tr>
    <td>C</td>
    <td>No</td>
    <td>Specifies the strength of the regularization. The smaller it is the bigger is the regularization Default 1.0.</td>
  </tr>
  <tr>
    <td>fit_intercept</td>
    <td>
      No
    </td>
    <td>
      Specifies if a constant (a.k.a. bias or intercept) should be added the
      decision function. Can be either true or false (default is True).
    </td>
  </tr>
  <tr>
    <td>intercept_scaling</td>
    <td>
      No
    </td>
    <td>
      When fit_intercept is True, instance vector x becomes [x, intercept_scaling],
      i.e. a "synthetic" feature with constant value equals to intercept_scaling is
      appended to the instance vector. The intercept becomes intercept_scaling 
      synthetic feature weight Note! the synthetic feature weight is subject to
      l1/l2 regularization as all other features. To lessen the effect of
      regularization on synthetic feature weight (and therefore on the intercept)
      intercept_scaling has to be increased. Default value is 1.
    </td>
  </tr>
  <tr>
    <td>class_weight</td>
    <td>No</td>
    <td>
      Can be either a dictionary or 'auto' (default null). Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The 'auto' mode uses the values of y to automatically adjust weights inversely proportional to class frequencies.
    </td>
  </tr>
  <tr>
    <td>tol</td>
    <td>
      No
    </td>
    <td>
      Tolerance for stopping criteria. Default value is 1E-4.
    </td>
  </tr>
</table>

## Feature types

Feature type definitions is a list of JSON objects. Each JSON object might
have the following keys and values:

<table>
  <tr>
    <th>Name</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>name</td>
    <td>Yes</td>
    <td>The name of the feature type. Will be used later in the document by features so
    that they can reference the appropriate feature type.</td>
  </tr>
  <tr>
    <td>type</td>
    <td>Yes</td>
    <td>Can be int, float, boolean, string-to-date, regex, map or composite. More
    details on these types can be found later in this document.</td>
  </tr>
  <tr>
    <td>params</td>
    <td>No</td>
    <td>A map of parameters that might be required by the type.</td>
  </tr>
</table>

## Features

Features are the actual source for the trainer. A feature plan may contain at
least one feature. The definition of each feature might include the following
keys and values:

<table>
  <tr>
    <th>Name</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>name</td>
    <td>Yes</td>
    <td>The name of the feature.</td>
  </tr>
  <tr>
    <td>type</td>
    <td>Yes</td>
    <td>Can be int, float, boolean, string-to-date, regex, map or composite.
    More details on these types can be found later in this document.</td>
  </tr>
  <tr>
    <td>params</td>
    <td>No</td>
    <td>A map of parameters that might be required by the type.</td>
  </tr>
  <tr>
    <td>is-target-variable</td>
    <td>No</td>
    <td>Can be either true or false. Default value is false. If set to true,
    then this feature is considered the target variable (or class) for the
    data. Note that if more than one target variables are defined, then only
    the last one encountered will be used. **TODO:** We might need to revise
    this logic.</td>
  </tr>
  <tr>
    <td>transformer</td>
    <td>No</td>
    <td>Defines a transformer to use for applying to the data of this feature
    in order to produce multiple features. See transformers later in this
    document for more details.</td>
  </tr>
  <tr>
    <td>is-required</td>
    <td>No</td>
    <td>Defines whether this is a required feature or not. Default is true.
    When processing input data, a check is performed on each input "row" to see
    if input data for this feature are empty. Data that are null or have length
    equal to zero (strings, lists, dictionaries, tuples) are considered as
    empty. In this case, the trainer will try to find a default value using the
    following priority:
    <ol>
      <li>If a default value has been defined on the feature model, it will be used</li>
      <li>If a transformer is defined, then the following values will be used as defaults:
        <ul>
          <li>Dictionary - empty dictionary - {}</li>
          <li>Count - empty string - ''</li>
          <li>Tfidf - empty string - ''</li>
          <li>Scale - 0.0</li>
        </ul>
      </li>
      <li>Finally, if a type is defined, then the following defaults will be used:
          <ul>
            <li>int - 0</li>
            <li>float - 0.0</li>
            <li>boolean - false</li>
            <li>string-to-date - 946684800 (January 1st, 2000)</li>
          </ul>
      </li>
    </ol>
    </td>
  </tr>
  <tr>
      <td>default</td>
      <td>No</td>
      <td>Defines a default value to use if value read is null or empty. See
      is-required for moredetails.
      </td>
  </tr>
</table>

## Predefined types

There's a variety of types that can be used for transforming the input data
while parsed. The following table describes the types and their parameters.
<table>
  <tr>
      <th>Name</th>
      <th>Required</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>int</td>
      <td>default (optional)</td>
      <td>Converts each item to an integer. In case the value is null, the
      trainer checks for parameter named default. If it is set, then its value
      is used, otherwise 0 is used.</td>
    </tr>
    <tr>
      <td>float</td>
      <td>default (optional)</td>
      <td>Converts each item to a integer. In case the value is null, the
      trainer checks for parameter named default. If it is set, then its value
      is used, otherwise 0.0 is used.</td>
    </tr>
    <tr>
      <td>boolean</td>
      <td>None</td>
      <td>Converts number to boolean. Uses python bool() function. Thus bool(0)
      = false, bool(null) = false, bool('') = false.</td>
    </tr>
    <tr>
      <td>string-to-date</td>
      <td>pattern</td>
      <td>Parses the input value as a date using the pattern defined in
      parameter 'pattern'. The result is converted to UNIX timestamp.</td>
    </tr>
    <tr>
      <td>regex</td>
      <td>pattern</td>
      <td>Uses the regular expression defined in parameter pattern to transform
      the input string. Note that in case of multiple matches, only the first
      one is used.</td>
    </tr>
    <tr>
      <td>map</td>
      <td>mappings</td>
      <td>Looks up the input value in the directory defined by parameter
      'mappings'. If there is no key in the directory equal to the input value,
      null is returned.</td>
    </tr>
    <tr>
      <td>composite</td>
      <td>chain</td>
      <td>Allows applying multiple types to input data. Parameter chain defines
      a list of types, which are applied sequentially to the input value. For
      example, first type can be a regular expression, while second a
      mapping.</td>
    </tr>
</table>

## Transformers

Transformers allow creating multiple features from a single one. Each feature
might have only one transformer (**Question: Is it possible that we need multiple transformers on the same feature?**). You can define a transformer by specifying
key "name" and any of the appropriate parameters for the transformer. The
following table contains a list of available transformers:


<table>
  <tr>
    <th>Name</th>
    <th>Required</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Dictionary</td>
    <td>separator, sparse</td>
    <td>Transforms lists of key-value mappings to vectors.</td>
  </tr>
  <tr>
    <td>Count</td>
    <td>charset, charset_error, strip_accents, lowercase, stop_words,
    token_pattern, analyzer, max_df, min_df, max_features, vocabulary,
    binary, ngram_range_min, ngram_range_max</td>
    <td>Converts text documents to a collection of string tokens and their
    counts.</td>
  </tr>
  <tr>
    <td>Tfidf</td>
    <td>charset, charset_error, strip_accents, lowercase, analyzer,
    stop_words, token_pattern, max_df, min_df, max_features, vocabulary,
    binary, use_idf, smooth_idf, sublinear_tf, ngram_range_min,
    ngram_range_max</td>
    <td>Transforms text documents to TF-IDF features.</td>
  </tr>
  <tr>
    <td>Scale</td>
    <td>copy, with_mean, with_std</td>
    <td>Scales the input data according to the given configuration.</td>
  </tr>
</table>

# Trainer

## Description

Trainer is the component that uses the configuration defined in features.json
and performs the actual training of the classifier. The trainer can also
perform testing of the generated model. The user will be able to either define
files containing data to be used for training and testing the classifier, or
specify the required parameters to invoke the import handler and retrieve the
data. More details on the import handler can be found later on this document.
The underlying implementation supporting the trainer is based on Python's
[scikit-learn](http://scikit-learn.org) package.

## Input data file format

The input data for the trainer can originate either from the import handler or
from an input file. In the first case, the trainer will simply ask for a
reference to the data, in a manner transparent to the user. However, if the
user decides to use a file for providing data to the trainer, he must use a
specific format.

The input data file is a file containing multiple JSON objects. Each JSON
object must contain at least all required features present in the feature
description file. Lines that contain invalid data will be ignored (the same
applies in case the user decides to load data using the import handler). Empty
lines are also ignored. Note that JSON objects do not need to be in the same
line.

As an example, consider the following feature model description:

    {
       "schema-name":"test",
       "classifier":{
          "type":"logistic regression"
       },
       "features":[
          {
             "name":"hire_outcome",
             "type":"int",
             "is-target-variable":true
          },
          {
             "name":"contractor.dev_adj_score_recent",
             "type":"float",
             "is-required":true
          },
          {
             "name":"contractor.dev_is_looking",
             "type":"boolean",
             "is-required":false
          },
          {
             "name":"contractor.dev_title",
             "transformer":{
                "type":"Tfidf",
                "ngram_range_min":1,
                "ngram_range_max":1,
                "min_df":3
             }
          }
       ]
    }

This file defines three required features:

1. contractor.dev_adj_score_recent, defined as required explicitly,
2. contractor.dev_title, as default value for is-required is true, and
3. hire_outcome, as this is the target variable used to find current row's class.

Consider the following input data:

    {
       "hire_outcome":1,
       "contractor.dev_adj_score_recent":3.2,
       "contractor.dev_is_looking":"1",
       "contractor.dev_title":"Python engineer working full time"
    }
    {
       "hire_outcome":0,
       "contractor.dev_is_looking":"1",
       "contractor.dev_title":"Senior Python developer"
    }
    {
       "contractor.dev_adj_score_recent":4,
       "contractor.dev_title":"PHP engineer front-end"
    }
    {
       "hire_outcome":1,
       "contractor.dev_adj_score_recent":3,
       "contractor.dev_is_looking":"1",
       "contractor.dev_title":"best python engineer you've ever heard of",
       "contractor.dev_recent_hours_fp":12
    }


The first object will be used. The second one will be ignored, because data
for required feature contractor.dev_adj_score_recent are missing. The third
object will be also ignored, because value for target variable is not defined.
Finally, the fourth object will be used, but contractor.dev_recent_hours_fp
will be ignored, as it was not defined in feature model description.

The same format can be used for test data.

## Running the trainer

You can run the trainer using:

python trainer.py [-h] [-V] [-d] [-o output] [-w weight-file] [-i input-file]
                                    [-t test-file] [-e extraction-plan-file] [-I train-param]
                                    [-T test-param] [--skip-test]
                                    path

The details of the parameters passed to trainer.py are the following:

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

# Predictor

## Description

The predictor loads a trained classifier and its configuration, and attempts to
classify incoming data. Different types of input data should be allowed,
including:
* file containing multiple JSON objects and
* import handler extraction plan (see Import Handler below).

Predictor also provides functionality for evaluating current trained classifier,
allowing definition of evaluation function to use.

Predictor's configuration is performed entirely from the command line.

## Running the predictor

You can run the predictor using:

python predictor.py [-h] [-V] [-d] [-o output] [-m {roc,csv}] [-P param] [-i]
                                   [-e extraction-plan-file] [-U eval-param]
                                   path


The details of the parameters passed to predict.py are the following:

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
      multiple JSON objects, each one containing the feature data for each row
      data.</td>
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

As an example, this command:
{code}
python -mpredictor.predictor -d -e conf/extract.json -U start=2012-01-01 -U end=2012-12-31 -m csv -P fields=application_id,opening_id -P out=data.csv trainer.pickle
{code}

will do the following:
* Turn debug mode on.
* Use the extraction plan (see Import Handler later in this document) stored in file conf/extract.json for extracting input data.
* Pass parameters {start: 2012-01-01, end: 2012-12-31} to the extraction plan.
* Output results in CSV file data.csv, adding fields application_id, opening_id.
* Use classifier configuration stored in trainer.pickle file.

### ROC

This method calculates the area under the ROC curve, and prints it on the
screen.

No parameters need to be defined when using area under the ROC curve for
processing the results of the predictor.

### CSV

CSV method allows saving results in a CSV file. Each row of the resulting file
will be in the following format:

{code}
field1,field2,...,fieldN,{label},prob1,prob2,...,probN
{code}

* Fields can be any field from the original set passed to the predictor either
from extracting data or by providing them as a file.
* Label will be added in the row only if the input data contain the label for
*each row.
* Probabilities are the results from the classifier.

The following configuration parameters can be passed using the -P flag. They
must be in the form key=value

<table>
  <tr>
    <th>Key</th>
    <th>Value description</th>
  </tr>
  <tr>
    <td>out</td>
    <td>The name of the file to store CSV results.</td>
  </tr>
  <tr>
    <td>fields</td>
    <td>A CSV string containing the name of the fields in the input dataset
    that the user would like to be added in each row.</td>
  </tr>
</table>

# Import Handler

## Scope

Import handler aims to offer a flexible way for retrieving data from various
data sources. First version of the import handler will offer the required
functionality to extract data from single SQL queries. An extraction plan is
used to describe database connection details, the query to execute, and
instructions on which data should be extracted from the query and in which
manner. Data can be - optionally - stored in a file as a set of JSON objects.

Future versions of the import handler will add functionality such as:

1. Subqueries (running additional queries for each row of the parent query),
2. HTTP and/or RESTful datasources.

## Extraction plan

The extraction plan is a configuration file describing the data to be
extracted. It is a JSON object, with a specific structure. An example of such
file is the following:

    {
       "target-schema":"bestmatch",
       "datasource":[
          {
             "name":"odw",
             "type":"sql",
             "db":{
                "conn":"host='localhost' dbname='odw' user='postgres' password='postgres'",
                "vendor":"postgres"
             }
          }
       ],
       "queries":[
          {
             "name":"retrieve",
             "sql":"SELECT qi.*, 'accept' as hire_outcome FROM public.ja_quick_info qi where qi.file_provenance_date >= '%(start)s' AND qi.file_provenance_date < '%(end)s';",
             "items":[
                {
                   "source":"hire_outcome",
                   "process-as":"string",
                   "target-features":[
                      {
                         "name":"hire_outcome"
                      }
                   ]
                },
                {
                   "source":"employer_info",
                   "process-as":"json",
                   "target-features":[
                      {
                         "name":"employer.op_timezone",
                         "jsonpath":"$.op_timezone"
                      },
                      {
                         "name":"employer.op_country_tz",
                         "jsonpath":"$.op_country_tz"
                      },
                      {
                         "name":"employer.op_tot_jobs_filled",
                         "jsonpath":"$.op_tot_jobs_filled"
                      },
                      {
                         "name":"employer.country",
                         "jsonpath":"$.country"
                      }
                   ]
                },
                {
                   "source":"contractor_info",
                   "process-as":"json",
                   "target-features":[
                      {    
                         "name":"contractor.skills",
                         "jsonpath":"$.skills.*.skl_name",
                         "to-csv":true
                      },
                      {
                         "name":"tsexams",
                         "jsonpath":"$.tsexams",
                         "key-path":"$.ts_name",
                         "value-path":"$.ts_score"
                      },
                      {
                         "name":"contractor.dev_is_looking",
                         "jsonpath":"$.dev_is_looking"
                      },
                      {
                         "name":"contractor.dev_recent_rank_percentile",
                         "jsonpath":"$.dev_recent_rank_percentile"
                      },
                      {
                         "name":"contractor.dev_recent_fp_jobs",
                         "jsonpath":"$.dev_recent_fp_jobs"
                      },
                      {
                         "name":"contractor.dev_blurb",
                         "jsonpath":"$.dev_blurb"
                      },
                      {
                         "name":"contractor.dev_country",
                         "jsonpath":"$.dev_country"
                      }
                   ]
                },
                {
                   "name":"country_pair",
                   "process-as":"expression",
                   "target-features":[
                      {
                         "name":"country_pair",
                         "expression":"%(employer.country)s,%(contractor.country)s"
                      }
                   ]
                }
             ]
          }
       ]
    }

### Datasource

The first part of the configuration contains information about the database to
connect to in order to execute the query defined later. It may contain one map
with the following fields:

<table>
    <tr>
      <th>Name</th>
      <th>Required</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>name</td>
      <td>No (**Note: In the future this should be required so that queries can refer to which DB
connection to use for executing.**)</td>
      <td>A name uniquely identifying this database.</td>
    </tr>
    <tr>
      <td>type</td>
      <td>No (**Note: should be required in the future or use default sql and offer HTTP/csv
etc for other query methods.**)</td>
      <td>Currently only 'sql' is supported.</td>
    </tr>
    <tr>
      <td>db.conn</td>
      <td>Yes</td>
      <td>This is field 'conn' defined A connection string containing the DB
      connection details.</td>
    </tr>
    <tr>
      <td>db.vendor</td>
      <td>Yes</td>
      <td>The name of the database's vendor. Currently only 'postgres' is
      supported.</td>
    </tr>
</table>

### Queries

The 'queries' section contains an array of objects describing each individual
query. Currently only a single query is supported. Each query might contain
the following fields:

<table>
    <tr>
      <th>Name</th>
      <th>Required</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>name</td>
      <td>No</td>
      <td>A name uniquely identifying this query.</td>
    </tr>
    <tr>
      <td>sql [f]</td>
      <td>Yes</td>
      <td>The SQL query to execute. It may contain parameters to be replaced by
      user input (i.e. either coming from a HTTP request or command line
      option). These parameters must be in the form %(name)s.</td>
    </tr>
    <tr>
      <td>items</td>
      <td>Yes</td>
      <td>An array of objects describing which items (and how) to extract from
      each row in the query's result. The possible types of items is described
      below.</td>
    </tr>
</table>

Three types of query items are supported so far:

1. string query items, that read the value from a field defined in the SQL query, and store it to a single item,
2. JSON query items, that parse the data from a field and allow extracting multiple items using JSONPath expressions, and
3. expression query items, which use data from other query items to produce new items.

String query items can define the following fields:

<table>
    <tr>
      <th>Name</th>
      <th>Required</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>source</td>
      <td>Yes</td>
      <td>The name of the SQL query's field that contains the value to
      use.</td>
    </tr>
    <tr>
      <td>process-as [f]</td>
      <td>Yes</td>
      <td>Must be "string".</td>
    </tr>
    <tr>
      <td>target-features</td>
      <td>Yes</td>
      <td>A list of the features to extract. The resulting data will include a
      feature with name the target-feature's name, and value the value of the
      SQL query's field defined in source. Note that only the first feature is
      used (**Note: Should we allow creating multiple features from a single SQL
      field?**). If no target features are defined, the configuration is
      considered invalid.</td>
    </tr>
</table>

JSON query items may define the following fields:

<table>
    <tr>
      <th>Name</th>
      <th>Required</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>source</td>
      <td>Yes</td>
      <td>The name of the SQL query's field that contains the value to
      use.</td>
    </tr>
    <tr>
      <td>process-as [f]</td>
      <td>Yes</td>
      <td>Must be "json".</td>
    </tr>
    <tr>
      <td>is-required</td>
      <td>No</td>
      <td>Can be either true or false. Default value is true. Whether we
      require that resulting value is not null or empty string. In case it is
      set to true and no data is defined, the line will be ignored.[g]</td>
    </tr>
    <tr>
      <td>target-features</td>
      <td>Yes</td>
      <td>A list of the features to extract.</td>
    </tr>
</table>

In case of JSON query items, the target-feature objects may contain the
following fields:

<table>
    <tr>
      <th>Name</th>
      <th>Required</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>name</td>
      <td>Yes</td>
      <td>The name of the resulting feature</td>
    </tr>
    <tr>
      <td>jsonpath [f]</td>
      <td>Yes</td>
      <td>A JSONPath expression that dictates the location of the value.</td>
    </tr>
    <tr>
      <td>to-csv</td>
      <td>No</td>
      <td>Can be either true or false. If it is set to true and the result of
      the JSONPath expression is a list, it will be converted to a CSV value
      using ',' as a separator.</td>
    </tr>
    <tr>
      <td>key-path, value-path</td>
      <td>No</td>
      <td>Both should contain a JSONPath expression. If both are defined, then
      a dictionary will be created as a value. key-path defines the expression
      for the dictionary's keys, while value-path defines the expression for
      the dictionary's value. Note that those two expressions are executed not
      on the entire JSON object, but on the part resulting from applying the
      expression in jsonpath.</td>
    </tr>
</table>

Finally, expression query items may contain the following fields:

<table>
    <tr>
      <th>Name</th>
      <th>Required</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>process-as</td>
      <td>Yes</td>
      <td>Must be set to 'expression'.</td>
    </tr>
    <tr>
      <td>target-features</td>
      <td>Yes</td>
      <td>A list of target features.</td>
    </tr>
</table>

In this case, target features may contain the following fields:

<table>
    <tr>
      <th>Name</th>
      <th>Required</th>
      <th>Description</th>
    </tr>
    <tr>
      <td>name</td>
      <td>Yes</td>
      <td>The name of the resulting feature</td>
    </tr>
    <tr>
      <td>expression</td>
      <td>Yes</td>
      <td>A string expression describing how the resulting value will be
      formatted. The string may include parameters in the format %(name)s.
      Possible values for name might be any feature extracted in previous query
      items. If any of the input parameters is not set, then the result will be
      null (**Note: Discuss this**).</td>
    </tr>
</table>

## Running the Import Handler

You can run the import handler using:

python importhandler.py [-h] [-o output] [-d] [-U user-param] [-V] path

The details of the parameters passed to importhandler.py are the following:

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
