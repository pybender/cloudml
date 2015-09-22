.. _features:

========================
Feature JSON file format
========================

Features.json file includes information such as:

1. Name of the schema.
2. The classifier's configuration.
3. List of the features (with the name, type and other relevant processing instructions).
4. Generic feature types, in the event that more than one feature share the same feature type.

There are four top-level elements:

* :ref:`classifier <classifier>` - defining the configuration of the classifier to use.
* `schema-name` - a string describing the schema in the document.
* :ref:`feature-types <named_feature_types>` - a list of feature type definitions.
* :ref:`features <features_list>` - a list of features that the trainer will read from the data.


.. note::

	A full example of the features.json file can found in :download:`feature.json <_static/features.json>`.

.. _classifier:

Classifier
==========

The first section of features.json defines the configuration of the classifier to use. The available options are as follows:

* `type` : string
	Defines the Scikit-learn classifier class, which would be used to create the model.
* `params` : dict
	Classifier specific parameters.

Currently, the following types of classifiers can be used:

* :ref:`logistic regression <classifier-logistic-regression>`
* :ref:`support vector regression <classifier-support-vector-regression>`
* :ref:`stochastic gradient descent classifier <classifier-stochastic-gradient-descent-classifier>`
* :ref:`decision tree classifier <decision_tree>`
* :ref:`extra tree classifier <extra_tree>`
* :ref:`random forest classifier <random_forest>`
* :ref:`gradient boosting classifier <gradient_boosting>`


Sample classifier defition:

.. code-block:: json

   "classifier": {
	    "type": "logistic regression",
	    "params": {"penalty": "l2"}
   }


.. _named_feature_types:

Named feature types
===================

These are user-specific feature types.

Feature type definitions are a list of JSON objects. Each JSON object might
have the following keys and values:

* `name` : string
	The name of the feature type. Will be used later in the document by features so that they can reference the appropriate feature
	type.
* `type` : string
	:ref:`feature type <core_feature_types>`
* `params` : dict (optional)
	A map of parameters that might be required by the type.


Sample section defition:

.. code-block:: json

	"feature-types":[
	    {
	      "name":"str_to_timezone",
	      "type": "composite",
	      "params": {
	        "chain": [
	          { "type": "regex", "params": { "pattern": "UTC([-\\+]+\\d\\d).*"  }},
	          { "type": "int" }
	        ]
	      }
	    }
	  ]

.. _features_list:

Features
========

Features are the actual source for the trainer. A feature plan may contain at
least one feature. The definition of each feature may include the following
keys and values:

* `name` : string
	Name of the feature.
* `type` : string
	One of :ref:`feature type <core_feature_types>` or named feature type.
* `params` : dict (optional)
	A map of parameters that might be required by the type.
* `is-target-variable` : boolean (optional)
  	Can be either true or false. Default value is false. If set to true, then this feature is considered the target variable (or class) for the data.
* `transformer` : dict, optional
	Defines a transformer to use for applying to the data of this feature, in order to produce multiple features. See :ref:`transformers <feature_transformers>` for more details.
* `scaler` : dict, optional
	Defines the scaler, which allows standardized features by removing the mean and scaling to unit variance.
	See :ref:`scalers <feature_scalers>` for more details.
* `is-required` : boolean (optional)
	Defines whether or not this is a required feature. Default is true. When processing input data, a check is performed on each input "row" in order to verify if input data for the feature is empty. Data which is null or has a length equal to zero (strings, lists, dictionaries, tuples) is considered to be empty.
* `default` : same to feature.type (optional)
	Defines a default value to use if the value read is null or empty.        

.. note::
	.. raw:: html

	    Data which is null or has length equal to zero (strings, lists, dictionaries, tuples) is considered to be empty. In this case, the trainer will attempt to find a default value using the following priority:
	    <ol>
	      <li>If a default value has been defined on the feature model, it will be used</li>.
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
	            <li>date - 946684800 (January 1st, 2000)</li>
	          </ul>
	      </li>
	    </ol>


If a named feature type need to be used, the name as a `type` attribute of the feature must be set:

.. code-block:: json

	{
      "name": "tz",
      "type": "str_to_timezone"
    }


.. _core_feature_types:

Feature types defined in CloudML core
-------------------------------------

* `int`
	Converts each item to an integer. In case the value is null, the trainer checks for parameter named default. If it is set, then its value is used, otherwise 0 is used.
* `float`
	Converts each item into a float value.
* `boolean`
	Converts number to boolean. Uses python bool() function. Thus bool(0) = false, bool(null) = false, bool('') = false.
* `numeric`
	Does the same as `float`.
* `date` : params: pattern
	Parses the input value as a date using the pattern defined in parameter 'pattern'. The result is converted to a UNIX timestamp.
* `regex` : params: pattern
	Uses the regular expression defined in parameter pattern to transform the input string. Note that in case of multiple matches, only the first one is used
* `map` : params: pattern
	Looks up the input value in the directory defined by parameter 'mappings'. If there is no key in the directory equal to the input value, null is returned.
* `composite` : params: chain
	Allows applying multiple types to input data. Parameter chain defines a list of types, which are applied sequentially to the input value. For example, the first type can be a regular expression, while second, a mapping.
* `categorical_label`
	A categorical label feature is a feature that can take on one of a limited, and usually fixed, number of possible values.
* `categorical`
	Use CountVectorizer preprocessor which implements tokenization and occurrence counting.
* `text`
	Converts value to a string.

.. _feature_scalers:

Feature Scalers
---------------

Scalers allow standardized features by removing the mean and scaling to unit variance.

In the following example, the age of the contractor is standardized (which is relatively little: about 18-100 years) to the range [0, 1]:

.. code-block:: json

	{
      "name": "contractor.age",
      "type": "int",
      "scaler": {
        "params": {
          "feature_range_max": "1",
          "feature_range_min": "0"
        },
        "type": "MinMaxScaler"
      }
    }

Similarly, in the following one code block, scaling count of working hours is undertaken which could be extremely big in the [0, 1] range.

.. code-block:: json

    {
      "name": "contractor.worked_hours_count",
      "type": "int",
      "scaler": {
        "params": {
          "feature_range_max": "1",
          "feature_range_min": "0"
        },
        "type": "MinMaxScaler"
      }
    }

Following scalers are available:

* :ref:`StandardScaler <standard_scaler>`
* :ref:`MinMaxScaler <min_max_scaler>`
* :ref:`NoScaler <no_scaler>`


.. _standard_scaler:

StandardScaler
~~~~~~~~~~~~~~

Standardize features by removing the mean and scaling to unit variance.
Centering and scaling occur independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored, in order to later be used on data using the transform method.

The following parameters could be defined:

* `with_mean` : boolean, True by default
	If True, center the data before scaling. This does not work (and will raise an exception) when attempted on sparse matrices, because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in the memory.
* `with_std` : boolean, True by default
	If True, scale the data to unit variance (or equivalently, unit standard deviation).
* `copy` : boolean, optional, default True
	If False, attempt to avoid a copy and instead, undertake inplace scaling. This is not always guaranteed to work; e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.

Underlying implementation is `scikit-learn's StandardScaler <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_


.. _min_max_scaler:

MinMaxScaler
~~~~~~~~~~~~

Standardizes features by scaling each feature to a given range.
This estimator scales and translates each feature individually, such that, it is within the given range on the training set, i.e. between zero and one.

The following parameters could be defined:

* `feature_range_min` : integer, default=0
	Desired min value of transformed data range.
* `feature_range_max` : integer, default=1
	Desired max value of transformed data range.
* `copy` : boolean, optional, default True
	Set to False in order to perform inplace row normalization and avoid a copy (if the input is already a numpy array).

Underlying implementation is `scikit-learn's MinMaxScaler <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_


.. _no_scaler:

NoScaler
~~~~~~~~

For most feature types, by deafult, `MinMaxScaler` is applied to the feature, therefore if scaling does not need to be applied, `NoScaler` will need to be applied:

.. code-block:: json

    {
      "name": "feature_without_scaling",
      "type": "int",
      "scaler": {
        "type": "NoScaler"
      }
    }

.. _feature_transformers:

Feature Transformers
--------------------

Transformers allow creating multiple features from a single features. Each feature might have only one transformer. A transformer can be defined by specifying key "name" and appropriate parameters for the transformer.

For example, in relation to contractor page title data, text such as the following exists:
"I'm a machine learning enthusiast" in one record, while "Python programmer" in the other, and so on:

.. code-block:: json

    {
      "name":"title",
      "type": "text",
      "transformer":{
        "type":"Tfidf",
        "params": {
          "ngram_range_min":1,
          "ngram_range_max":1,
          "min_df":10}
      }
    }

However, in order to use this field in the model, it first requires conversion (encoding) to to numeric values. In this case, after applying the transformer, a list of features will be obtained: one for each word, which was often used in the title field of the records.
For "I'm a machine learning enthusiast", the following feature values will be applicable:

.. code-block:: python

	title.machine = 1
	title.learning = 1
	title.entusiast = 1
	title.python = 0
	title.programmer = 0
	...

Pre-trained transformers
~~~~~~~~~~~~~~~~~~~~~~~

If the pre-trained transformer feature need to be used,`type` key as pre-trained transformer name must be set:

.. code-block:: json

	{
      "transformer": {
        "type": "job-title-pretrained-transformer",
      },
      "type": "string",
      "name": "title",
      "is-required": true
    }

.. note::

	In order to train the transformer separately, :ref:`transformer.py <transformer_py>` command must be used and the transformer file must be saved to a folder.

.. note::

	To train the model, it is important to specify the `--transformer-path` parameter. This should contain the path to the folder, where pre-trained transformers are saved.


The following transformers are available:

* :ref:`Dictionary <dictionary_transformer>`
* :ref:`Count <count_transformer>`
* :ref:`Tfidf <tfidf_transformer>`
* :ref:`Lda <lda_transformer>`
* :ref:`Lsi <lsi_transformer>`
* :ref:`Ntile <ntile_transformer>`

.. _dictionary_transformer:

Dictionary
~~~~~~~~~~

Transforms lists of key-value.

The following parameters could be defined:

* `separator` 
* `sparse`

Underlying implementation is `scikit-learn's DictVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html>`_

.. _count_transformer:

Count
~~~~~

Converts text documents to a collection of string tokens and their counts.

The following parameters could be defined:

* `charset`
* `charset_error`
* `strip_accents`
* `lowercase`
* `stop_words`
* `token_pattern` 
* `analyzer` 
* `max_df` 
* `min_df` 
* `max_features` 
* `vocabulary` 
* `binary`
* `ngram_range_min`
* `ngram_range_max`

Underlying implementation is `scikit-learn's CountVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html>`_

.. _tfidf_transformer:

TF-IDT
~~~~~~~~~~

Transforms text documents to TF-IDF features.

The following parameters could be defined:

* `charset`
* `charset_error`
* `strip_accents`
* `lowercase`
* `stop_words`
* `token_pattern`
* `analyzer`
* `max_df`
* `min_df`
* `max_features`
* `vocabulary`
* `binary`
* `ngram_range_min`
* `ngram_range_max`

Underlying implementation is `scikit-learn's TfidfVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>`_

.. _lda_transformer:

LDA
~~~

Latent dirichlet allocation (LDA) is a widely-used generative model to extract atent topics from a collection of documents. Each document is modeled as a distribution over a set of topics, and each topic is modeled as a distribution over a set of keywords. The LdaModel from gensim is used as the LDA implementation.

The following parameters could be defined:

* `num_topics`
* `alpha`
* `eta`
* `distributed`
* `topic_file`

Underlying implementation is `scikit-learn's LdaVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.LdaVectorizer.html>`_

.. _lsi_transformer:

LSI
~~~

Latent semantic analysis/indexing (LSA/LSI) is a widely-used technique to analyze documents and find the underlying meaning or concepts of those documents. LSA assumes words which are similar in meaning will occur in similar pieces of text. A matrix containing word counts per document is constructed from a corpus of documents and a linear algebra technique called singular value decomposition (SVD) is used to reduce the number of words while preserving the similarity structure among documents. The LsiModel from gensim is used as the LSI implementation.

The following parameters could be defined:

* `num_topics`
* `id2word`
* `distributed`
* `onepass` power_iters
* `extra_samples`
* `topic_file`

Underlying implementation is `scikit-learn's LsiVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.LsiVectorizer.html>`_

.. _ntile_transformer:

NTILE
~~~~~

NTILE is an analytic function. It divides an ordered dataset into a number of buckets, indicated by expr and assigns the appropriate bucket number to each row. The buckets are numbered 1 through expr.

The following parameters could be defined:

* `number_tile` : integer
