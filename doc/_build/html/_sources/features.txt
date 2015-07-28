.. _features:

========================
Feature JSON file format
========================

Features.json file includes information like:

1. Name of the schema.
2. The classifier's configuration.
3. List of the features (with their name, type and other processing instructions).
4. Generic feature types, in case more than one feature share the same feature type.

So there are four top-level elements:

* :ref:`classifier <classifier>` - defining the configuration of the classifier to use
* `schema-name` - a string describing the schema in the document
* :ref:`feature-types <named_feature_types>` - a list of feature type definitions
* :ref:`features <features_list>` - a list of the features that the trainer will read from the data


.. note::

	Full example of the features.json file can found in :download:`feature.json <_static/features.json>`.

.. _classifier:

Classifier
==========

The first section of features.json is used to define the configuration of the classifier to use. The available options are the following:

* `type` : string
	Defines the Scikit-learn classifier class, that would be used for creating the model.
* `params` : dict
	Classifier specific parameters.

Currently following types of classifiers could be used:

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

This is user-specific feature types.

Feature type definitions is a list of JSON objects. Each JSON object might
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
least one feature. The definition of each feature might include the following
keys and values:

* `name` : string
	name of the feature
* `type` : string
	one of :ref:`feature type <core_feature_types>` or named feature type
* `params` : dict (optional)
	A map of parameters that might be required by the type
* `is-target-variable` : boolean (optional)
  	Can be either true or false. Default value is false. If set to true, then this feature is considered the target variable (or class) for the data
* `transformer` : dict, optional
	Defines a transformer to use for applying to the data of this feature in order to produce multiple features. See :ref:`transformers <feature_transformers>` for more details.
* `scaler` : dict, optional
	Defines the scaler, that allows standartize features by removing the mean and scaling to unit variance.
	See :ref:`scalers <feature_scalers>` for more details.
* `is-required` : boolean (optional)
	Defines whether this is a required feature or not.Default is true. When processing input data, a check is performed on each input "row" to see if input data for this feature are empty. Data that are null or have length equal to zero (strings, lists, dictionaries, tuples) are considered as empty.
* `default` : same to feature.type (optional)
	Defines a default value to use if value read is null or empty        

.. note::
	.. raw:: html

	    Data that are null or have length equal to zero (strings, lists, dictionaries, tuples) are considered as empty. In this case, the trainer will try to find a default value using the following priority:
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
	            <li>date - 946684800 (January 1st, 2000)</li>
	          </ul>
	      </li>
	    </ol>


If you want to use named feature type, set it's name as `type` attribute of the feature:

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
	Converts each item to a float value.
* `boolean`
	Converts number to boolean. Uses python bool() function. Thus bool(0) = false, bool(null) = false, bool('') = false.
* `numeric`
	Does same as `float`.
* `date` : params: pattern
	Parses the input value as a date using the pattern defined in parameter 'pattern'. The result is converted to UNIX timestamp.
* `regex` : params: pattern
	Uses the regular expression defined in parameter pattern to transform the input string. Note that in case of multiple matches, only the first one is used
* `map` : params: pattern
	Looks up the input value in the directory defined by parameter 'mappings'. If there is no key in the directory equal to the input value, null is returned.
* `composite` : params: chain
	Allows applying multiple types to input data. Parameter chain defines a list of types, which are applied sequentially to the input value. For example, first type can be a regular expression, while second a mapping.
* `categorical_label`
	A categorical label feature is a feature that can take on one of a limited, and usually fixed, number of possible values.
* `categorical`
	Use CountVectorizer preprocessor which implements tokenization and occurrence counting.
* `text`
	Converts value to string.

.. _feature_scalers:

Feature Scalers
---------------

Scalers allow standartize features by removing the mean and scaling to unit variance.

In following example we standartize age of the contractor (which is a bit little: about 18-100 years) to the range [0, 1]:

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

And in this one code block we scaling count of working hours, which could be really big to the [0, 1] range.

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

Folloving scalers are available:

* :ref:`StandartScaler <standart_scaler>`
* :ref:`MinMaxScaler <min_max_scaler>`
* :ref:`NoScaler <no_scaler>`


.. _standart_scaler:

StandartScaler
~~~~~~~~~~~~~~

Standardize features by removing the mean and scaling to unit variance
Centering and scaling happen independently on each feature by computing the relevant statistics on the samples in the training set. Mean and standard deviation are then stored to be used on later data using the transform method.

You could define following parameters:

* `with_mean` : boolean, True by default
	If True, center the data before scaling. This does not work (and will raise an exception) when attempted on sparse matrices, because centering them entails building a dense matrix which in common use cases is likely to be too large to fit in memory.
* `with_std` : boolean, True by default
	If True, scale the data to unit variance (or equivalently, unit standard deviation).
* `copy` : boolean, optional, default True
	If False, try to avoid a copy and do inplace scaling instead. This is not guaranteed to always work inplace; e.g. if the data is not a NumPy array or scipy.sparse CSR matrix, a copy may still be returned.

Underlying implementation is `scikit-learn's StandartScaler <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html>`_


.. _min_max_scaler:

MinMaxScaler
~~~~~~~~~~~~

Standardizes features by scaling each feature to a given range.
This estimator scales and translates each feature individually such that it is in the given range on the training set, i.e. between zero and one.

You could define following parameters:

* `feature_range_min` : integer, default=0
	Desired min value of transformed data range.
* `feature_range_max` : integer, default=1
	Desired max value of transformed data range.
* `copy` : boolean, optional, default True
	Set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array).

Underlying implementation is `scikit-learn's MinMaxScaler <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html>`_


.. _no_scaler:

NoScaler
~~~~~~~~

By default for most feature types we applying `MinMaxScaler` to the feature. So if you don't want to apply scaling, you need to use `NoScaler`:

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

Transformers allow creating multiple features from a single one. Each feature might have only one transformer. You can define a transformer by specifying key "name" and any of the appropriate parameters for the transformer.

For example we have a contrator page title data. It's some text like
"I'm a machine learning entusiast" in the one record, "Python programmer" in the other, etc:

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

But for using this field in the model, we need to convert (encode) this to numeric values. In this case after applying the transformer we will have a list of features: one for each word, that often was used in title field of the records.
For the "I'm a machine learning entusiast" we will have following values of the features:

.. code-block:: python

	title.machine = 1
	title.learning = 1
	title.entusiast = 1
	title.python = 0
	title.programmer = 0
	...

Pretrained transformers
~~~~~~~~~~~~~~~~~~~~~~~

If you want use pretrained transformer for the feature, you need to set `type` key as pretrained transformer name:

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

	For training the transformer separately you need to use :ref:`transformer.py <transformer_py>` command and store transformer file to some folder.

.. note::

	For training the model don't forgot to specify `--transformer-path` parameter, which should contains path to the folder, where prtrained transformers are saved.


Following transformers are available:

* :ref:`Dictionary <dictionary_transformer>`
* :ref:`Count <count_transformer>`
* :ref:`Tfidf <tfidf_transformer>`
* :ref:`Lda <lda_transformer>`
* :ref:`Lsi <lsi_transformer>`
* :ref:`Ntile <ntile_transformer>`

.. _dictionary_transformer:

Dictionary
~~~~~~~~~~

Transforms lists of key-value 

You could define following parameters:

* `separator` 
* `sparse`

Underlying implementation is `scikit-learn's DictVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html>`_

.. _count_transformer:

Count
~~~~~

Converts text documents to a collection of string tokens and their counts.

You could define following parameters:

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

Tfidf
~~~~~~~~~~

Transforms text documents to TF-IDF features

You could define following parameters:

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

Lda
~~~

Latent dirichlet allocation (LDA) is a widely-used generative model to extract atent topics from a collection of documents. Each document is modeled as a distribution over a set of topics, and each topic is modeled as a distribution over a set of keywords. The LdaModel from gensim is used as the LDA implementation.

You could define following parameters:

* `num_topics`
* `alpha`
* `eta`
* `distributed`
* `topic_file`

Underlying implementation is `scikit-learn's LdaVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.LdaVectorizer.html>`_

.. _lsi_transformer:

Lsi
~~~

Latent semantic analysis/indexing (LSA/LSI) is a widely-used technique to analyze documents and find the unerlying meaning or concepts of those documents. LSA assumes that words that are close in meaning will occur in similar pieces of text. A matrix containing word counts per document is constructed from a corpus of documents and a linear algebra technique called singular value decomposition (SVD) is used to reduce the number of words while preserving the similarity structure among documents. The LsiModel from gensim is used as the LSI implementation.

You could define following parameters:

* `num_topics`
* `id2word`
* `distributed`
* `onepass` power_iters
* `extra_samples`
* `topic_file`

Underlying implementation is `scikit-learn's LsiVectorizer <http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.LsiVectorizer.html>`_

.. _ntile_transformer:

Ntile
~~~~~

NTILE is an analytic function. It divides an ordered data set into a number of buckets indicated by expr and assigns the appropriate bucket number to each row. The buckets are numbered 1 through expr.

You could define following parameters:

* `number_tile` : integer
