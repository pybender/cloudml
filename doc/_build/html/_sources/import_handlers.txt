.. _import_handlers:

==========================
Import Handler file format
==========================

.. contents:: 
   :depth: 3

There are two different methods for importing data:

* **Import Handler** is used for importing data from different data sources while training and testing the model. Now, CSV files, HTTP with JSON or XML data, database and some others data sources are also supported.
* **Online Import Handler** is used for importing data and feeding them to a trained classifier in order to get result.

Both handlers may be required to perform logic transforming processes on the
import, in order to produce the final results.

Although both handlers prepare data for the same classification model, their formats vary.

Top level element
=================

The Top level element is `<plan>`. There are no attributes expected for this element. Plan may contain the following elements:

- author (one or zero).
- version (one or zero).
- :ref:`script <script>` (any).
- :ref:`inputs <inputs>` (one or zero).
- :ref:`datasources <datasources>` (exactly one).
- :ref:`import <import>` (exactly one).
- :ref:`predict <predict>` (should be present for Online Import Handler).

.. _script:

Script
======

A `script` element is used to define python functions that can be
used to transform data. Code inside the script tag will be added
whenever a python function is called. Wrapping
scripts in <![CDATA[ ...]]> elements is recommend.

Example:

.. code-block:: xml

    <script>
        <![CDATA[
            def intToBoolean(a):
                return a == 1
        ]]>
    </script>

It is also possible to reference external Python files. This can be
undertaken to ease development. Scripts should be expected in the same
directory as the XML file.

Example:

.. code-block:: xml

    <script src="functions.py" />

.. note::

    Functionality for scripts from external python files has not been implemented yet.

.. _inputs:

Inputs
======

Tag `<inputs>` groups all input parameters required to execute the import handler. Input parameters are defined in `<param>` tags.

Each param may have one of the following attributes:

- `name` : string
    the name of the parameter.
- `type` : {integer, boolean, string, float, date}, optional
    the type of the input parameter. If omitted, it should be considered a string.
- `format` : string, optional
    formating instructions for the parameter (i.e. date format etc).
- `regex` : regular expression string, optional
    a regular expression that can be used to validate input parameter values.

.. note::

    Format can be applied only to the date input parameter using python's `strptime` method. Further details on format string can be found in 
    `python documetation <https://docs.python.org/2/library/datetime.html#strftime-and-strptime-behavior>`_

Example:

.. code-block:: xml

    <inputs>
        <!-- Define an integer parameter. Use regex to dictate only positive integers -->
        <param name="application" type="integer" regex="\d+" />

        <!-- Date parameter with instructions on how to interpret date-->
        <param name="year" type="date" format="%Y"/>

        <!-- Boolean parameter -->
        <param name="only_fjp" type="boolean" />
    </inputs>

.. _datasources:

Datasources
===========

Data is fed to the system using various data sources. The `<datasources>` part of the handler contains the connection details.

Datasources may be:

- :ref:`Database connections <db_datasource>`
- :ref:`CSV files <csv_datasource>`
- :ref:`HTTP GET/POST <http_datasource>`
- :ref:`Hadoop with Pig <pig_datasource>`
- :ref:`Input params <input_datasource>`

Datasources are identified by their unique names and can be accessed at any point in the file. Each datasource uses a different tag for configuration.

.. _db_datasource:

Database connections
--------------------

Database connections can be defined either by directly inserting the
connection details, or by referencing a named connection. In both cases,
the element used is `<db>`.

The following are possible attributes:

- `name` : string
    a unique name for this datasource.
- `name-ref` : string, optional
    a reference to the named connection (not currently supported).
- `host` : string
    the name of host to connect to.
- `dbname` : string, optional
    the database name.
- `user` : string, optional
    the username to use for connecting to the database.
- `password` : string, optional
    the password to use for connecting to the database.
- `port` : int, optional
    the port number to connect to at the server host.
- `vendor` : string, {postgres}
    the DB's vendor. Only `postgres` vendor is currently supported.


Note that name is required in both cases. For named connections, only name-ref should be present. When defining the DB connection details in handler's file, host, dbname and vendor should be present.

Examples:

.. code-block:: xml

    <db name="odw" host="localhost" dbname="odw" user="postgres"
        password="postgres" vendor="postgres" />


.. _csv_datasource:

CSV files
---------

A CSV file can be used for importing data from local files. It is possible
to reuse headers from CSV files, or define aliases for the column names
in the import handler.

The related tag is `csv`, and the possible attributes are:

- `name` : string
    a unique name for this data source.
- `src` : string
    the path to the CSV file.

Header information can be defined by adding child `<header>` elements
to the `<csv>` element. Each `<header>` element must exactly contain
two fields:

- `name` : string
    the name of the column.
- `index` : integer
    the column's index (columns are zero-indexed).

Examples:

.. code-block:: xml

    <!-- Defines a CSV datasource with headers in file. -->
    <csv name="csvDataSource1" src="stats.csv" />

    <!-- Defines a CSV datasource with headers in handler. -->
    <csv name="csvDataSource2" src="stats.csv">
        <!-- Note that some columns are ignored -->
        <header name="id" index="0" />
        <header name="name" index="2" />
        <header name="score" index="7" />
    </csv>

.. note::
    
    When importing data, if the CSV file does not contain a column with the index specified in <header> tag, users will receive a `ImportHandlerException`.
    For example, this exception in `csvDataSource2` datasource (declared up in the document) will be received if the `stats.csv` file has six columns.

.. _http_datasource:

HTTP
----

HTTP requests are used for importing JSON data from remote HTTP
services.

The tag used for defining them is `<http>`, and the possible attributes are as follows:

- `name` : string
    a unique name for this datasource.
- `method` : {GET, POST, PUT, DELETE}, default=GET
    the HTTP method to use.
- `url` : string
    the base URL to use.


When using this datasource with RESTful services, try to define the base
URL. If specific entities need to be queried, the query
parameters can be defined at a later stage during the import phase:

.. code-block:: xml

    <plan>
        <inputs>
            <param name="opening_id" type="string"/>
        </inputs>
        <datasources>
            <http name="jar" method="GET" url="http://service.com:11000/jar/" />
        </datasources>
        <import>
            <entity datasource="jar" name="opening">
                <query><![CDATA[#{opening_id}.json]]></query>
                <field jsonpath="$.op_title" name="opening.title" type="string"/>
                <field jsonpath="$.op_job" name="opening.description" type="string"/>
                ...
            </entity>
        </import>
    </plan>

In this case, when importing data, the system will query http://service.com:11000/jar/1.json url (User sets user paramer `opening_id` as 1, when running importhandler.py command).

.. _pig_datasource:

Pig
---

Pig is a tool for analyzing large data sets based on Hadoop. Pig Latin
is the language which allows querying and/or transformation of the data. A Pig
datasource is a connection to a remote Hadoop/Pig cluster. It is defined
using `<pig>` tag. Possible attributes are as follows:

- `name` : string
    A unique name for this data source.
- `jobid` : string, optional
    Define job flow id, if one is required to use the existing cluster.
- `amazon_access_token` : string
    By default use cloudml-control api keys.
- `amazon_token_secret` : string
    By default use cloudml-control api keys.
- `ami_version` : string, optional
    Amazon Machine Image (AMI) version to use for instances.
    `Supported ami and pig versions <http://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/Pig_SupportedVersions.html>`_.
- `bucket_name` : string, optional
    Amazon S3 bucket name for saving results, logs, etc.
- `ec2_keyname` : string, optional
    EC2 key used for the start instances, by default use cloudml-control keypair.
- `keep_alive` : boolean, optional
    Denotes whether or not the cluster should stay alive upon completion.
- `hadoop_params` : string, optional
    This attribute can be used to specify cluster-wide Hadoop settings. If it attribute isn't setted, *s3://elasticmapreduce/bootstrap-actions/configure-hadoop* script will not run.
    `More details in Configure Hadoop Bootstrap Action block <http://docs.aws.amazon.com/ElasticMapReduce/latest/DeveloperGuide/emr-plan-bootstrap.html>`_.
- `num_instances` : integer, optional
    Number of instances in the Hadoop cluster.
- `master_instance_type` : string, optional
    EC2 instance type of the master node.
    `List of Amazon EC2 Instance types <http://aws.amazon.com/ec2/instance-types/>`_.
- `slave_instance_type` : string, optional
    EC2 instance type of the slave nodes.

Example:

.. code-block:: xml

    <pig name="pig" amazon_access_token="token"         bucket_name="the_bucket" amazon_token_secret="secret" master_instance_type="c3.4xlarge" slave_instance_type="c3.4xlarge" num_instances="2" hadoop_params="-m,mapreduce.map.java.opts=-Xmx864m,-m,mapreduce.reduce.java.opts=-Xmx1536m,-m,mapreduce.map.memory.mb=1024,-m,mapreduce.reduce.memory.mb=2048,-y,yarn.nodemanager.resource.memory-mb=18226"/>


Pig query
~~~~~~~~~

Query tag of the entity with a pig datasource could have the following attributes:

- `target` : string optional
    Name of target dataset that will be stored.
- `autoload_sqoop_dataset` : string, optional
    When it is true, sqoop dataset will be auto loaded in the pig script (without defining loading statement in script). Required to define `sqoop_dataset_name` attr.
- `sqoop_dataset_name` : string, optional
    Variable name that would be used in the pig script for sqoop results, when `autoload_sqoop_dataset` is set.

There are two methods for storing pig results:

* The target attribute must be specified in the <query> tag of the entity with pig data source. In this case, results would be stored as JsonStorage:

.. code-block:: sql

    <query target="result">
        <![CDATA[
            ...
            result = FOREACH B GENERATE application, opening;
        ]]>
    </query>

*  '$output' parameter as output dir should be used. For example:

.. code-block:: sql

    <query>
        <![CDATA[
            C = FOREACH B GENERATE application, opening;
            STORE C INTO '$output' USING JsonStorage();
        ]]>
    </query>

.. _input_datasource:

Input
-----

.. note:: Input data source only using online import handlers.

If Post data is required to be used, input params need to be specified. Following this, access will be granted to them by using ``input``
datasource with the query ``any``. For example:

.. code-block:: xml

    <plan>      
        <inputs>      
           <param name="rate" type="float"/>       
           <param name="title" type="string"/>     
        </inputs>     
        <datasources/>        
        <import>      
            <entity datasource="input" name="Test_input">       
                <query>any</query>        
                <field column="title" name="dev_title" type="string"/>   
                <field column="rate" name="rate" type="float"/>       
            </entity>     
        </import>          
    </plan>

Another example with processing json input parameter is as follows:

.. code-block:: xml

    <plan>
        <inputs>
            <param name="contractor_info" type="string"/>
        </inputs>
        <datasources/>
        <import>
            <entity datasource="input" name="contractor_info">
                <query>contractor_info</query>
                <field jsonpath="$.dev_recno" name="contractor.id" type="string"/>
                <field jsonpath="$.dev_region" name="contractor.region" type="string"/>
            </entity>
        </import>
    </plan>

.. _import:

Import
======

Once the data sources have been defined, the import handler is required to be defined in relation to how 
data from each data source input will be translated. This is undertaken within the
`<import>` element. In order to be able to understand how the mapping
is undertaken, the concept of entity needs to be introduced.

An entity model's data is derived from various data sources. i.e. an entity
may describe the data being derived from a database table or view. Each
entity is associated with a datasource and (possibly) some query
parameters. For example, a database entity might use a SQL query, while
an HTTP entity might add some path and query parameters to the
data source's URL. An entity describes multiple entity "instances". i.e.
if an entity describes a database table, an entity "instance" describes
a row in the database.

An entity is defined using the `<entity>` tag. The possible attributes
of the element are as follows:

- `name` : string
    a unique name to identify the entity.
- `datasource` : string
    the datasource to use for importing data.
- `query` : string, optional
    a string that provides instructions on how to query a datasource (i.e. a SQL query or a path template). Queries can be also defined as child elements (to be discussed later).
- `autoload_fields` : boolean, optional
    when set, fields are unable to be defined. These loaded from the pig results. 
  
.. note::

    `autoload_fields` works only with `pig` data sources currently.

.. note::

    If `autoload_fields` are set, declared entity fields would be overridden with automaticaly created fields by parsing the result data row. If the field is declared, that is not present in the row,
    it would not be deleted.

Examples:

.. code-block:: xml

    <!-- An entity that uses a DB connection -->
    <entity name="employer" datasource="mysqlConn" query="SELECT * FROM table">
        ...
    </entity>

    <!-- An entity that uses an HTTP datasource -->
    <entity name="employer" datasource="odr" query="opening/f/#{opening}.json">
        ...
    </entity>


Queries
-------

The first possible child of a `<entity>` is a query. This can be used
to improve readability of the XML file and replace the query attribute
of the entity. It is also useful if the query does not return data, but
actually triggers data calculation. Examples of such cases include
running a set of SQL queries that create tables or execute a Pig
script. In this case, attribute `target` is required to be defined inside
the `<query>` tag. The value of this attribute provides details on
where to look for the actual data.

Examples:

.. code-block:: xml

    <!-- An entity that uses a DB connection -->
    <entity name="employer" datasource="mysqlConn">
        <query>
            <![CDATA[
                SELECT *
                FROM table t1 JOIN table t2 ON t1.id = t2.reference
                WHERE t2.creation_time < '#{start_date}'
            ]]>
        </query>
        ...
    </entity>

    <!-- An entity that uses an HTTP datasource -->
    <entity name="employer" datasource="json_ds">
     <query>
            <![CDATA[
                opening/f/#{opening}.json
            ]]>
        </query>
        ...
    </entity>


Query strings depend on the data source:

- :ref:`Database connections <db_datasource>`  requires SQL queries.
- :ref:`CSV datasources <csv_datasource>` do not support queries.
- :ref:`HTTP datasources <http_datasource>` can add values to end of the path. 
- :ref:`Hadoop with Pig datasource <pig_datasource>` - requires pig script.
- :ref:`Input params <input_datasource>` 

It is possible to use variables in queries using the notation `#{variable}`. This will be replaced by an input parameter with the name equal to the variable.


Fields
------

Fields are used to define how data is extracted from each entity
"instance". They are defined using the `<field>` tag, and can define
the following attributes:

- `name` : string
    a unique name for the field.
- `column` : string
    if entity is using a DB or CSV data source, it will use data from this column.
- `jsonpath` : string
    if entity is a JSON datasource, or field type is json, it will use this jsonpath to extract data.
    `More details on JsonPath strings <http://goessner.net/articles/JsonPath/>`_.
- `type` : {integer, boolean, string, float, json}, optional, default=string
    If defined, the value will be converted to the given type. If it is not possible, then the resulting value will be null.
- `regex` : string, optional
    applies the given regular expression and assigns the first match to the value.
- `split` : string, optional
    splits the value to an array of values using the provided regular expression.
- `dateFormat` : string, optional
    transforms value to a date using the given date/time format.
- `join` : string, optional
    concatenates values using the defined separator. Used together with `jsonpath` only.
- `delimiter` : string, optional
    concatenates values using the defined separator. Used together with `jsonpath` only.
- `template` : string, optional
    used to define a template for strings. May use variables.
- `script` : string, optional
    call the python script defined in this element and assign the result to this field. May use any of the built-in functions or any one defined in a `Script` element. Variables can also be used in script elements. Could also be defined as inner <script> tag.
- `transform` : {'json', 'csv'}, optional
    transforms this field to a datasource. For example, it can be used to parse JSON or CSV data stored in a DB column. It's values can either be `json` or `csv`.
- `headers` : list, optional
    used only if `transform="csv"`. Defines the header names for each item in the CSV field. Not currently implemented.
- `required` : {'true', 'false'}, optional, default=false
    whether or not this field is required to have a value.
- `multipart` : {'true', 'false'}, optional
    if the results of `jsonpath` are complex/multipart value or simple value, Used only with `jsonpath`
- `key_path` : string, optional
    a JSON path expression for identifying the keys of a map. Used together with `value_path`
    `More details on JsonPath strings <http://goessner.net/articles/JsonPath/>`_.
- `value_path` : string, optional
    a JSON path expression for identifying the values of a map. Used together with `key_path`.
  
.. note::
    It is not possible to use the name for field 'opening' if fields as 'opening.title' is also required. 

Examples:

HTTP JSON entities:

.. code-block:: xml

    <entity name="jar_application" datasource="jar" query="get_s/#{employer}/#{application}.json">
        <field name="ja.bid_rate" type="float" jsonpath="$.result.hr_pay_rate" />
        <field name="ja.bid_amount" type="float" jsonpath="$.result.fp_pay_amount" />
        <field name="opening.pref_count" type="int" jsonpath="$.result.job_pref_matches.prefs_match" />
        <field name="application.creation_time" jsonpath="$.result.creation_time" dateFormat="YYYY-mm-DD" />

    </entity>

    <entity name="contractor" datasource="jar" query="opening/f/#{opening}.json">
        <field name="contractor.skills" path="$.skills.*.skl_name" join="," />
        <field name="contractor.greeting" template="Hello #{contractor.name}" />
        <field name="matches_pref_english" script="#{contractor.dev_eng_skill}> #{pref_english})" />
    </entity>

DB entity:

.. code-block:: xml

    <entity name="dbentity" datasource="mysqlConnection">
        <query>
            <![CDATA[
                SELECT *
                FROM table t1 JOIN table t2 ON t1.id = t2.reference
                WHERE t2.creation_time < '#{start_date}'
            ]]>
        </query>
        <field name="id" column="t1.id" />
        <field name="name" column="t1.full_name" />
        <field name="category" column="t2.category" />
        <field name="active" type="boolean" column="t2.is_active" />
        <field name="opening.segment" script="getSegment('#{category}')" />
    </entity>

DB entity where results should be read by table:

.. code-block:: xml

    <entity name="dbentity" datasource="mysqlConnection">
        <query target="data">
            <![CDATA[
                CREATE TEMP TABLE data AS (
                SELECT *
                FROM table t1 JOIN table t2 ON t1.id = t2.reference
                WHERE t2.creation_time < '#{start_date}')
            ]]>
        </query>
        <field name="id" column="t1.id" />
        <field name="name" column="t1.full_name" />
        <field name="category" column="t2.category" />
        <field name="active" type="boolean" column="t2.is_active" />
        <field name="opening.segment" script="getSegment('#{category}')" />
    </entity>

Pig entity:

.. code-block:: xml

    <entity name="dbentity" datasource="pigConnection">
        <query target="output">
            <![CDATA[
                batting = load 'Batting.csv' using PigStorage(',');
                runs = FOREACH batting GENERATE $0 as playerID, $1 as year, $8 as runs;
                grp_data = GROUP runs by (year);
                STORE grp_data INTO 'output';
            ]]>
        </query>
        <field name="id" column="t1.id" />
        <field name="name" column="t1.full_name" />
        <field name="category" column="t2.category" />
        <field name="active" type="boolean" column="t2.is_active" />
        <field name="opening.segment" script="getSegment('#{category}')" />
    </entity>

Entity with field json datasource:

.. code-block:: xml

    <field name="contractor_info" transform="json" column="contractor_info"/>
    <entity name="contractor_info" datasource="contractor_info">
        <field name="contractor.dev_is_looking" jsonpath="$.dev_is_looking" />
        <field name="contractor.dev_is_looking_week" jsonpath="$.dev_is_looking_week" />
        <field name="contractor.dev_active_interviews" jsonpath="$.dev_active_interviews" />
        <field name="contractor.dev_availability" type="integer" jsonpath="$.dev_availability" />
    </entity>


Python Scripts
~~~~~~~~~~~~~~

There are two variants to pass variables to the python script:

* using template formatting:

.. code-block:: xml

    <field name="uniqueName" script="myFunction(#{id}, '#{name}')" />

* using variables:

.. code-block:: xml

    <field name="uniqueName" script="myFunction(id, name)" />

.. _sqoop:

Sqoop
-----

Tag sqoop instructs import handler to run a Sqoop import. It should be
used only on entities that have a pig datasource. A sqoop tag may
contain the following attributes:

- `target` : string, required
    the target file to save imported data on HDFS.
- `datasource` : string, required
    a reference to the DB datasource to use for importing the data.
- `table` : string, required
    the name of the table to import its data.
- `where` : string, optional
    an expression that might be passed to the table for filtering the rows to be imported.
- `direct` : boolean, optional
    whether to use direct import (see `Sqoop documentation <https://sqoop.apache.org/docs/1.4.4/SqoopUserGuide.html#_importing_views_in_direct_mode>`_ on --direct for more details)
- `mappers` : integer, optional
    an integer number with the mappers to use for importing data. If table is a view or does not have a key, it should be 1. Default value is 1.
- `options` : string, optional
    Extra options for sqoop import command.

If the sqoop tag contains body, then it should be a valid SQL statement.
These statements will be executed on the database before the Sqoop
import. This feature is particularly useful if the following needs to be run:

.. code-block:: xml

    <entity name="myEntity" datasource="pigConnection">
        <query target="output">
        <![CDATA[
            batting = load 'Batting.csv' using PigStorage(',');
            runs = FOREACH batting GENERATE $0 as playerID, $1 as year, $8 as runs;
            grp_data = GROUP runs by (year);
            STORE grp_data INTO 'output';
        ]]>
        </query>
        <!-- Transfer table dataset to HDFS -->
        <sqoop target="dataset" table="dataset" datasource="sqoop_db_datasource" />

        <!-- Query inside sqoop tag needs to be executed on the DB before running the sqoop command -->
        <!-- Multiple sqoop tags should be allowed, in case more than one imports are reqyuired -->
        <sqoop target="new_data" table="temp_table" datasource="sqoop_db_datasource" direct="true" mappers="1">
        <![CDATA[
            CREATE TEMP TABLE target_openings AS SELECT * FROM openings WHERE creation_time BETWEEN #{start} AND #{end};
            CREATE TABLE temp_table AS SELECT to.*, e.* FROM target_openings to JOIN employer e ON to.employer=e."Record ID#";
        ]]>
        </sqoop>
        <!-- Fields -->
        <field ... />
    </entity>

In order to load sqoop results in the pig script, the following must be defined:

.. code-block:: xml

    <entity name="myEntity" datasource="pigConnection">
        <sqoop target="openings_dataset" table="temp_table" datasource="sqoop_db_datasource" direct="true" mappers="1"/>
        <query autoload_sqoop_dataset="true" sqoop_dataset_name="openings_dataset" target="result">
            <![CDATA[
            register 's3://odesk-match-staging/pig/lib/elephant-bird-core-4.4.jar';
            register 's3://odesk-match-staging/pig/lib/elephant-bird-pig-4.4.jar';
            register 's3://odesk-match-staging/pig/lib/elephant-bird-hadoop-compat-4.4.jar';
            register 's3://odesk-match-staging/pig/lib/piggybank-0.12.0.jar';

            openings_dataset = LOAD '$openings_dataset*' USING org.apache.pig.piggybank.storage.CSVExcelStorage(',', 'YES_MULTILINE') AS (
                bidid:long
                , jobid:long
                , seller_userid:long
                , is_hired:chararray
                , seller_country:chararray
            );

            result = FOREACH openings_dataset GENERATE 
                * 
                , funcs.join((job_country, seller_country), ',') as buyer_seller_countries
            ;
            ]]>
        <query/>
    <entity/>


In addition, sqoop results can also auto load, for example:

.. code-block:: xml

    <entity name="myEntity" datasource="pigConnection">
        <sqoop target="openings_dataset" table="temp_table" datasource="sqoop_db_datasource" direct="true" mappers="1"/>
        <query autoload_sqoop_dataset="true" sqoop_dataset_name="openings_dataset" target="result">
            <![CDATA[
            result = FOREACH openings_dataset GENERATE 
                * 
                , funcs.join((job_country, seller_country), ',') as buyer_seller_countries
            ;
            ]]>
        <query/>
    <entity/>

When `autoload_sqoop_dataset` set CloudML will automatically add sqoop results definition on the top of the pig script. For example:

.. code-block:: txt

    register 's3://odesk-match-staging/pig/lib/elephant-bird-core-4.4.jar';
    register 's3://odesk-match-staging/pig/lib/elephant-bird-pig-4.4.jar';
    register 's3://odesk-match-staging/pig/lib/elephant-bird-hadoop-compat-4.4.jar';
    register 's3://odesk-match-staging/pig/lib/piggybank-0.12.0.jar';
                

    result = LOAD '$dataset*' USING org.apache.pig.piggybank.storage.CSVExcelStorage(',', 'YES_MULTILINE') AS ( some_field:field_type ); 


Nested entities
---------------

It is possible that, not all data required will originate from one
entity, also, it may be possible to gather data from more than one
data source. For example, consider the following use case::

    A really important feature is application ranking.
    In order to rank the application, data regarding the application,
    the employer, the job opening and the contractor are required.
    However, this data may be derived from different HTTP URLs.


A solution for this problem is to use nested entities. A nested entity is a normal entity, with the benefit that it is able to use data from it's parent entity to formulate the query. A nested entity may result in two ways:

- querying a 'global' datasource (i.e. querying a different table in DB, calling a different HTTP service).
- converting one of the parent entity's field to a new entity (i.e. parsing the data of a DB column as a JSON document). In this case, the field acts as a data source.

A nested entity is defined inside another `<entity>` and follows exactly the same syntax. However, it might also use the values of parent entity as variables, in addition to the input parameter values.

Example:

.. code-block:: xml

    <entity name="application" datasource="ods" query="job_application/pa/#{application}.json">
        <field name="opening" jsonpath="$.result.#{application}.opening_ref" />
        <field name="contractor" jsonpath="$.result.#{application}.developer_ref" />
        <field name="employer" jsonpath="$.result.#{application}.team_ref" />

        <!-- Nested entity using a global datasource -->
        <entity name="opening" datasource="odr" query="opening/f/#{opening}.json">
            <field name="opening.title" jsonpath="$.op_title" />
            <field name="opening.description" jsonpath="$.op_job" />
        </entity>
    </entity>


The second option is to convert one of the parent entity's fields to a
new entity. This is useful if a field in the parent entity contains CSV
or JSON data. In order to undertake this, two things need to be done:

- Define property 'transform' in the parent entity field, using the appropriate type. This creates a datasource accessible from all child entities. The data source's name is the field's name, while the data source type depends on the the value of the transform entity.
- In the new entity, define the name of the parent entity's field as the data source name. 

Example:

.. code-block:: xml

    <!-- Parent entity -->
    <entity name="user" datasource="dbEntity" query="SELECT * FROM users">
        <!-- Convert field to CSV datasource -->
        <field name="permissions" transform="csv" headers="read,write,execute"/>
        <!-- Nested entity using data from CSV field -->
        <entity name="permissionEntity" datasource="permissions">
            <field name="user.read" column="read" />
            <field name="user.execute" column="execute" />
        </entity>

        <
        <!-- Convert field to JSON datasource -->
        <field name="profile" transform="json" />

        <!-- Nested entity using data from JSON field -->
        <entity name="profileEntity" datasource="profile">
            <field name="score" jsonpath="$.score" />
        </entity>
    </entity>

.. _predict:

Predict
=======

The final part of the data import handler describes which models to
invoke and how the response is formulated. While the old import handler
was used with a single model, the new version should allow
multiple binary classifier models use, provided that the same
input vector are expected.

Response format is defined inside `<predict>` tag. Predict tag is required to needs have the following sub-elements:

- `<model>` - defines parameters for using a model with the data from the `<import>` part of the handler.
- `<result>` - defines how to formulate the response.

Model
-----

In order to calculate the result of a prediction, one or more models
need to be invoked together with the data from the import handler. Each
model invocation is defined using a `<model>` tag. A model tag could
have the following attributes:

- `name` : string, required
    a name to uniquely identify the results of this model.
- `value` : string, optional
    holds the name of the model to use.
- `script` : string, optional
    calls python code to decide the name of the model to use.

.. note::

    Either the value or the script attribute need to be defined. Failure to do this will result in an error.

Also, additional child elements could be defined:

* positive_label
* weight

positive_label
~~~~~~~~~~~~~~

Allows overriding which label to use as positive label. If not defined, true is considered as a positive label. Example:

.. code-block:: xml

    <model name="rank" value="BestMatch.v31">
        <positive_label value="false" />
    </model>

A positive_label tag may have the following attributes:

* `value` : string
    holds the value of the model positive label.
* `script` : string
    calls python code to decide the value of the model positive label.

weight
~~~~~~

A positive_label tag may have the following attributes:

* `value` : string
    holds the weight.
* `script` : string
    calls python code to decide the weight.
* `label` : string
    a label of data to apply the weight.


.. code-block:: xml

    <model name="recommend" value="my_model">
        <weight label="good_hire" script="3 if isHourly('#{opening.type_raw}') else 2" />
        <weight label="no_hire" value="1.0" />
        <weight label="bad_hire" value="1.0" />
    </model>

.. note::

    It could be few `<model>` sub-elements in `<predict>` tag.

Result
------

Defines the method to render results.

It could include two sub-elements:

* `label`
* `probability`

Label defines .... todo ... and contains the following attributes:

* `model` : string, optional
* `script` : string, optional

Probability defines ... todo ... and contains the following attributes:

* `label` : string, optional
* `model` : string, optional
* `script` : string, optional
    python script to ...

Examples of `result` section:

.. code-block:: xml

    <result>
      <label script="True if getPredictedLabel() == 'good_hire' and getProbabilityFor('good_hire') > 0.5 else False"/>
      <probability script="getProbabilityFor('good_hire')"/>
    </result>