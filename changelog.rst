Changelog
=========

0.4.3
---------
* Task - Added chained exceptions with traceback info
* Support - Migrated pigdatasource to aws.describe_cluster method

0.4.2
---------
* Support - Removed jsonpath from requirements and temporary included it to project

0.4.1
---------
* Task - Added log loss metric
* Feature - Added strict mode. In this mode does not use default values for required features


0.4.0
---------
* Task - Set params default values for all existing transformers
* Task - Added exclude param to categorical feature type
* Task - Migrate from boto to boto3
* Task - Disabled warning message when skip records
* Task - Added support categories param for categorycal feature type
* Bug - Fixed issues with empty data on training transformer
