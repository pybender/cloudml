#!/bin/bash
# Install sqoop and postgresql connector. Store in s3 and load 
# as bootstrap step.
 
bucket_location='s3://odesk-match-prod/cloudml/lib/'
sqoop_jar='sqoop-1.4.4.bin__hadoop-1.0.0'
sqoop_jar_gz=$sqoop_jar.tar.gz
postgres_jar='postgresql-9.1-901.jdbc4.jar'
postgres_dir_gz=$postgres_dir.tar.gz
 
cd
 
hadoop fs -copyToLocal $bucket_location$sqoop_jar_gz .
tar -xzf $sqoop_jar_gz
hadoop fs -copyToLocal $bucket_location$postgres_jar .
cp $postgres_jar $sqoop_jar/lib 