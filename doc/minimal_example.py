import cloudml
from cloudml.importhandler.datasources import CsvDataSource
from cloudml.importhandler.importhandler import ImportHandler
import cloudml.importhandler.importhandler as imp
>>> import urllib

# url with a dataset
>>> url = "http://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data"
>>> filename = "TA_dataset.data"

# download a dataset
>>> opener = urllib.URLopener()
>>> opener.retrieve(url, filename)

>>> import psycopg2
>>> dsn = "dbname=test user=postgres host=localhost"
>>> con = psycopg2.connect(dsn)
>>> cur = con.cursor()
>>> cur.execute("CREATE DATABASE CloudML-test")
>>> cur.execute("CREATE TABLE TA_dataset (native_speaker INT, instructor INT, course INT, summer_regular INT, class_size INT, TA_score INT)")
>>> my_file = open(filename)
>>> sql = "COPY TA_dataset FROM stdin DELIMITER \',\' CSV header;"
>>> cur.copy_expert(sql, my_file)
>>> cur.close()
>>> con.close()




$ psql user=postgres
Password:
postgres=# CREATE DATABASE cloudml_db;
postgres=# CREATE TABLE ta_dataset (native_speaker INT, instructor INT, course INT, summer_regular INT, class_size INT, TA_score INT);
postgres=# \copy ta_dataset FROM <insert path to the dataset> DELIMITER ',' CSV


$ python importhandler.py explanation_plan
