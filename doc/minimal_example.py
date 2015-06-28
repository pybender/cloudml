import cloudml
from cloudml.importhandler.datasources import CsvDataSource
from cloudml.importhandler.importhandler import ImportHandler
import cloudml.importhandler.importhandler as imp
import urllib

# url with a dataset
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data"
filename = "TA_dataset.csv"

# download a dataset
opener = urllib.URLopener()
opener.retrieve(url, filename)

# load a config file
plan = imp.ExtractionPlan("TA_dataset_config")

# feed it to the import handler
import_handler = ImportHandler(plan=plan)

print(import_handler.plan)
