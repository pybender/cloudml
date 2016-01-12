"""
Unittests for sqoop section of entity with pig datasource.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import unittest
from lxml import objectify

from cloudml.importhandler.entities import Sqoop


class TestSqoop(unittest.TestCase):
    TAG = objectify.fromstring(
        '<sqoop target="dataset" table="tbl" datasource="db">'
        '<![CDATA[CREATE TABLE tbl AS SELECT #{field} from public);]]>'
        '</sqoop>')
    NO_QUERY = objectify.fromstring(
        '<sqoop target="dataset" table="tbl" datasource="db">'
        '</sqoop>')

    def test_build_query(self):
        sqoop = Sqoop(self.TAG)
        self.assertEquals(
            sqoop.build_query({"field": 'name'}),
            "CREATE TABLE tbl AS SELECT name from public);")

        sqoop = Sqoop(self.NO_QUERY)
        self.assertEquals(
            sqoop.build_query({"field": 'name'}), None)
