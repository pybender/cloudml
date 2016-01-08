"""
Unittests for entities and fields related classes.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import unittest
from lxml import objectify

from cloudml.importhandler.entities import Field, ProcessException, Entity
from cloudml.importhandler.importhandler import ImportHandlerException


class TestEntity(unittest.TestCase):
    ENTITY = objectify.fromstring("""
        <entity datasource="odw" name="retrieve">
            <sqoop target="dataset" table="tbl" datasource="sqoop_ds">
            <![CDATA[CREATE TABLE tbl AS SELECT qi.* FROM qi;]]>
            </sqoop>
            <query target="C"><![CDATA[query here]]></query>
            <field name="opening_id" type="integer" column="opening"/>
            <field name="name" type="string" column="opening"/>
            <field column="contractor_info" name="contractor_info"
                transform="json"/>
            <entity datasource="contractor_info" name="contractor_info">
                <field join="," jsonpath="$.skills.*.skl_name" name="skills"/>
            </entity>
            <entity datasource="global_ds" name="global">
                <field name="title"/>
            </entity>
        </entity>""")

    def test_entity_declaration(self):
        entity = Entity(self.ENTITY)
        self.assertEquals(entity.name, "retrieve")
        self.assertEquals(entity.query, "query here")
        self.assertEquals(entity.query_target, "C")
        self.assertEquals(len(entity.fields), 3)
        self.assertItemsEqual(
            [f.name for f in entity.fields.values()],
            ['opening_id', 'name', 'contractor_info'])
        self.assertItemsEqual(
            entity.fields.keys(),
            ['opening_id', 'name', 'contractor_info'])
        self.assertEquals(entity.datasource_name, "odw")

        self.assertFalse(entity.autoload_fields)
        self.assertFalse(entity.fields_loaded)

        self.assertEquals(
            [e.name for e in entity.nested_entities_global_ds],
            ['global'])
        self.assertEquals(
            entity.nested_entities_field_ds.keys(), ['contractor_info'])

        self.assertEquals(
            [s.target for s in entity.sqoop_imports],
            ['dataset'])
        self.assertEquals(
            [s.datasource_name for s in entity.sqoop_imports],
            ['sqoop_ds'])
