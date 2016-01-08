"""
Unittests for utility methods.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import unittest


class TestUtils(unittest.TestCase):

    def test_autoload_fields_by_row(self):
        from cloudml.importhandler.entities import Entity, Field
        from cloudml.importhandler.utils import autoload_fields_by_row
        ent = Entity({'name': 'ent'})
        row = {'name': 'Name', 'count': 5, 'accuracy': 3.5,
               'data': '{"x": "x", "y": 5.4}'}
        autoload_fields_by_row(ent, row)
        self.assertEquals(
            ent.fields.keys(),
            ['count', 'data', 'name', 'accuracy'])
        self.assertEquals(ent.fields['name'].type, 'string')
        self.assertEquals(ent.fields['count'].type, 'integer')
        self.assertEquals(ent.fields['accuracy'].type, 'float')
        self.assertEquals(ent.fields['data'].type, 'string')
        self.assertEquals(ent.fields['data'].transform, 'json')
        self.assertEquals(ent.fields['data'].column, 'data')
        self.assertEquals(
            ent.nested_entities_field_ds.keys(), ['data'])
        nested_ent = ent.nested_entities_field_ds['data']
        self.assertEquals(nested_ent.name, 'data')
        self.assertItemsEqual(nested_ent.fields.keys(),
                              ['x', 'y'])

        ent = Entity({'name': 'ent'})
        ent.fields['count'] = Field({
            'name': 'count',
            'type': 'float'}, ent)
        self.assertEquals(ent.fields['count'].type, 'float')
