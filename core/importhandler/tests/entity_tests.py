"""
Unittests for entities and fields related classes.
"""

# Author: Nikolay Melnik <nmelnik@upwork.com>

import unittest
from lxml import objectify

from core.importhandler.entities import Field, FieldException, Entity
from core.importhandler.importhandler import ImportHandlerException


class TestField(unittest.TestCase):
    FIELD_WITH_SCRIPT = objectify.fromstring("""
        <field name="name"><script>"Name is: " + #{value}</script></field>""")
    FIELD_WITH_JSONPATH = objectify.fromstring("""
        <field name="employer.el" jsonpath="$.el" />""")

    def test_declaration(self):
        with self.assertRaisesRegexp(
                ImportHandlerException,
                "Type of the field field_name is invalid: invalid. \
Choose one of boolean, integer, json, float, string"):
            field = Field({
                'name': 'field_name',
                'type': 'invalid'}, entity=None)

        with self.assertRaisesRegexp(
                ImportHandlerException,
                "Field field_name declaration is invalid: use dateFormat \
only for string fields"):
            field = Field({
                'name': 'field_name',
                'type': 'integer',
                'dateFormat': '%M'}, entity=None)

    def test_required(self):
        field_required = Field({
            'name': 'field_name',
            'type': 'string',
            'required': 'true'
        }, entity=None)
        field = Field({
            'name': 'field_name',
            'type': 'string'
        }, entity=None)
        with self.assertRaises(FieldException):
            field_required.process_value(None, None)
        value = field.process_value(None, None)
        self.assertEqual(value, None)

    def test_script(self):
        from core.importhandler.scripts import ScriptManager
        script_manager = ScriptManager()
        field = Field(self.FIELD_WITH_SCRIPT, entity=None)
        self.assertEquals(
            field.process_value("Nikolay", script_manager),
            "Name is: Nikolay")

    def test_json_path(self):
        field = Field(self.FIELD_WITH_JSONPATH, entity=None)
        self.assertEquals(field.process_value({"el": "val"}, None),
                          "val")

    def test_regexp(self):
        field = Field({'name': 'f',
                       'regex': '(?<=-)\w+'}, entity=None)
        self.assertEquals(field.process_value("cloud-ml", None), "ml")
        self.assertEquals(field.process_value("cloudml", None), None)

    def test_split(self):
        field = Field({'name': 'f',
                       'split': ';'}, entity=None)
        self.assertEquals(field.process_value("a;b;c", None),
                          ['a', 'b', 'c'])

        field = Field({'name': 'f',
                       'split': ';'}, entity=None)
        self.assertEquals(field.process_value("abc", None), ['abc'])

    def test_delimiter(self):
        field = Field({'name': 'f',
                       'delimiter': ';'}, entity=None)
        self.assertEquals(field.process_value(['a', 'b', 'c'], None), "a;b;c")

    def test_dateFormat(self):
        from datetime import datetime
        field = Field({'name': 'f',
                       'dateFormat': '%d/%m/%y'}, entity=None)
        self.assertEquals(field.process_value("1/1/06", None),
                          datetime(2006, 1, 1, 0, 0))

    def test_template(self):
        field = Field({'name': 'f',
                       'template': 'result=#{value}'}, entity=None)
        self.assertEquals(field.process_value("hello", None),
                          "result=hello")

    def test_converion_error(self):
        entity = Entity({})
        field = Field({'name': 'f', 'type': 'integer'}, entity=entity)
        self.assertEquals(field.process_value("hello", None), None)
