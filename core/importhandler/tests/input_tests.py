"""
Unittests for processing user input data classes.
"""

# Author: Nikolay Melnik <nmelnik@upwork.com>

import unittest
from lxml import objectify
from datetime import datetime

from core.importhandler.inputs import Input
from core.importhandler.exceptions import ImportHandlerException


class TestInput(unittest.TestCase):
    BOOLEAN = objectify.fromstring(
        '<param name="only_fjp" type="boolean" />')
    INT = objectify.fromstring(
        '<param name="application" type="integer" regex="\d+" />')
    DATE = objectify.fromstring(
        '<param name="created" type="date" format="%A %d. %B %Y" />')

    def test_params_validation(self):
        inp = Input(self.INT)
        self.assertEqual(inp.process_value('1'), 1)
        self.assertRaises(ImportHandlerException, inp.process_value, 'str')
        self.assertRaises(ImportHandlerException, inp.process_value, '-1')

        inp = Input(self.DATE)
        self.assertEqual(inp.process_value('Monday 11. March 2002'),
                         datetime(2002, 3, 11, 0, 0))
        with self.assertRaisesRegexp(
                ImportHandlerException, "Value of the input parameter \
created is invalid date in format %A %d. %B %Y: 11/03/02"):
            inp.process_value('11/03/02')
        with self.assertRaisesRegexp(
                ImportHandlerException, "Input parameter created is required"):
            inp.process_value(None)

        inp = Input(self.BOOLEAN)
        self.assertEqual(inp.process_value('1'), 1)
        self.assertEqual(inp.process_value('0'), 0)

        inp = Input(dict(name="application", type="invalid"))
        self.assertRaises(ImportHandlerException, inp.process_value, 'str')
