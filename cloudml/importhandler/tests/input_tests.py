"""
Unittests for processing user input data classes.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import os
import unittest
from lxml import objectify
from datetime import datetime

from cloudml.importhandler.inputs import Input
from cloudml.importhandler.exceptions import ImportHandlerException

BASEDIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../testdata'))


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


class InputDatasourceTest(unittest.TestCase):

    def setUp(self):
        from cloudml.importhandler.importhandler import ExtractionPlan
        self._plan = ExtractionPlan(os.path.join(
                                    BASEDIR,
                                    'extractorxml',
                                    'input-datasource-handler.xml'))

    def test_json(self):
        from cloudml.importhandler.importhandler import ImportHandler
        self._extractor = ImportHandler(self._plan, {
            'contractor_info': '{ "skills":[{"skl_status":"0","ts_tests_count"\
:"0","skl_name":"microsoft-excel","skl_external_link":"http:\/\/en.wikipedia.\
org\/wiki\/Microsoft_Excel","skl_has_tests":"1","skl_pretty_name":"Microsoft\
 Excel","skill_uid":"475721704063008779","skl_rank":"1","skl_description":\
 "Microsoft Excel is a proprietary commercial spreadsheet application written\
 and distributed by Microsoft for Microsoft Windows and Mac OS X. It features\
 calculation, graphing tools, pivot tables, and a macro programming language\
 called Visual Basic for Applications."},{"skl_status":"0","ts_tests_count":\
 "0","skl_name":"microsoft-word","skl_external_link":"http:\/\/en.wikipedia.\
 org\/wiki\/Microsoft_Word","skl_has_tests":"1","skl_pretty_name":"Microsoft\
  Word","skill_uid":"475721704071397377","skl_rank":"2","skl_description":\
  "Microsoft Office Word is a word processor designed by Microsoft."}]}',
        })
        row = self._extractor.next()
        self.assertEqual(row['contractor.skills'],
                         'microsoft-excel,microsoft-word')
