# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>
import os
import unittest
from mock import patch

from importhandler import INVALID_EXTRACTION_PLAN, main, DONE
from test_utils import db_row_iter_mock


class ImportHandlerTestCase(unittest.TestCase):
    @patch('importhandler.logging')
    def test_import_handler(self, logging_mock):
        res = main(argv=['testdata/extractorxml/train-import-handler.xml'])
        self.assertEquals(res, INVALID_EXTRACTION_PLAN)
        logging_mock.warn.assert_called_with(
            "Invalid extraction plan: Missing input parameters: start, end")
        logging_mock.reset_mock()

        res = main(argv=[
            'testdata/extractorxml/train-import-handler.xml',
            '-U', 'start=2012-12-03', '-U', 'end=2012-12-04'])
        self.assertEquals(res, DONE)
        self.assertFalse(logging_mock.called)

    @patch('importhandler.logging')
    @patch('cloudml.importhandler.datasources.DbDataSource._get_iter',
           return_value=db_row_iter_mock())
    def test_output(self, db_mock, logging_mock):
        res = main(argv=[
            'testdata/extractorxml/train-import-handler.xml',
            '-U', 'start=2012-12-03', '-U', 'end=2012-12-04',
            '-o', 'data.bak'])
        self.assertEquals(res, DONE)
        self.assertFalse(logging_mock.called)
        self.assertTrue(os.path.isfile('data.bak'))
        os.remove('data.bak')
