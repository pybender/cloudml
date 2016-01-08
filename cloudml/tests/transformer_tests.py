# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>
import os
import unittest
from mock import patch

from transformer import main, PARAMETERS_REQUIRED, \
    INVALID_TRANSFORMER_CONFIG, INVALID_EXTRACTION_PLAN, \
    DONE
from test_utils import db_row_iter_mock


class ImportHandlerTestCase(unittest.TestCase):
    @patch('transformer.logging')
    def test_params_validation(self, logging_mock):
        res = main(argv=['testdata/transformers/transformer.json'])
        self.assertEquals(res, PARAMETERS_REQUIRED)
        logging_mock.warn.assert_called_with(
            "You must define either an input file or an extraction plan")
        logging_mock.reset_mock()

    @patch('transformer.logging')
    @patch('cloudml.importhandler.datasources.DbDataSource._get_iter',
           return_value=db_row_iter_mock(
               filename='testdata/transformers/train.data.json'))
    def test_extraction(self, db_mock, logging_mock):
        res = main(argv=[
            'testdata/transformers/transformer.json',
            '-e', 'testdata/extractorxml/train-transformer-import-handler.xml',
            '-I', 'start=2012-12-03', '-I', 'end=2012-12-04',
            '-o', 'out.data'])
        self.assertEquals(res, DONE)
        self.assertFalse(logging_mock.called)
        os.remove('out.data')

    @patch('importhandler.logging')
    def test_input_data(self, logging_mock):
        res = main(argv=[
            'testdata/transformers/transformer.json',
            '-i', 'testdata/transformers/train.data.json'])
        self.assertEquals(res, DONE)
        self.assertFalse(logging_mock.called)

    @patch('importhandler.logging')
    def test_wrong_config(self, logging_mock):
        res = main(argv=[
            'testdata/transformers/empty_name.json',
            '-i', 'testdata/transformers/train.data.json'])
        self.assertEquals(res, INVALID_TRANSFORMER_CONFIG)

    @patch('importhandler.logging')
    def test_wrong_ih(self, logging_mock):
        res = main(argv=[
            'testdata/transformers/transformer.json',
            '-e', 'testdata/extractorxml/invalid/no-entity.xml',
            '-I', 'start=2012-12-03', '-I', 'end=2012-12-04'])
        self.assertEquals(res, INVALID_EXTRACTION_PLAN)
