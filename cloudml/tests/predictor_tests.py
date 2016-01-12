# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>
import os
import unittest
from mock import patch

from predictor import main, PARAMETERS_REQUIRED, INVALID_TRAINER, \
    DONE
from test_utils import db_row_iter_mock


class ImportHandlerTestCase(unittest.TestCase):
    @patch('predictor.logging')
    def test_predictor_validation(self, logging_mock):
        res = main(argv=['testdata/trainer/model.dat'])
        self.assertEquals(res, PARAMETERS_REQUIRED)
        logging_mock.warn.assert_called_with(
            "Need to either specify -i or -e")
        logging_mock.reset_mock()

        res = main(argv=['testdata/extractorxml/train-import-handler.xml'])
        self.assertEquals(res, INVALID_TRAINER)
        self.assertFalse(logging_mock.called)

    @patch('importhandler.logging')
    @patch('cloudml.importhandler.datasources.DbDataSource._get_iter',
           return_value=db_row_iter_mock())
    def test_extraction(self, db_mock, logging_mock):
        res = main(argv=[
            'testdata/trainer/model.dat',
            '-e', 'testdata/extractorxml/train-import-handler.xml',
            '-U', 'start=2012-12-03', '-U', 'end=2012-12-04'])
        self.assertEquals(res, DONE)
        self.assertFalse(logging_mock.called)

    @patch('predictor.logging')
    def test_input_data(self, logging_mock):
        res = main(argv=[
            'testdata/trainer/model.dat',
            '-i', 'testdata/trainer/trainer.data.csv'])
        self.assertEquals(res, DONE)

    @patch('predictor.logging')
    def test_roc_method(self, logging_mock):
        res = main(argv=[
            'testdata/trainer/model.dat',
            '-i', 'testdata/trainer/trainer.data.json',
            '-m', 'roc'])
        self.assertEquals(res, DONE)

    @patch('predictor.logging')
    def test_csv_method(self, logging_mock):
        res = main(argv=[
            'testdata/trainer/model.dat',
            '-i', 'testdata/trainer/trainer.data.json',
            '-m', 'csv'])
        self.assertEquals(res, DONE)
        os.remove('result.csv')
