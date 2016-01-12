# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>
import os
import unittest
from mock import patch, ANY

from trainer import main, INVALID_FEATURE_MODEL, \
    PARAMETERS_REQUIRED, INVALID_EXTRACTION_PLAN, DONE
from test_utils import db_row_iter_mock


class TrainerTestCase(unittest.TestCase):
    @patch('trainer.logging')
    def test_features_not_found(self, logging_mock):
        filename = 'not-exsistant-file.json'
        res = main(argv=[filename])
        self.assertEquals(res, INVALID_FEATURE_MODEL)
        logging_mock.warn.assert_called_with(
            "Can't load features file. [Errno 2] "
            "No such file or directory: '{0}'".format(filename))

    @patch('trainer.logging')
    def test_invalid_features(self, logging_mock):
        filename = 'testdata/trainer/invalid-features.json'
        res = main(argv=[filename])
        self.assertEquals(res, INVALID_FEATURE_MODEL)
        logging_mock.warn.assert_called_with(
            "Invalid feature model: {0} No JSON object could"
            " be decoded ".format(filename))

    @patch('trainer.logging')
    def test_not_all_params_filled(self, logging_mock):
        res = main(argv=['testdata/trainer/features.json'])
        self.assertEquals(res, PARAMETERS_REQUIRED)
        logging_mock.warn.assert_called_with(
            'You must define either an input file or an extraction plan')

    @patch('trainer.logging')
    @patch('cloudml.trainer.trainer.Trainer.train')
    @patch('cloudml.trainer.trainer.Trainer.test')
    def test_percent(self, test_mock, train_mock, logging_mock):
        res = main(argv=[
            'testdata/trainer/features.json', '-tp', 'percent',
            '-i', 'testdata/trainer/trainer.data.json'])
        self.assertEquals(res, DONE)
        logging_mock.warn.assert_called_with(
            "Percent value 'percent' would be"
            " ignored. Should be value from 0 to 100.")
        train_mock.assert_called_with(ANY, 0, store_vect_data=False)
        self.assertFalse(test_mock.called)
        train_mock.reset()
        test_mock.reset()
        logging_mock.reset()

        res = main(argv=[
            'testdata/trainer/features.json', '-tp', '200',
            '-i', 'testdata/trainer/trainer.data.json'])
        self.assertEquals(res, DONE)
        logging_mock.warn.assert_called_with(
            "Percent value '200' would be"
            " ignored. Should be value from 0 to 100.")
        train_mock.assert_called_with(ANY, 0, store_vect_data=False)
        train_mock.reset()
        test_mock.reset()
        logging_mock.reset()

        res = main(argv=[
            'testdata/trainer/features.json', '-tp', '40',
            '-i', 'testdata/trainer/trainer.data.json'])
        self.assertEquals(res, DONE)
        train_mock.assert_called_with(ANY, 40, store_vect_data=False)
        test_mock.assert_called_with(ANY, 40)

        res = main(argv=[
            'testdata/trainer/features.json', '-tp', '60',
            '-e', 'testdata/extractorxml/train-import-handler.xml',
            '-I', 'start=2012-12-03', '-I', 'end=2012-12-04'])
        self.assertEquals(res, DONE)
        train_mock.assert_called_with(ANY, 60)
        test_mock.assert_called_with(ANY, 60)

    @patch('trainer.logging')
    @patch('cloudml.trainer.trainer.Trainer.train')
    @patch('cloudml.trainer.trainer.Trainer.vect_data2csv')
    def test_store_train_vect(self, vect_data2csv_mock,
                              train_mock, logging_mock):
        res = main(argv=[
            'testdata/trainer/features.json',
            '-i', 'testdata/trainer/trainer.data.json',
            '-v', 'vect.bak'])
        self.assertEquals(res, DONE)
        train_mock.assert_called_with(ANY, 0, store_vect_data=True)
        self.assertTrue(vect_data2csv_mock.called)
        train_mock.reset()

    @patch('trainer.logging')
    @patch('cloudml.trainer.trainer.Trainer.train')
    @patch('cloudml.trainer.trainer.Trainer.test')
    def test_skip_tests(self, test_mock, train_mock, logging_mock):
        res = main(argv=[
            'testdata/trainer/features.json', '-tp', '10',
            '-i', 'testdata/trainer/trainer.data.json',
            '--skip-test'])
        self.assertEquals(res, DONE)
        train_mock.assert_called_with(ANY, 10, store_vect_data=False)
        self.assertFalse(test_mock.called)
        train_mock.reset()
        test_mock.reset()

        res = main(argv=[
            'testdata/trainer/features.json', '--skip-test',
            '-t', 'testdata/trainer/test.data.json',
            '-i', 'testdata/trainer/trainer.data.json'])
        self.assertEquals(res, DONE)
        train_mock.assert_called_with(ANY, 0, store_vect_data=False)
        self.assertFalse(test_mock.called)
        train_mock.reset()
        test_mock.reset()

        res = main(argv=[
            'testdata/trainer/features.json', '--skip-test',
            '-e', 'testdata/extractorxml/train-import-handler.xml',
            '-I', 'start=2012-12-03', '-I', 'end=2012-12-04',
            '-tp', '20'])
        self.assertEquals(res, DONE)
        train_mock.assert_called_with(ANY, 20)
        self.assertFalse(test_mock.called)
        train_mock.reset()
        test_mock.reset()

    @patch('trainer.logging')
    @patch('cloudml.trainer.trainer.Trainer.train')
    @patch('cloudml.trainer.trainer.Trainer.test')
    def test_define_train_and_test_dataset(self, test_mock, train_mock,
                                           logging_mock):
        res = main(argv=[
            'testdata/trainer/features.json',
            '-t', 'testdata/trainer/test.data.json',
            '-i', 'testdata/trainer/trainer.data.json'])
        self.assertEquals(res, DONE)
        train_mock.assert_called_with(ANY, 0, store_vect_data=False)
        test_mock.assert_called_with(ANY)

    @patch('trainer.logging')
    @patch('cloudml.trainer.trainer.Trainer.train')
    @patch('cloudml.trainer.trainer.Trainer.store_feature_weights')
    @patch('cloudml.trainer.trainer.Trainer.test')
    def test_store_weights(self, test_mock, weights_mock,
                           train_mock, logging_mock):
        res = main(argv=[
            'testdata/trainer/features.json',
            '-w', 'weights.bak',
            '-i', 'testdata/trainer/trainer.data.json'])
        self.assertEquals(res, DONE)
        train_mock.assert_called_with(ANY, 0, store_vect_data=False)
        weights_mock.assert_called_with(ANY)
        self.assertFalse(test_mock.called)
        os.remove('weights.bak')

    @patch('trainer.logging')
    def test_store_vect(self, logging_mock):
        res = main(argv=[
            'testdata/trainer/features.json',
            '--store-vect', 'vect.bak',
            '-i', 'testdata/trainer/trainer.data.json'])
        self.assertEquals(res, PARAMETERS_REQUIRED)
        logging_mock.warn.assert_called_with(
            "Model was trained, but not evaluated. You need "
            "to add --test or --test-percent param.")
        logging_mock.reset_mock()

        res = main(argv=[
            'testdata/trainer/features.json',
            '--store-vect', 'vect.bak',
            '-tp', '50',
            '-i', 'testdata/trainer/trainer.data.json'])
        self.assertEquals(res, DONE)
        self.assertFalse(logging_mock.warn.called)
        os.remove('vect.bak.npz')

    def test_store_trainer(self):
        res = main(argv=[
            'testdata/trainer/features.json',
            '-o', 'trainer.bak',
            '-i', 'testdata/trainer/trainer.data.json'])
        self.assertEquals(res, DONE)
        self.assertTrue(os.path.isfile('trainer.bak'))
        os.remove('trainer.bak')

    @patch('trainer.logging')
    @patch('cloudml.importhandler.datasources.DbDataSource._get_iter',
           return_value=db_row_iter_mock())
    def test_train_with_extraction_plan(self, db_mock, logging_mock):
        res = main(argv=[
            'testdata/trainer/features.json',
            '-e', 'testdata/trainer/trainer.data.json'])
        self.assertEquals(res, INVALID_EXTRACTION_PLAN)
        self.assertTrue(logging_mock.warn.called)
        logging_mock.reset_mock()

        res = main(argv=[
            'testdata/trainer/features.json',
            '-e', 'testdata/extractorxml/train-import-handler.xml'])
        self.assertEquals(res, INVALID_EXTRACTION_PLAN)
        logging_mock.warn.assert_called_with(
            "Invalid extraction plan: Missing input parameters: start, end")

        res = main(argv=[
            'testdata/trainer/features.json',
            '-e', 'testdata/extractorxml/train-import-handler.xml',
            '-I', 'start=2012-12-03', '-I', 'end=2012-12-04'])

        self.assertEquals(res, DONE)
        logging_mock.reset_mock()
        self.assertFalse(logging_mock.warn.called)

    @patch('trainer.logging')
    @patch('cloudml.trainer.trainer.Trainer.train')
    @patch('cloudml.trainer.trainer.Trainer.test')
    def test_with_test_params(self, test_mock, train_mock, logging_mock):
        res = main(argv=[
            'testdata/trainer/features.json',
            '-e', 'testdata/extractorxml/train-import-handler.xml',
            '-I', 'start=2012-12-03', '-I', 'end=2012-12-04',
            '-T', 'start=2013-12-03', '-T', 'end=2013-12-04'])

        self.assertEquals(res, DONE)
        logging_mock.reset_mock()
        self.assertFalse(logging_mock.warn.called)
        train_mock.assert_called_with(ANY, 0)
        test_mock.assert_called_with(ANY)

    def test_with_pretrained_transformers(self):
        res = main(argv=[
            'testdata/trainer/features-with-pretrained-transformer.json',
            '-i', 'testdata/trainer/trainer.data.json',
            '--transformer-path', 'testdata/transformers/'])
        self.assertEquals(res, DONE)
