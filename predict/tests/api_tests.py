import unittest
import json
import httplib

from api import app
from api.utils import ERR_NO_SUCH_MODEL


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_model_list(self):
        rv = self.app.get('/cloudml/model')
        self.assertEqual(rv.mimetype, 'application/json')
        data = json.loads(rv.data)
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(len(data['models']), 1)
        self.assertDictEqual(data['models'][0],
                             {'name': 'model',
                              'schema-name': 'bestmatch',
                              'created': '2013-04-19 06:41:28 +0000'})

    def test_import_handler_list(self):
        rv = self.app.get('/cloudml/import/handler')
        self.assertEqual(rv.mimetype, 'application/json')
        data = json.loads(rv.data)
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(len(data['import_handlers']), 1)
        self.assertDictEqual(data['import_handlers'][0],
                             {'name': 'extract'})

    def test_predict(self):
        rv = self.app.post('/cloudml/model/model/extract/predict')
        self.assertEqual(rv.mimetype, 'application/json')
        data = json.loads(rv.data)
        self.assertEqual(rv.status_code, 201)
