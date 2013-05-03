import unittest
import json

from api import app
from api.utils import ERR_NO_SUCH_MODEL, ERR_NO_SUCH_IMPORT_HANDLER


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_server_health(self):
        rv = self.app.get('/cloudml/server_health.json')
        self.assertEqual(rv.mimetype, 'application/json')
        data = json.loads(rv.data)
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(data['Overall_health'], "OK")
        self.assertEqual(data['resources'][0]['Models_Loaded'], "OK")
        self.assertEqual(data['resources'][1]['ImportHandlers_Loaded'], "OK")

        app.models = {}
        rv = self.app.get('/cloudml/server_health.json')
        self.assertEqual(rv.mimetype, 'application/json')
        data = json.loads(rv.data)
        self.assertEqual(rv.status_code, 200)
        self.assertEqual(data['resources'][0]['Models_Loaded'], "ERR")
        self.assertEqual(data['Overall_health'], "ERR")

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
        self.assertEqual(data['prediction'], True)
        self.assertEqual(data['probs'], [
            {u'label': False, u'prob': 0.30698654913887147},
            {u'label': True, u'prob': 0.6930134508611285}
        ])

    def test_predict_no_such_model(self):
        rv = self.app.post('/cloudml/model/model1/extract/predict')
        self.assertEqual(rv.mimetype, 'application/json')
        data = json.loads(rv.data)['response']['error']
        self.assertEqual(data['message'], "Model model1 doesn't exist")
        self.assertEqual(data['code'], ERR_NO_SUCH_MODEL)
        self.assertEqual(rv.status_code, 404)

    def test_predict_no_such_modelhandler(self):
        rv = self.app.post('/cloudml/model/model/extract1/predict')
        self.assertEqual(rv.mimetype, 'application/json')
        data = json.loads(rv.data)['response']['error']
        self.assertEqual(data['message'],
                         "Import handler extract1 doesn't exist")
        self.assertEqual(data['code'], ERR_NO_SUCH_IMPORT_HANDLER)
        self.assertEqual(rv.status_code, 404)
