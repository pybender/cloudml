import unittest
import json
import httplib

from api.models import Model
from api import app
from api.utils import ERR_INVALID_DATA


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def tearDown(self):
        pass

    def test_list(self):
        resp = self.app.get('/cloudml/model/')
        self.assertEquals(resp.status_code, httplib.OK)
        data = json.loads(resp.data)
        self.assertTrue('models' in data)

    def test_post_with_invalid_data(self):
        name = 'new'
        resp = self.app.post('/cloudml/model/%s' % name)
        self.assertEquals(resp.status_code, httplib.BAD_REQUEST)
        data = json.loads(resp.data)
        err_data = data['response']['error']
        self.assertEquals(err_data['code'], ERR_INVALID_DATA)
        self.assertEquals(err_data['message'],
                          'importhandler is required in values')

        post_data = {'importhandler': 'smth'}
        resp = self.app.post('/cloudml/model/%s' % name, data=post_data)
        self.assertEquals(resp.status_code, httplib.BAD_REQUEST)
        data = json.loads(resp.data)
        err_data = data['response']['error']
        self.assertEquals(err_data['code'], ERR_INVALID_DATA)
        self.assertEquals(err_data['message'],
                          'Either features, either pickled trained model is required')
