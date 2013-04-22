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
        self.assertTrue('models' in data)
        count = app.db.Model.find().count()
        self.assertEquals(count, len(data['models']))

    def test_post_new_model_with_invalid_data(self):
        uri = '/cloudml/model/new'
        self._checkValidationErrors(uri, {}, 'importhandler is required in \
values')

        post_data = {'importhandler': 'smth'}
        self._checkValidationErrors(uri, post_data, 'Either features, either \
pickled trained model is required')

        post_data = {'importhandler': 'smth', 'features': 'smth'}
        self._checkValidationErrors(uri, post_data, 'Invalid features: \
smth No JSON object could be decoded ')

        post_data = {'importhandler': 'smth', 'features': '{}'}
        self._checkValidationErrors(uri, post_data, 'Invalid features: \
schema-name is missing')

        features = open('./conf/features.json', 'r').read()
        post_data = {'importhandler': 'smth',
                     'features': features}
        self._checkValidationErrors(uri, post_data, 'Invalid Import Handler: \
No JSON object could be decoded')

        features = open('./conf/features.json', 'r').read()
        post_data = {'importhandler': '{}',
                     'features': features}
        self._checkValidationErrors(uri, post_data, 'Invalid Import Handler: \
No target schema defined in config')

    def test_post_new_model(self):
        name = 'UnitTestModel1'
        features = open('./conf/features.json', 'r').read()
        handler = open('./conf/extract.json', 'r').read()
        post_data = {'importhandler': handler,
                     'features': features}
        resp = self.app.post('/cloudml/model/%s' % name, data=post_data)
        self.assertEquals(resp.status_code, httplib.CREATED)
        self.assertTrue('model' in resp.data)

    # def test_post_trained_model(self):
    #     name = 'UnitTestModel2'
    #     from werkzeug.datastructures import FileStorage
    #     trainer = None
    #     handler = open('./conf/extract.json', 'r').read()
    #     post_data = {'importhandler': handler,
    #                  'trainer': trainer}
    #     resp = self.app.post('/cloudml/model/%s' % name, data=post_data)
    #     print resp.data
    #     self.assertEquals(resp.status_code, httplib.CREATED)
    #     self.assertTrue('model' in resp.data)

    def _checkValidationErrors(self, uri, post_data, message,
                               code=ERR_INVALID_DATA,
                               status_code=httplib.BAD_REQUEST):
        resp = self.app.post(uri, data=post_data)
        self.assertEquals(resp.status_code, status_code)
        data = json.loads(resp.data)
        err_data = data['response']['error']
        self.assertEquals(err_data['code'], code)
        self.assertEquals(err_data['message'], message)
