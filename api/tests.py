import unittest
import os
import json
import httplib

from api.models import Model
from api.serialization import encode_model
from api import app
from api.utils import ERR_INVALID_DATA


class BaseTestCase(unittest.TestCase):
    FIXTURES = []
    _LOADED_COLLECTIONS = []

    def setUp(self):
        self.app = app.test_client()
        self.fixtures_load()

    def tearDown(self):
        self.fixtures_cleanup()

    @property
    def db(self):
        return app.db

    def fixtures_load(self):
        for fixture in self.FIXTURES:
            data = self._load_fixture_data(fixture)
            for collection_name, documents in data.iteritems():
                self._LOADED_COLLECTIONS.append(collection_name)
                collection = self._get_collection(collection_name)
                collection.insert(documents)

    def fixtures_cleanup(self):
        for collection_name in self._LOADED_COLLECTIONS:
            collection = self._get_collection(collection_name)
            collection.remove()

    def _load_fixture_data(self, filename):
        filename = os.path.join('./api/fixtures/', filename)
        content = open(filename, 'rb').read()
        return json.loads(content)

    def _get_collection(self, name):
        callable_model = getattr(self.db, name)
        return callable_model.collection


class ModelTests(BaseTestCase):
    FIXTURES = ('models.json', )

    def test_list(self):
        resp = self.app.get('/cloudml/model/')
        self.assertEquals(resp.status_code, httplib.OK)
        data = json.loads(resp.data)
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


def dumpdata(document_list, fixture_name):
    content = json.dumps(document_list, default=encode_model)
    file_path = os.path.join('./api/fixtures/', fixture_name)
    with open(file_path, 'w') as ffile:
        ffile.write(content)


class TestTests(BaseTestCase):
    MODEL_NAME = 'TrainedModel'
    TEST_NAME = 'Test-1'
    FIXTURES = ('models.json', 'tests.json')

    def test_list(self):
        url = self._get_url(self.MODEL_NAME, search='show=name,status')
        resp = self.app.get(url)
        self.assertEquals(resp.status_code, httplib.OK)
        data = json.loads(resp.data)
        self.assertTrue('tests' in data)
        tests = app.db.Test.find({'model_name': self.MODEL_NAME})
        count = tests.count()
        self.assertEquals(count, len(data['tests']))
        self.assertTrue(tests[0].name in resp.data, resp.data)
        self.assertTrue(tests[0].status in resp.data, resp.data)
        self.assertFalse(tests[0].model_name in resp.data, resp.data)

    def test_details(self):
        url = self._get_url(self.MODEL_NAME, self.TEST_NAME, 'show=name,status')
        resp = self.app.get(url)
        self.assertEquals(resp.status_code, httplib.OK)
        data = json.loads(resp.data)
        self.assertTrue('test' in data, data)
        test_data = data['test']
        test = app.db.Test.find_one({'model_name': self.MODEL_NAME,
                                     'name': self.TEST_NAME})
        self.assertEquals(test.name, test_data['name'], resp.data)
        self.assertEquals(test.status, test_data['status'], resp.data)
        self.assertFalse('model_name' in test_data, test_data)

    def test_delete(self):
        url = self._get_url(self.MODEL_NAME, self.TEST_NAME, 'show=name,status')
        resp = self.app.get(url)
        self.assertEquals(resp.status_code, httplib.OK)

        resp = self.app.delete(url)
        self.assertEquals(resp.status_code, 204)
        test = app.db.Test.find_one({'model_name': self.MODEL_NAME,
                                     'name': self.TEST_NAME})
        self.assertEquals(test, None, test)

    def _get_url(self, model, test=None, search=None):
        if test:
            return '/cloudml/model/%s/test/%s?%s' % (model, test, search)
        else:
            return '/cloudml/model/%s/tests?%s' % (model, search)
