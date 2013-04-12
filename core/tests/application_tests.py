# __author__ = 'ifouk'

# try:
#     import simplejson as json
# except ImportError:
#     import json

# import logging
# import unittest
# import os

# import application

# from application import models, ERR_NO_SUCH_MODEL
# from core.trainer.trainer import Trainer

# BASEDIR = 'testdata'


# class ApplicationTest(unittest.TestCase):
#     def setUp(self):
#         logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
#                             level=logging.DEBUG)

#         application.app.config['TESTING'] = True
#         application.app.config['UPLOAD_FOLDER'] = os.path.join(BASEDIR, 'web')
#         application.init()
#         self.app = application.app.test_client()

#     def test_load_models(self):
#         """
#         Test that models have been successfully loaded.

#         """
#         self.assertNotIn('test', models)
#         self.assertIn('mymodel', models)

#     def test_get_model(self):
#         """
#         Attempt to get information about a known model.

#         """
#         rv = self.app.get('/cloudml/b/v1/model/mymodel')
#         self.assertEqual(rv.mimetype, 'application/json')
#         data = json.loads(rv.data)
#         self.assertDictEqual(data, {'name': 'mymodel',
#                                     'schema-name': 'test',
#                                     'created': '2013-02-06 18:05:21 +0000'})
#         self.assertEqual(rv.status_code, 200)

#     def test_get_model_unknown_model(self):
#         """
#         Attempt to get information about a model that doesn't exist.

#         """
#         rv = self.app.get('/cloudml/b/v1/model/notexists')
#         self.assertEqual(rv.mimetype, 'application/json')
#         self.assertEqual(rv.status_code, 404)
#         data = json.loads(rv.data)
#         self.assertEqual(data['response']['error']['status'], 404)
#         self.assertEqual(data['response']['error']['code'], ERR_NO_SUCH_MODEL)
#         self.assertEqual(rv.headers.get('X-Odesk-Error-Code'),
#                          str(ERR_NO_SUCH_MODEL))

#     def test_list_models(self):
#         """
#         Attempt to get information about a known models.

#         """
#         rv = self.app.get('/cloudml/b/v1/model')
#         self.assertEqual(rv.mimetype, 'application/json')
#         data = json.loads(rv.data)
#         self.assertEqual(rv.status_code, 200)
#         self.assertEqual(len(data['models']), 1)
#         self.assertDictEqual(data['models'][0],
#                              {'name': 'mymodel',
#                               'schema-name': 'test',
#                               'created': '2013-02-06 18:05:21 +0000'})

#     def test_evaluate_with_labels(self):
#         data = [{
#             "hire_outcome": 1,
#             "contractor.dev_adj_score_recent": 2,
#             "contractor.dev_is_looking": "1",
#             "contractor.dev_title": "python engineer",
#             "contractor.dev_recent_hours_fp": 6
#         }, {
#             "hire_outcome": 0,
#             "contractor.dev_is_looking": "1",
#             "contractor.dev_title": "python engineer",
#             "contractor.dev_recent_hours_fp": 6
#         }]
#         rv = self.app.post('/cloudml/b/v1/model/mymodel/evaluate',
#                            content_type='application/json',
#                            data=json.dumps(data))
#         self.assertEqual(rv.status_code, 200)
#         data = json.loads(rv.data)
#         self.assertEqual(2, len(data['probabilities']))

#     def test_evaluate_without_labels(self):
#         data = [{
#             "contractor.dev_adj_score_recent": 2,
#             "contractor.dev_is_looking": "1",
#             "contractor.dev_title": "python engineer",
#             "contractor.dev_recent_hours_fp": 6
#         }, {
#             "contractor.dev_is_looking": "1",
#             "contractor.dev_title": "python engineer",
#             "contractor.dev_recent_hours_fp": 6
#         }]
#         rv = self.app.post('/cloudml/b/v1/model/mymodel/evaluate',
#                            content_type='application/json',
#                            data=json.dumps(data))
#         self.assertEqual(rv.status_code, 200)
#         data = json.loads(rv.data)
#         self.assertEqual(2, len(data['probabilities']))

#     def test_evaluate_incorrect_content_type(self):
#         data = [{
#             "contractor.dev_adj_score_recent": 2,
#             "contractor.dev_is_looking": "1",
#             "contractor.dev_title": "python engineer",
#             "contractor.dev_recent_hours_fp": 6
#         }, {
#             "contractor.dev_is_looking": "1",
#             "contractor.dev_title": "python engineer",
#             "contractor.dev_recent_hours_fp": 6
#         }]
#         rv = self.app.post('/cloudml/b/v1/model/mymodel/evaluate',
#                            content_type='text/plain',
#                            data=json.dumps(data))
#         self.assertEqual(rv.status_code, 405)


# if __name__ == '__main__':
#     unittest.main()
