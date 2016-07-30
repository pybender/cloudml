# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>

# import unittest
# import os

# from core.importhandler.processors import extract_parameters, \
#     process_primitive, ProcessException, process_composite, process_json

# from core.importhandler.scripts import ScriptManager

# BASEDIR = 'testdata'


# class ProcessorCase(unittest.TestCase):
#     def setUp(self):
#         path = os.path.join(BASEDIR, 'extractor', 'test.data.json')
#         with open(path, 'r') as fp:
#             self._data = fp.read()

#     def test_extract_parameters(self):
#         input = '%(employer.country)s,%(contractor.country)s'
#         self.assertEqual(['employer.country', 'contractor.country'],
#                          extract_parameters(input))

#         input = 'Another %(test)s.'
#         self.assertEqual(['test'], extract_parameters(input))

#         input = 'Param at the  %(end)s'
#         self.assertEqual(['end'], extract_parameters(input))

#         input = '%(starting)s with a param'
#         self.assertEqual(['starting'], extract_parameters(input))

#         input = 'Should find nothing'
#         self.assertEqual([], extract_parameters(input))

#         input = 'Should find nothing here %()s too'
#         self.assertEqual([], extract_parameters(input))

#         input = 'Even here %s nothing'
#         self.assertEqual([], extract_parameters(input))

#         input = 'a more complex (%(test1)s%(test2)s)) one '
#         self.assertEqual(['test1', 'test2'], extract_parameters(input))

#         input = 'as complex as %(%(test1)s%(test2)s))s it might get'
#         self.assertEqual(['test1', 'test2'], extract_parameters(input))

#     def test_process_string_valid_data(self):
#         row_data = {'should', 'ignore'}
#         item = {
#             'source': 'testme',
#             'process_as': 'string',
#             'is_required': True,
#             'target_features': [
#                 {'name': 'test.feature'}
#             ]
#         }
#         result = process_primitive(str)('abc', item, row_data)
#         self.assertDictEqual(result, {'test.feature': 'abc'})

#     def test_process_string_no_input_value(self):
#         row_data = {'should', 'ignore'}
#         item = {
#             'source': 'testme',
#             'process_as': 'string',
#             'is_required': True,
#             'target_features': [
#                 {'name': 'test.feature'}
#             ]
#         }
#         result = process_primitive(str)(None, item, row_data)
#         self.assertDictEqual(result, {'test.feature': None})

#     def test_process_string_many_targets(self):
#         row_data = {'should', 'ignore'}
#         item = {
#             'source': 'testme',
#             'process_as': 'string',
#             'is_required': True,
#             'target_features': [
#                 {'name': 'test.feature'},
#                 {'name': 'test.feature2'}
#             ]
#         }
#         result = process_primitive(str)('abc', item, row_data)
#         self.assertDictEqual(result, {'test.feature': 'abc'})

#     def test_process_expression_valid_data(self):
#         row_data = {'param1': 42,
#                     'param2': 'value',
#                     'param3': 'test',
#                     'param4': 3}
#         item = {
#             'process_as': 'expression',
#             'target_features': [
#                 {
#                     'name': 'test.feature1',
#                     'expression':  {
#                         "type": "string",
#                         "value": '%(param1)s,%(param2)s'
#                     }
#                 },
#                 {
#                     'name': 'test.feature2',
#                     'expression':  {
#                         "type": "string",
#                         "value": '%(param3)s %(param1)s'
#                     }
#                 }
#             ]
#         }
#         result = process_composite('should ignore', item, row_data)
#         self.assertDictEqual(result, {'test.feature1': '42,value',
#                                       'test.feature2': 'test 42'})

#     def test_process_expression_missing_params(self):
#         row_data = {'param1': 42,
#                     'param2': 'value',
#                     'param4': 3}
#         item = {
#             'process_as': 'expression',
#             'target_features': [
#                 {
#                     'name': 'test.feature1',
#                     'expression': {
#                         "type": "string",
#                         "value": '%(param1)s,%(param2)s'
#                     }
#                 },
#                 {
#                     'name': 'test.feature2',
#                     'expression': {
#                         "type": "string",
#                         "value": '%(param3)s %(param1)s'
#                     }
#                 }
#             ]
#         }
#         result = process_composite('should ignore', item, row_data)
#         self.assertDictEqual(result, {'test.feature1': '42,value',
#                                       'test.feature2': None})

#     def test_process_expression_without_target_expression(self):
#         row_data = {'param1': 42, 'param2': 'value'}
#         item = {
#             'process_as': 'expression',
#             'target_features': [
#                 {
#                     'name': 'test.feature1'
#                 }
#             ]
#         }
#         try:
#             process_composite('should ignore', item, row_data)
#             self.fail('Should not be able to process expression when '
#                       'expression is missing')
#         except ProcessException:
#             # Should happen
#             pass

#     def test_process_json(self):
#         item = {
#             'source': 'contractor',
#             'process_as': 'json',
#             'target_features': [
#                 {'name': 'name', 'jsonpath': '$.person_info.name'},
#                 {'name': 'age', 'jsonpath': '$.person_info.age'},
#                 {
#                     'name': 'friends', 'jsonpath': '$.person_info.friends',
#                     'key_path': '$.*.name', 'value_path': '$.*.race'
#                 },
#                 {'name': 'notthere', 'jsonpath': '$.notthere'},
#                 {
#                     'name': 'friend_names1',
#                     'jsonpath': '$.person_info.friends.*.name',
#                     'to_csv': True
#                 },
#                 {
#                     'name': 'friend_names2',
#                     'jsonpath': '$.person_info.friends.*.name',
#                 }
#             ]
#         }

#         result = process_json(self._data, item, {'should', 'ignore'})
#         expected = {
#             'name': u'Bilbo',
#             'age': u'111',
#             'friends': {
#                 'Frodo': 1.0,
#                 'Thorin': 11.0,
#             },
#             'notthere': None,
#             'friend_names1': u'Frodo,Thorin',
#             'friend_names2': [u'Frodo', u'Thorin']
#         }
#         print result
#         self.assertDictEqual(result, expected)

#     def test_process_readability(self):
#         row_data = {'text': """We are close to wrapping up our 10
# week Rails Course. This week we will cover a handful of topics
# commonly encountered in Rails projects. We then wrap up with part
# 2 of our Reddit on Rails exercise!  By now you should be hard at
# work on your personal projects. The students in the course just
# presented in front of the class with some live demos and a brief
# intro to to the problems their app were solving. Maybe set aside
# some time this week to show someone your progress, block off 5
# minutes and describe what goal you are working towards, the
# current state of the project (is it almost done, just getting
# started, needs UI, etc.), and then show them a quick demo
# of the app. Explain what type of feedback you are looking for
# (conceptual, design, usability, etc.) and see what they have
# to say.  As we are wrapping up the course you need to be
# focused on learning as much as you can, but also making sure
# you have the tools to succeed after the class is over."""}
#         item = {
#             'target_features': [
#                 {
#                     'name': 'test.feature1',
#                     'expression': {
#                         "type": "readability",
#                         "value": '%(text)s',
#                         "readability_type": 'flesch_reading_ease'
#                     }
#                 },
#             ]
#         }
#         result = process_composite('should ignore', item, row_data)
#         self.assertDictEqual(result, {'test.feature1': 93.898})

#         item['target_features'][0]['expression']['readability_type'] = \
#             'coleman_liau_index'
#         result = process_composite('should ignore', item, row_data)
#         self.assertDictEqual(result, {'test.feature1': 5.6121})

#         # encoding test
#         row_data = {'text': """\u2019 We are close to wrapping up our
# 10 week Rails Course. This week we will cover a handful of topics
# commonly encountered in Rails projects. We then wrap up with part 2
# of our Reddit on Rails exercise!  By now you should be hard at work
# on your personal projects. The students in the course just presented
# in front of the class with some live demos and a brief intro to to
# the problems their app were solving. Maybe set aside some time this
# week to show someone your progress, block off 5 minutes and describe
# what goal you are working towards, the current state of the project (is
#  it almost done, just getting started, needs UI, etc.), and then show
# them a quick demo of the app. Explain what type of feedback you are
# looking for (conceptual, design, usability, etc.) and see what they
# have to say.  As we are wrapping up the course you need to be focused
# on learning as much as you can, but also making sure you have the
# tools to succeed after the class is over."""}
#         result = process_composite('should ignore', item, row_data)
#         self.assertDictEqual(result, {'test.feature1': 5.6042})

#     def test_process_expression_encoding(self):
#         row_data = {'param1': u'\u2019 value', 'param2': 'test'}
#         item = {
#             'process_as': 'expression',
#             'target_features': [
#                 {
#                     'name': 'test.feature1',
#                     'expression': {
#                         "type": "string",
#                         "value": '%(param1)s,%(param2)s'
#                     }
#                 }
#             ]
#         }
#         result = process_composite('should ignore', item, row_data)
#         self.assertDictEqual(result,
#                              {'test.feature1': '\xe2\x80\x99 value,test'})

#         row_data = {'param1': u'\u2019 value', 'param2': 'test'}
#         item = {
#             'process_as': 'expression',
#             'target_features': [
#                 {
#                     'name': 'test.feature1',
#                     'expression': {
#                         "type": "python",
#                         "value": '"%(param1)s " + str(len("%(param2)s"))'
#                     }
#                 }
#             ]
#         }

#         result = process_composite('should ignore', item, row_data)
#         self.assertDictEqual(result, {'test.feature1':
#                                       '\xe2\x80\x99 value 4'})
#         script_manager = ScriptManager()
#         script_manager.add_python("""def intToBoolean(a):
#             return a == 1
#         """)
#         row_data = {'param1': u'\u2019 value',
#                     'param2': 'test',
#                     'param3': ['1', '2', '3']}
#         item = {
#             'process_as': 'expression',
#             'target_features': [
#                 {
#                     'name': 'test.feature1',
#                     'expression': {
#                         "type": "newpython",
#                         "value": "'#{param1} ' + str(len('#{param2}')) + \
# ', '.join(#{param3})+ str(intToBoolean(1))"
#                     }
#                 }
#             ]
#         }
#         result = process_composite('should ignore',
#                                    item,
#                                    row_data,
#                                    script_manager)
#         self.assertDictEqual(result,
#                              {'test.feature1': '\xe2\x80\x99 \
# value 41, 2, 3True'})

# if __name__ == '__main__':
#     unittest.main()
