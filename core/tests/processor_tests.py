__author__ = 'ifouk'

import unittest
import os

from importhandler.processors import extract_parameters, \
    process_string, ProcessException, process_composite, process_json

BASEDIR = 'testdata'


class ProcessorCase(unittest.TestCase):
    def setUp(self):
        path = os.path.join(BASEDIR, 'extractor', 'test.data.json')
        with open(path, 'r') as fp:
            self._data = fp.read()

    def test_extract_parameters(self):
        input = '%(employer.country)s,%(contractor.country)s'
        self.assertEqual(['employer.country', 'contractor.country'],
                         extract_parameters(input))

        input = 'Another %(test)s.'
        self.assertEqual(['test'], extract_parameters(input))

        input = 'Param at the  %(end)s'
        self.assertEqual(['end'], extract_parameters(input))

        input = '%(starting)s with a param'
        self.assertEqual(['starting'], extract_parameters(input))

        input = 'Should find nothing'
        self.assertEqual([], extract_parameters(input))

        input = 'Should find nothing here %()s too'
        self.assertEqual([], extract_parameters(input))

        input = 'Even here %s nothing'
        self.assertEqual([], extract_parameters(input))

        input = 'a more complex (%(test1)s%(test2)s)) one '
        self.assertEqual(['test1', 'test2'], extract_parameters(input))

        input = 'as complex as %(%(test1)s%(test2)s))s it might get'
        self.assertEqual(['test1', 'test2'], extract_parameters(input))

    def test_process_string_valid_data(self):
        row_data = {'should', 'ignore'}
        item = {
            'source': 'testme',
            'process-as': 'string',
            'is-required': True,
            'target-features': [
                {'name': 'test.feature'}
            ]
        }
        result = process_string('abc', item, row_data)
        self.assertDictEqual(result, {'test.feature': 'abc'})

    def test_process_string_no_input_value(self):
        row_data = {'should', 'ignore'}
        item = {
            'source': 'testme',
            'process-as': 'string',
            'is-required': True,
            'target-features': [
                {'name': 'test.feature'}
            ]
        }
        result = process_string(None, item, row_data)
        self.assertDictEqual(result, {'test.feature': None})

    def test_process_string_many_targets(self):
        row_data = {'should', 'ignore'}
        item = {
            'source': 'testme',
            'process-as': 'string',
            'is-required': True,
            'target-features': [
                {'name': 'test.feature'},
                {'name': 'test.feature2'}
            ]
        }
        result = process_string('abc', item, row_data)
        self.assertDictEqual(result, {'test.feature': 'abc'})

    def test_process_expression_valid_data(self):
        row_data = {'param1': 42,
                    'param2': 'value',
                    'param3': 'test',
                    'param4': 3}
        item = {
            'process-as': 'expression',
            'target-features': [
                {
                    'name': 'test.feature1',
                    'expression': '%(param1)s,%(param2)s'
                },
                {
                    'name': 'test.feature2',
                    'expression': '%(param3)s %(param1)s'
                }
            ]
        }
        result = process_composite('should ignore', item, row_data)
        self.assertDictEqual(result, {'test.feature1': '42,value',
                                      'test.feature2': 'test 42'})

    def test_process_expression_missing_params(self):
        row_data = {'param1': 42,
                    'param2': 'value',
                    'param4': 3}
        item = {
            'process-as': 'expression',
            'target-features': [
                {
                    'name': 'test.feature1',
                    'expression': '%(param1)s,%(param2)s'
                },
                {
                    'name': 'test.feature2',
                    'expression': '%(param3)s %(param1)s'
                }
            ]
        }
        result = process_composite('should ignore', item, row_data)
        self.assertDictEqual(result, {'test.feature1': '42,value',
                                      'test.feature2': None})

    def test_process_expression_without_target_expression(self):
        row_data = {'param1': 42, 'param2': 'value'}
        item = {
            'process-as': 'expression',
            'target-features': [
                {
                    'name': 'test.feature1'
                }
            ]
        }
        try:
            process_composite('should ignore', item, row_data)
            self.fail('Should not be able to process expression when '
                      'expression is missing')
        except ProcessException:
            # Should happen
            pass

    def test_process_json(self):
        item = {
            'source': 'contractor',
            'process-as': 'json',
            'target-features': [
                {'name': 'name', 'jsonpath': '$.person_info.name'},
                {'name': 'age', 'jsonpath': '$.person_info.age'},
                {
                    'name': 'friends', 'jsonpath': '$.person_info.friends',
                    'key-path': '$.*.name', 'value-path': '$.*.race'
                },
                {'name': 'notthere', 'jsonpath': '$.notthere'},
                {
                    'name': 'friend_names1',
                    'jsonpath': '$.person_info.friends.*.name',
                    'to-csv': True
                },
                {
                    'name': 'friend_names2',
                    'jsonpath': '$.person_info.friends.*.name',
                }
            ]
        }

        result = process_json(self._data, item, {'should', 'ignore'})
        expected = {
            'name': 'Bilbo',
            'age': '111',
            'friends': {
                'Frodo': 'hobbit',
                'Thorin': 'dwarf'
            },
            'notthere': None,
            'friend_names1': 'Frodo,Thorin',
            'friend_names2': ['Frodo', 'Thorin']
        }
        self.assertDictEqual(result, expected)

if __name__ == '__main__':
    unittest.main()
