# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>


import unittest
from cloudml.trainer.utils import copy_expected, is_empty


class UtilsTestCase(unittest.TestCase):
    def test_copy_expected(self):
        params = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        expected = ['a', 'd']
        result = copy_expected(params, expected)
        self.assertDictEqual(result, {'a': 1, 'd': 4})

        params = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
        expected = ['e', 'f']
        result = copy_expected(params, expected)
        self.assertDictEqual(result, {})

    def test_is_empty(self):
        self.assertFalse(is_empty(False))
        self.assertFalse(is_empty(True))
        self.assertFalse(is_empty(42))
        self.assertFalse(is_empty(0))
        self.assertFalse(is_empty(42.0))
        self.assertFalse(is_empty(0.0))
        self.assertFalse(is_empty(' '))
        self.assertFalse(is_empty((1)))

        self.assertTrue(is_empty(''))
        self.assertTrue(is_empty(None))
        self.assertTrue(is_empty({}))
        self.assertTrue(is_empty([]))
        self.assertTrue(is_empty(()))

if __name__ == '__main__':
    unittest.main()
