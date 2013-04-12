__author__ = 'ifoukarakis'

"""
Created on Jan 24, 2013

@author: ifoukarakis

"""

import unittest

from core.trainer.transformers import get_transformer, ScalerDecorator
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import Scaler


class TransformersTest(unittest.TestCase):

    def test_get_transformer_count(self):
        data = {
            'type': 'Count',
            'ngram_range_min': 1,
            'ngram_range_max': 3,
            'min_df': 10, 'max_df': 20
        }
        transformer = get_transformer(data)
        self.assertIsInstance(transformer, CountVectorizer)
        self.assertEqual(transformer.ngram_range, (1, 3))
        self.assertEqual(transformer.min_df, 10)
        self.assertEqual(transformer.max_df, 20)

    def test_get_transformer_tfidf(self):
        data = {
            'type': 'Tfidf',
            'ngram_range_min': 1,
            'ngram_range_max': 3,
            'min_df': 10
        }
        transformer = get_transformer(data)
        self.assertIsInstance(transformer, TfidfVectorizer)
        self.assertEqual(transformer.ngram_range, (1, 3))
        self.assertEqual(transformer.min_df, 10)

    def test_get_transformer_dict(self):
        data = {
            'type': 'Dictionary',
            'separator': ','
        }
        transformer = get_transformer(data)
        self.assertIsInstance(transformer, DictVectorizer)
        self.assertEqual(transformer.separator, ',')

    def test_get_transformer_scaler(self):
        data = {
            'type': 'Scale',
            'with_mean': False
        }
        transformer = get_transformer(data)
        self.assertIsInstance(transformer, ScalerDecorator)
        self.assertEqual(transformer._scaler.with_mean, False)

    def test_get_transformer_unknown(self):
        data = {
            'type': 'MyCustomVectorizer',
            'ngram_range_min': 1,
            'ngram_range_max': 3,
            'min_df': 10, 'max_df': 20
        }
        self.assertIsNone(get_transformer(data))

    def test_get_transformer_no_type(self):
        data = {
            'ngram_range_min': 1,
            'ngram_range_max': 3,
            'min_df': 10, 'max_df': 20
        }
        self.assertIsNone(get_transformer(data))

if __name__ == '__main__':
    unittest.main()
