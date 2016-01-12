# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>


import unittest

from cloudml.trainer.transformers import get_transformer, Ntile
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class TransformersTest(unittest.TestCase):

    def test_ntile(self):
        data = {
            'type': 'Ntile',
            'params': {'number_tile': 4}
        }
        transformer = get_transformer(data)
        self.assertIsInstance(transformer, Ntile)
        self.assertEqual(transformer.number_tile, 4)
        X = [1, 2, 3, 7, 78, 8, 35235, 353, 3555, 3535, 3657, 6868, 865]
        X1 = [1, 2, 3, 7, 78, 8, 35235, 353, 3555, 3535, 3657, 6868, 865, 4,
              66, 342323]
        transformer.fit(X)
        Y = transformer.transform(X)
        self.assertEqual(Y.transpose().todense().tolist()[0],
                         [1, 1, 1, 1, 2, 2, 4, 2, 3, 3, 4, 4, 3])
        Y = transformer.transform(X1)
        self.assertEqual(Y.transpose().todense().tolist()[0],
                         [1, 1, 1, 1, 2, 2, 4, 2, 3, 3, 4, 4, 3, 1, 2, 4])

    def test_get_transformer_count(self):
        data = {
            'type': 'Count',
            'params': {'ngram_range_min': 1,
                       'ngram_range_max': 3,
                       'min_df': 10, 'max_df': 20}
        }
        transformer = get_transformer(data)
        self.assertIsInstance(transformer, CountVectorizer)
        self.assertEqual(transformer.ngram_range, (1, 3))
        self.assertEqual(transformer.min_df, 10)
        self.assertEqual(transformer.max_df, 20)

    def test_get_transformer_tfidf(self):
        data = {
            'type': 'Tfidf',
            'params': {'ngram_range_min': 1,
                       'ngram_range_max': 3,
                       'min_df': 10}
        }
        transformer = get_transformer(data)
        self.assertIsInstance(transformer, TfidfVectorizer)
        self.assertEqual(transformer.ngram_range, (1, 3))
        self.assertEqual(transformer.min_df, 10)

    def test_get_transformer_dict(self):
        data = {
            'type': 'Dictionary',
            'params': {'separator': ','}
        }
        transformer = get_transformer(data)
        self.assertIsInstance(transformer, DictVectorizer)
        self.assertEqual(transformer.separator, ',')

#    def test_get_transformer_scaler(self):
#        data = {
#            'type': 'Scale',
#            'with_mean': False
#        }
#        transformer = get_transformer(data)
#        self.assertIsInstance(transformer, ScalerDecorator)
#        self.assertEqual(transformer._scaler.with_mean, False)

    def test_get_transformer_unknown(self):
        data = {
            'type': 'MyCustomVectorizer',
            'params': {'ngram_range_min': 1,
                       'ngram_range_max': 3,
                       'min_df': 10, 'max_df': 20}
        }
        self.assertIsNone(get_transformer(data))

    def test_get_transformer_no_type(self):
        data = {
            'params': {
                'ngram_range_min': 1,
                'ngram_range_max': 3,
                'min_df': 10, 'max_df': 20}
        }
        self.assertIsNone(get_transformer(data))

if __name__ == '__main__':
    unittest.main()
