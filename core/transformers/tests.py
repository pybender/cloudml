import unittest
from mock import MagicMock, patch
import os

from transformer import Transformer
from core.trainer.streamutils import streamingiterload


BASEDIR = 'testdata'


class TransformerTestCase(unittest.TestCase):

    def setUp(self):
        self._config = os.path.join(BASEDIR, 'transformers',
                                    'transformer.json')

    def _get_iterator(self, fmt='json'):
        with open(os.path.join(BASEDIR, 'transformers',
                               'train.data.{}'.format(fmt))) as fp:
            self._data = list(streamingiterload(
                fp.readlines(), source_format=fmt))
        return self._data

    def test_train_transformer(self):
        transformer = Transformer(self._config)
        transformer.train(self._get_iterator())
        print transformer.transform(["Python", "engineer", "eee", "time"])
