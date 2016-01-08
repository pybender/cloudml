"""
Unittests for pretrained transformers.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import unittest
from mock import MagicMock, patch
import os

from transformer import Transformer, TransformerSchemaException
from cloudml.trainer.streamutils import streamingiterload


BASEDIR = 'testdata'


class TransformerTestCase(unittest.TestCase):

    def setUp(self):
        self._config = _get_config('transformer.json')

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

    def test_invalid_schema(self):
        config = _get_config('empty_name.json')
        with self.assertRaisesRegexp(TransformerSchemaException,
                                     "transformer-name is missing"):
            Transformer(config)

        config = _get_config('invalid.json')
        with self.assertRaisesRegexp(TransformerSchemaException,
                                     "No JSON object could be decoded"):
            Transformer(config)

        config = _get_config('invalid_type.json')
        with self.assertRaisesRegexp(TransformerSchemaException,
                                     "Unknown type: invalid"):
            Transformer(config)


def _get_config(file_name):
    return os.path.join(BASEDIR, 'transformers', file_name)
