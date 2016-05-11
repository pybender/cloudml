"""
Module gathers classes and methods for implementing
training and using the pretrained transformers.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import logging
import json

from cloudml.trainer.transformers import get_transformer
from cloudml.trainer.feature_types import FEATURE_TYPE_FACTORIES


class TransformerSchemaException(Exception):
    """
    Exception to be raised if there is an error parsing or using the
    configuration.
    """
    def __init__(self, message, errors=None):
        super(TransformerSchemaException, self).__init__(message)
        self.errors = errors


class Transformer(object):

    def __init__(self, config, is_file=True):
        try:
            if is_file:
                with open(config, 'r') as fp:
                    data = json.load(fp)
            else:
                data = json.loads(config)
        except ValueError as e:
            raise TransformerSchemaException(message='%s %s ' % (config, e))

        if 'transformer-name' not in data:
            raise TransformerSchemaException(
                message="transformer-name is missing")

        self.name = data['transformer-name'].strip(' \t\n\r')
        if not self.name:
            raise TransformerSchemaException(
                message="transformer-name is missing")

        self.type = data['type']

        # Get transformer
        transformer_config = data.get('transformer', None)
        transformer_type = None
        if transformer_config is not None:
            transformer_type = transformer_config.get('type')
        transformer = get_transformer(transformer_config)

        factory = FEATURE_TYPE_FACTORIES.get(data['type'], None)

        if factory is None:
            raise TransformerSchemaException('Unknown type: %s'
                                             % (data['type']))

        try:
            feature_type = factory.get_instance(data.get('params', None),
                                                data.get('input-format',
                                                         'plain'))
        except:
            raise TransformerSchemaException('Feature type error: %s'
                                             % (data['type']))

        self.feature = {'name': data['field-name'],
                        'type': feature_type,
                        'transformer-type': transformer_type,
                        'transformer': transformer}
        self.voc_size = None

    def train(self, iterator):
        logging.info('Start train transformer "%s"' % self.name)
        self._prepare_data(iterator)
        transformer = self.feature['transformer']
        transformer.fit(self._vect_data)
        if hasattr(transformer, 'vocabulary_'):
            self.voc_size = len(transformer.vocabulary_)
            logging.info('Vocabulary size: %d' % self.voc_size)
        logging.info('Train completed')

    def transform(self, data):
        return self.feature['transformer'].transform(data)

    def _prepare_data(self, iterator, ignore_error=True):
        self._count = 0
        self._ignored = 0
        self._vect_data = []
        for row in iterator:
            self._count += 1
            try:
                ft = self.feature.get('type', None)
                item = row.get(self.feature['name'], None)
                if not item:
                    raise ValueError("No data for %s feature in row" %
                                     self.feature['name'])
                data = ft.transform(item)
                self._vect_data.append(data)
            except Exception, e:
                logging.debug('Ignoring item #%d: %s'
                              % (self._count, e))
                if ignore_error:
                    self._ignored += 1
                else:
                    raise e
