import logging
import json


from core.trainer.transformers import get_transformer
from core.trainer.feature_types import FEATURE_TYPE_FACTORIES


class TransformerSchemaException(Exception):
    """
    Exception to be raised if there is an error parsing or using the
    configuration.

    """

    def __init__(self, message, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        # Now for your custom code...
        self.Errors = Errors


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

        if not 'transformer-name' in data:
            raise TransformerSchemaException(message="transformer-name is missing")

        self.name = data['transformer-name']
        self.type = data['type']

        # Get transformer
        transformer_config = data.get('transformer', None)
        transformer_type = None
        if transformer_config is not None:
            transformer_type = transformer_config.get('type')
        transformer = get_transformer(transformer_config)

        factory = FEATURE_TYPE_FACTORIES.get(data['type'], None)

        if factory is None:
            raise TransformerSchemaException('Unknown type: %s' % (data['type']))

        try:
            feature_type = factory.get_instance(data.get('params', None),
                                        data.get('input-format', 'plain'))
        except:
            raise TransformerSchemaException('Feature type error: %s' % (data['type']))

        self.feature = {'name': data['field-name'],
                        'type': feature_type,
                        'transformer-type': transformer_type,
                        'transformer': transformer}

    def train(self, iterator):
        logging.info('Start train transformer "%s"' % self.name)
        self._prepare_data(iterator)
        self.feature['transformer'].fit(self._vect_data)
        self.voc_size = len(self.feature['transformer'].vocabulary_)
        logging.info('Vocabulary size: %d' % self.voc_size)
        logging.info('Train completed')


    def transform(self, data):
        return self.feature['transformer'].transform(data)

    def _prepare_data(self, iterator, ignore_error=True):
        self._count = 0
        self._ignored = 0
        self._vect_data = []
        for row in iterator:
            if self._count % 1000 == 0:
                logging.info('Processed %s rows so far' % (self._count, ))
            self._count += 1
            try:
                ft = self.feature.get('type', None)
                item = row.get(self.feature['name'], None)
                data = ft.transform(item)
                self._vect_data.append(data)
            except Exception, e:
                raise
                logging.debug('Ignoring item #%d: %s'
                              % (self._count, e))
                if ignore_error:
                    self._ignored += 1
                else:
                    raise e



