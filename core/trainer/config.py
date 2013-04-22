"""
Created on Dec 13, 2012

@author: ifoukarakis
"""

import json
import importlib

from featuretype import FEATURE_TYPE_FACTORIES
from featuretype import InvalidFeatureTypeException
from utils import copy_expected
from transformers import get_transformer
from scalers import get_scaler
from collections import OrderedDict
from classifier_settings import CLASSIFIERS


class SchemaException(Exception):
    """
    Exception to be raised if there is an error parsing or using the
    configuration.

    """

    def __init__(self, message, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        # Now for your custom code...
        self.Errors = Errors


class FeatureModel(object):
    """
    Reads training data configuration from a file containing a JSON object.

    """
    def __init__(self, config, is_file=True):
        try:
            if is_file:
                with open(config, 'r') as fp:
                    data = json.load(fp)
            else:
                data = json.loads(config)
        except ValueError as e:
            raise SchemaException(message='%s %s ' % (config, e))

        if not 'schema-name' in data:
            raise SchemaException(message="schema-name is missing")

        self.schema_name = data['schema-name']
        self.classifier = {}
        self.target_variable = None
        self._named_feature_types = {}
        self.features = OrderedDict()
        self.required_feature_names = []

        self._process_classifier(data)

        # Add feature types defined in 'feature-types section
        if 'feature-types' in data:
            for feature_type in data['feature-types']:
                self._process_named_feature_type(feature_type)

        for feature in data['features']:
            self._process_feature(feature)

        if self.target_variable is None:
            raise SchemaException('No target variable defined!!!')

    def _process_classifier(self, config):
        """
        Reads config for classifier and stores it to classifier attribute.

        Keyword arguments
        config -- a dictionary containing the configuration of the feature type

        """
        classifier_config = config.get('classifier', None)
        if classifier_config is None:
            raise SchemaException('No classifier configuration defined')

        self.classifier_type = classifier_config.get('type')
        if self.classifier_type not in CLASSIFIERS.keys():
            raise SchemaException('Invalid classifier type')

        # Filter only valid parameters
        classifier_settings = CLASSIFIERS[self.classifier_type]

        defaults = classifier_settings.get('defaults', {})
        self.classifier.update(defaults)

        parameters = classifier_settings.get('parameters', [])
        self.classifier.update(copy_expected(classifier_config, parameters))

        # Trying to load classifier class
        module, name = classifier_settings.get('cls').rsplit(".", 1)
        module = importlib.import_module(module)
        self.classifier_cls = getattr(module, name)

    def _process_named_feature_type(self, config):
        """
        Used for processing named feature types. Named feature types are the
        ones defined in "feature-type" part of the configuration.

        Keyword arguments
        config -- a dictionary containing the configuration of the feature type

        """

        # Check if named feature type has a name
        if 'name' not in config:
            raise SchemaException('Feature types must have a name')

        # Check if named feature type is not already defined
        if config['name'] in self._named_feature_types:
            raise SchemaException('Feature %s already defined'
                                  % (config['name']))

        feature_type_instance = self._process_feature_type(config)

        self._named_feature_types[config['name']] = feature_type_instance

    def _process_feature_type(self, config):
        """
        Creates the actual feature type parser object using the given
        configuration.

        Keyword arguments
        config -- a dictionary containing the configuration of the feature type

        """
        if 'type' not in config:
            raise SchemaException('Type not set on individual feature type')

        factory = FEATURE_TYPE_FACTORIES.get(config['type'], None)

        if factory is None:
            raise SchemaException('Unknown type: %s' % (config['type']))

        try:
            return factory.get_instance(config.get('params', None),
                                        config.get('input-format', 'plain'))
        except InvalidFeatureTypeException, e:
            raise SchemaException('Cannot create instance of feature type', e)

    def _process_feature(self, feature):
        '''
        Validate each feature.
        '''
        if 'name' not in feature:
            raise SchemaException('Features must have a name')

        if 'type' not in feature:
            raise SchemaException('Feature %s must have a type' % 
                                  feature['name'])

        # Check if feature has a type definition
        feature_type = None
        if 'type' in feature:
            # Check if it is a named type
            feature_type = self._named_feature_types.get(feature['type'], None)

            # If not a named type, try to get a new instance
            if feature_type is None:
                feature_type = self._process_feature_type(feature)

        # Check if it is a target variable
        if feature.get('is-target-variable', False) is True:
            self.target_variable = feature['name']

        # Get Scaler
        default_scaler = feature_type.default_scaler
        scaler_config = feature.get('scaler', None)
        scaler = get_scaler(scaler_config, default_scaler)

        # Get 'input-format'
        input_format = feature.get('input-format', 'plain')

        # Get transformer
        transformer_config = feature.get('transformer', None)
        transformer_type = None
        if transformer_config is not None:
            transformer_type = transformer_config.get('type')
        transformer = get_transformer(transformer_config)

        required = feature.get('is-required', True)
        default = feature.get('default', None)
        if required:
            self.required_feature_names.append(feature['name'])
        self.features[feature['name']] = {'name': feature['name'],
                                          'type': feature_type,
                                          'input-format': input_format,
                                          'transformer-type': transformer_type,
                                          'transformer': transformer,
                                          'required': required,
                                          'scaler': scaler,
                                          'default': default}

    def __str__(self):
        """
        Returns a string with information about this cofniguration.

        """
        return """Schema name: %s
               # of named feature types: %d
               Target variable: %s
               """ % (self.schema_name, len(self._named_feature_types),
                      self.target_variable)
