"""
This module holds classes and utils to parse model configuration
from features.json file.
"""

# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>

import json
import importlib
from collections import OrderedDict

from feature_types import FEATURE_TYPE_FACTORIES
from feature_types import InvalidFeatureTypeException
from utils import parse_parameters
from transformers import get_transformer
from scalers import get_scaler
from classifier_settings import CLASSIFIERS
from exceptions import SchemaException


class FeatureModel(object):
    """
    Reads training data configuration from a file containing a JSON o

    config: string
        path to features configuration json file, when 'is_file' is True.
        Otherwise it's features configuration string in json format.
    is_file: boolean
        whether needed to load file or plain configuration string passed.
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

        if not isinstance(data, dict):
            raise SchemaException(message="Parsed JSON data is of type %s. "
                                          "Dictionary is expected." %
                                          type(data))

        if 'schema-name' not in data:
            raise SchemaException(message="schema-name is missing")

        self.schema_name = data['schema-name']
        self.classifier = {}
        self.target_variable = None
        self._named_feature_types = {}
        self.features = OrderedDict()
        self.required_feature_names = []
        self.group_by = []

        self._process_classifier(data)

        # Add feature types defined in 'feature-types section
        if 'feature-types' in data:
            for feature_type in data['feature-types']:
                self._process_named_feature_type(feature_type)

        if 'features' not in data:
            raise SchemaException('Features are missing in config')

        self.feature_names = []
        for feature in data['features']:
            self._process_feature(feature)

        if self.target_variable is None:
            raise SchemaException('No target variable defined')

        group_by = data.get('group-by', None)
        if group_by:
            self._process_group_by(group_by)

    def _process_group_by(self, group_by):
        for i in group_by:
            if i not in self.features:
                raise SchemaException(
                    'Can not group by %s. Feature doesn\'t exist' % i)
            self.group_by.append(i)

    def _process_classifier(self, config):
        """
        Reads config for classifier and stores it to classifier attribute.

        config: dict
            a dictionary containing the configuration of the feature type
        """
        classifier_config = config.get('classifier', None)
        if classifier_config is None:
            raise SchemaException('No classifier configuration defined')

        self.classifier_type = classifier_config.get('type')
        if self.classifier_type not in CLASSIFIERS.keys():
            raise SchemaException('Invalid classifier type')

        # Filter only valid parameters
        settings = CLASSIFIERS[self.classifier_type]
        params = parse_parameters(classifier_config, settings)
        self.classifier.update(params)

        # Trying to load classifier class
        module, name = settings.get('cls').rsplit(".", 1)
        module = importlib.import_module(module)
        self.classifier_cls = getattr(module, name)

    def _process_named_feature_type(self, config):
        """
        Used for processing named feature types. Named feature types are the
        ones defined in "feature-type" part of the configuration.

        config: dict
            a dictionary containing the configuration of the feature type
        """

        # Check if named feature type has a name
        if 'name' not in config:
            raise SchemaException(
                'Feature types should have a name: {0}'.format(config))

        # Check if named feature type is not already defined
        if config['name'] in self._named_feature_types:
            raise SchemaException("Feature type '%s' already defined"
                                  % (config['name']))

        feature_type_instance = self._process_feature_type(config)

        self._named_feature_types[config['name']] = feature_type_instance

    def _process_feature_type(self, config):
        """
        Creates the actual feature type parser object using the given
        configuration.

        config: dict
            a dictionary containing the configuration of the feature type
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
            raise SchemaException(
                'Cannot create instance of feature type: {0}. '
                'Err: {1}'.format(config, e), e)

    def _process_feature(self, feature):
        '''
        Validates each feature.

        feature: dict
            a dictionary containing the configuration of the feature
        '''
        if 'name' not in feature:
            raise SchemaException(
                'Features should have a name: {0}'.format(feature))

        if 'type' not in feature:
            raise SchemaException(
                'Feature {0} should have a type: {1}'.format(
                    feature['name'], feature))

        # Check if feature has a type definition
        feature_type = None
        if 'type' in feature:
            # Check if it is a named type
            feature_type = self._named_feature_types.get(feature['type'], None)

            # If not a named type, try to get a new instance
            if feature_type is None:
                feature_type = self._process_feature_type(feature)

        required = feature.get('is-required', True)
        # Check if it is a target variable
        if feature.get('is-target-variable', False) is True:
            self.target_variable = feature['name']
            # target variable should be required
            required = True

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

        default = feature.get('default', None)
        name = feature['name']
        if name in self.feature_names:
            raise SchemaException(
                'Duplicated feature with name {0}'.format(name))

        self.feature_names.append(name)
        if required:
            self.required_feature_names.append(name)
        self.features[feature['name']] = {'name': name,
                                          'type': feature_type,
                                          'input-format': input_format,
                                          'transformer-type': transformer_type,
                                          'transformer': transformer,
                                          'required': required,
                                          'scaler': scaler,
                                          'default': default}


    def __str__(self):
        """
        Returns a string with information about this configuration.
        """
        return """Schema name: %s
               # of named feature types: %d
               Target variable: %s
               """ % (self.schema_name, len(self._named_feature_types),
                      self.target_variable)
