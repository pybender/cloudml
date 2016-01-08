# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

from base import FeatureTypeBase, FeatureTypeInstanceBase, \
    InvalidFeatureTypeException


class CompositeFeatureType(FeatureTypeBase):
    """
    Factory class for creating composite feature type instances. A composite
    feature type applies transformations based on individual 'primitive'
    feature types.
    """
    required_params = ['chain']

    def __init__(self):
        """
        Invoked whenever creating a feature type.
        """
        super(CompositeFeatureType, self).__init__(None)

    def get_instance(self, params, input_format=None):
        """
        Iterate over the feature type definitions in "chain" array, and create
        the appropriate composite feature type instance.

        params: dict
            a dictionary containing the configuration for individual feature
            type instances.
        """
        from . import FEATURE_TYPE_FACTORIES
        # Check that we have 'chain' as param
        if 'chain' not in params:
            raise InvalidFeatureTypeException(
                'Composite feature types should '
                'define property "chain"')
        # For each item in chain, check that it is a valid feature type.
        ft_instances = []
        if not isinstance(params['chain'], (list, tuple)):
            raise InvalidFeatureTypeException(
                'Composite feature types should '
                'define a list of individual feature types')
        for item in params['chain']:
            if 'type' not in item:
                raise InvalidFeatureTypeException('Type not set on individual '
                                                  'feature type')
            factory = FEATURE_TYPE_FACTORIES.get(item['type'], None)
            if factory is None:
                raise InvalidFeatureTypeException('Unknown type: %s'
                                                  % (item['type']))
            ft_instances.append(factory.get_instance(item.get('params', None)))
        return CompositeFeatureTypeInstance(ft_instances)


class CompositeFeatureTypeInstance(FeatureTypeInstanceBase):
    """
    Decorator object for composite feature type instances. Invokes
    sequentially the strategies created.
    """
    def __init__(self, strategies):
        self.strategies = strategies
        super(CompositeFeatureTypeInstance, self).__init__()

    def transform(self, value):
        """
        Iterate over all feature type instances and transform value
        sequentially.

        value: string
            the value to transform
        """
        current_value = value
        for ft_instance in self.strategies:
            current_value = ft_instance.transform(current_value)
        return current_value
