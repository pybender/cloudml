"""
Feature types allow defining the data type of each feature and all
possible transformations to be made on each item.

"""

__author__ = 'ifoukarakis'


import calendar
import re
from functools import wraps
from datetime import datetime


class FeatureType(object):
    """
    Class for defining feature type factory objects. Provides basic
    functionality for validating feature type configuration.

    """
    def __init__(self, strategy, required_params=None, default_params=None):
        """
        Invoked whenever creating a feature type.

        Keyword arguments:
        strategy -- the method to invoke when transforming some data
        required_params -- a list containing the required parameters for this
                           feature type.

        """
        self._strategy = strategy
        if required_params is None:
            self._required = []
        else:
            self._required = required_params
        self._default_params = default_params

    def get_instance(self, params):
        set_params = set()
        if params is not None:
            set_params = set(params)
        if set(self._required).issubset(set_params):
            return FeatureTypeInstance(self._strategy, params,
                                       self._default_params)
        raise InvalidFeatureTypeException('Not all required parameters set')


class FeatureTypeInstance(object):
    """
    Decorator object for feature type instances.

    """
    def __init__(self, strategy, params=None, default_params=None):
        self._strategy = strategy
        self._params = params
        self._default_params = default_params
        
    def transform(self, value):
        if self._default_params is None and self._params is None:
            return self._strategy(value)
        final_params = {}
        if self._default_params is not None:
            final_params.update(self._default_params)
        if self._params is not None:
            final_params.update(self._params)
        return self._strategy(value, final_params)
    
    # The following is defined to allow pickling (see  
    # http://stackoverflow.com/questions/14550577/pickling-wrapped-partial-
    #  functions)



class CompositeFeatureType(FeatureType):
    """
    Factory class for creating composite feature type instances. A composite
    feature type applies transformations based on individual 'primitive'
    feature types.

    """
    def __init__(self):
        """
        Invoked whenever creating a feature type.

        """
        super(CompositeFeatureType, self).__init__(None, None)
        
    def get_instance(self, params):
        """
        Iterate over the feature type definitions in "chain" array, and create
        the appropriate composite feature type instance.

        Keyword arguments
        params -- a dictionary containing the configuration for individual
                  feature type instances.

        """
        # Check that we have 'chain' as param
        if 'chain' not in params:
            raise InvalidFeatureTypeException('Composite feature types must '
                                              'define property "chain"')
        # For each item in chain, check that it is a valid feature type.
        ft_instances = []
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


class CompositeFeatureTypeInstance(FeatureTypeInstance):
    """
    Decorator object for composite feature type instances. Invokes
    sequentially the strategies created.

    """
    def __init__(self, strategies):
        super(CompositeFeatureTypeInstance, self).__init__(strategies, None)
        
    def transform(self, value):
        """
        Iterate over all feature type instances and transform value
        sequentially.

        Keyword arguments
        value -- the value to transform

        """
        current_value = value
        for ft_instance in self._strategy:
            current_value = ft_instance.transform(current_value)
        return current_value


class InvalidFeatureTypeException(Exception):
    """
    Exception to be raised if there is an error parsing or using the
    configuration.

    """

    def __init__(self, message, Errors=None):
        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)
        # Now for your custom code...
        self.Errors = Errors


def regex_parse(value, params):
    """
    Parses using a regular expression, and returns the first match. If
    no matches are found, returns None.

    Keyword arguments:
    value -- the value to convert
    params -- params containing the pattern

    """
    p = re.compile(params['pattern'])
    result = p.findall(value)
    if len(result) > 0:
        return result[0]
    return None


def date_strategy(value, params):
    """
    Convert date to UNIX timestamp.

    Keyword arguments:
    value -- the value to convert
    params -- params containing the pattern

    """
    default = FEATURE_TYPE_DEFAULTS['date']
    if value is None:
        return default
    try:
        return calendar.timegm(
            datetime.strptime(value, params['pattern']).timetuple())
    except ValueError:
        pass
    except TypeError:
        pass
    return default



def primitive_type_strategy(constructor):
    """
    Generates a strategy function for a primitive python type.

    Keyword arguments:
    constructor: the constructor of the primitive type, e.g., int, float, etc.
    """
    @wraps(constructor)
    def strategy(value, params):
        """
        Convert the input value to a primitive type

        Keyword arguments:
        value -- the value to convert
        params -- params containing the pattern
        """
        default = constructor()
        if params is not None:
            default = params.get('default', default)
        if value is None:
            return default
        try:
            return constructor(value)
        except ValueError:
            pass
        return default
    return strategy


def ordinal_strategy(value, params={}):
    """
    Looks up the original value at a dictionary and returns the value from
    the dictionary. If value wasn't found returns null.

    Keyword arguments:
    value -- the value to convert
    params -- params containing the pattern

    """
    if 'mappings' not in params:
        return None
    return params['mappings'].get(value, None)


def categorical_strategy(value, params={}):
    """
    Returns the set of categorical values contained in the value param.
    
    Keyword arguments:
    value -- the categorical value(s)
    params -- params possibly containing the split pattern
    """
    if 'split_pattern' not in params:
        return value
    return re.split(params['split_pattern'], value)


def identity_strategy(value, params={}):
    return value

# Predefined feature types factories
FEATURE_TYPE_FACTORIES = {
    'boolean': FeatureType(primitive_type_strategy(bool), None, {}),
    'int': FeatureType(primitive_type_strategy(int), None, {}),
    'float': FeatureType(primitive_type_strategy(float), None, {}),
    'numeric': FeatureType(primitive_type_strategy(float), None, {}),
    'date': FeatureType(date_strategy, ['pattern']),
    'map': FeatureType(ordinal_strategy, ['mappings']),
    'categorical': FeatureType(categorical_strategy, None, {}),
    'text': FeatureType(identity_strategy, None, {}),
    'regex': FeatureType(regex_parse, ['pattern']),
    'composite': CompositeFeatureType()
}

FEATURE_TYPE_DEFAULTS = {
    # Default is Jan 1st, 2000
    'date': 946684800
}

if __name__ == '__main__':
    params = {
            'chain': [
                {'type': 'regex',
                 'params': {'pattern': '(\d*\.\d+)'}},
                {'type': 'float'}
            ],
            'should': 'ignore'
        }
    factory = FEATURE_TYPE_FACTORIES['composite']
    ft_instance = factory.get_instance(params)
    result = ft_instance.transform('This is a test. Price is 4.99$')
