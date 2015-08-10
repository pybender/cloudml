"""
This module contains map feature type processor
"""

# Author: Nikolay Melnik <nmelnik@upwork.com>

from base import FeatureTypeBase, FeatureTypeInstanceBase


class OrdinalFeatureTypeInstance(FeatureTypeInstanceBase):
    def transform(self, value):
        """
        Looks up the original value at a dictionary and returns the value from
        the dictionary. If value wasn't found returns null.

        value: string, int, float
            the value to convert

        Note:
            Feature should containing the 'mappings' in parameters.
        """
        params = self.active_params()
        try:
            return float(params['mappings'].get(value, None))
        except:
            raise ValueError('not numerical value: {0}'.format(value))


class OrdinalFeatureType(FeatureTypeBase):
    instance = OrdinalFeatureTypeInstance
    required_params = ['mappings']
