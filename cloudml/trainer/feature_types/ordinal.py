"""
This module contains map feature type processor
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

from base import FeatureTypeBase, FeatureTypeInstanceBase
from cloudml.utils import isfloat, isint


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
            val = params['mappings'].get(value, None)
            if isint(val):
                return int(val)
            elif isfloat(val):
                return float(val)
            return val
        except:
            return None


class OrdinalFeatureType(FeatureTypeBase):
    instance = OrdinalFeatureTypeInstance
    required_params = ['mappings']
