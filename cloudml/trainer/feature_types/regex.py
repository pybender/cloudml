# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import re

from base import FeatureTypeBase, FeatureTypeInstanceBase


class RegexFeatureTypeInstance(FeatureTypeInstanceBase):
    def transform(self, value):
        """
        Parses using a regular expression, and returns the first match. If
        no matches are found, returns None.

        value: string
            the value to convert

        Note:
            Feature should containing the 'pattern' in parameters.
        """
        params = self.active_params()

        p = re.compile(params['pattern'])
        result = p.findall(value)
        if len(result) > 0:
            return result[0]
        return None


class RegexFeatureType(FeatureTypeBase):
    """
    Uses the regular expression defined in parameter pattern to transform
    the value of the feature.
    Note that in case of multiple matches, only the first one is used
    """
    instance = RegexFeatureTypeInstance
    required_params = ['pattern']
