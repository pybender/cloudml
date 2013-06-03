import re

from base import FeatureTypeBase, FeatureTypeInstanceBase

__author__ = 'nmelnik'


class RegexFeatureTypeInstance(FeatureTypeInstanceBase):
    """
        Parses using a regular expression, and returns the first match. If
        no matches are found, returns None.
    """
    def transform(self, value):
        params = self.active_params()

        p = re.compile(params['pattern'])
        result = p.findall(value)
        if len(result) > 0:
            return result[0]
        return None


class RegexFeatureType(FeatureTypeBase):
    instance = RegexFeatureTypeInstance
    required_params = ['pattern']
