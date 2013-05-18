from base import FeatureTypeBase, FeatureTypeInstanceBase

_author__ = 'nmelnik'


class OrdinalFeatureTypeInstance(FeatureTypeInstanceBase):
    """
        Looks up the original value at a dictionary and returns the value from
        the dictionary. If value wasn't found returns null.
    """
    def transform(self, value):
        params = self.active_params()
        if 'mappings' not in params:
            return None
        return params['mappings'].get(value, None)

class OrdinalFeatureType(FeatureTypeBase):
    instance = OrdinalFeatureTypeInstance
    required_params = ['mappings']
