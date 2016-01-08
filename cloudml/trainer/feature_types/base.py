# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>


class InvalidFeatureTypeException(Exception):
    """
    Exception to be raised if there is an error parsing or using the
    configuration.
    """
    def __init__(self, message, Errors=None):
        super(InvalidFeatureTypeException, self).__init__(message)
        self.Errors = Errors


class FeatureTypeInstanceBase(object):
    """
    Decorator object for feature type instances.
    """
    def __init__(self, params=None, default_params=None,
                 default_value=None, preprocessor=None,
                 default_scaler=None):
        self._params = params
        self.preprocessor = preprocessor
        self._default_params = default_params
        self.default_scaler = default_scaler
        self._default_value = default_value

    def active_params(self):
        active_params = {}
        if self._default_params is not None:
            active_params.update(self._default_params)
        if self._params is not None:
            active_params.update(self._params)
        return active_params

    def transform(self, value):
        return value


class FeatureTypeBase(object):
    """
    Class for defining feature type factory objects. Provides basic
    functionality for validating feature type configuration.
    """

    instance = FeatureTypeInstanceBase
    # a list containing the required parameters for this
    # feature type
    required_params = []
    optional_params = []
    default_params = []
    default_value = None
    default_scaler = 'MinMaxScaler'

    def __init__(self, preprocessor=None):
        """
        Invoked whenever creating a feature type.
        """
        self._preprocessor = preprocessor

    def get_instance(self, params, input_format=None):
        set_params = set()
        if params is not None:
            set_params = set(params)
        if set(self.required_params).issubset(set_params):
            return self.instance(params,
                                 self.default_params,
                                 self.default_value,
                                 self._preprocessor,
                                 self.default_scaler)
        raise InvalidFeatureTypeException('Not all required parameters set')
