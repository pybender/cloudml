# Author: Nikolay Melnik <nmelnik@upwork.com>

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import parse_parameters


class NoScaler(StandardScaler):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, copy=None):
        return X

    def inverse_transform(self, X, copy=None):
        return X


DEFAULT_SCALER = "MinMaxScaler"

SCALERS = {
    'NoScaler': {
        'class': NoScaler,
        'defaults': {},
        'parameters': []},
    'MinMaxScaler': {
        'class': MinMaxScaler,
        'defaults': {
            'feature_range_min': 0,
            'feature_range_max': 1,
            'copy': True},
        'parameters': ['feature_range_min', 'feature_range_max', 'copy']},
    'StandardScaler': {
        'class': StandardScaler,
        'defaults': {
            'copy': True,
            'with_std': True,
            'with_mean': True},
        'parameters': ['copy', 'with_std', 'with_mean']
    }
}


class ScalerException(Exception):
    pass


def get_scaler(scaler_config, default_scaler):
    if scaler_config is None:
        scaler_type = default_scaler
        scaler_config = {}
    else:
        scaler_type = scaler_config.get('type', None)
    if scaler_type is None:
        scaler_type = default_scaler
        scaler_config = {}
    if scaler_type is None:
        return None

    if scaler_type not in SCALERS:
        raise ScalerException(
            "Scaler '{0}' isn\'t supported.".format(scaler_type))

    params = parse_parameters(scaler_config, SCALERS[scaler_type])

    # process range params
    for param_name, param in params.copy().iteritems():
        if param_name.endswith('_min'):
            params.pop(param_name)
            param_name = param_name.replace('_min', '')
            param_max = params.pop(param_name + '_max')
            params[param_name] = (param, param_max)
    return SCALERS[scaler_type]['class'](**params)
