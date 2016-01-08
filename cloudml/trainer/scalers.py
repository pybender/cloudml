# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

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
        'parameters': [
            {'name': 'feature_range_min', 'type': 'integer', 'default': 0},
            {'name': 'feature_range_max', 'type': 'integer', 'default': 1},
            {'name': 'copy', 'type': 'boolean', 'default': True}]},
    'StandardScaler': {
        'class': StandardScaler,
        'defaults': {
            'copy': True,
            'with_std': True,
            'with_mean': True},
        'parameters': [
            {'name': 'copy', 'type': 'boolean', 'default': True},
            {'name': 'with_std', 'type': 'boolean', 'default': True},
            {'name': 'with_mean', 'type': 'boolean', 'default': True}]
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
    return SCALERS[scaler_type]['class'](**params)
