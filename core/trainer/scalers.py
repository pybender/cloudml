from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils import copy_expected


DEFAULT_SCALER = "MinMaxScaler"

SCALERS = {
    'MinMaxScaler': {
        'class': MinMaxScaler,
        'params': {
            'feature_range_min': 0,
            'feature_range_max': 1,
            'copy': True
                  },
        },
    'StandardScaler': {
        'class': StandardScaler,
        'params': {
            'copy': True,
            'with_std': True,
            'with_mean': True
                  },
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
        raise ScalerException('Scaler %s do not support' % scaler_type)

    params = SCALERS[scaler_type]['params'].copy()
    scaler_params = copy_expected(scaler_config,
                           SCALERS[scaler_type]['params'].keys())
    params.update(scaler_params)
    # process range params
    for param_name, param in params.copy().iteritems():
        if param_name.endswith('_min'):
            params.pop(param_name)
            param_name = param_name.replace('_min','')
            param_max = params.pop(param_name + '_max')
            params[param_name] = (param, param_max)
    return SCALERS[scaler_type]['class'](**params)
