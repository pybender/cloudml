from string import Template
from distutils.util import strtobool


class ParametrizedTemplate(Template):
    delimiter = '#'
    idpattern = r'[a-z][_a-z0-9]*(\.[a-z][_a-z0-9]*)*'


def iterchildren(config):
    for child_config in config.iterchildren():
        if child_config.tag != 'comment':
            yield child_config


def get_key(config, key):
    """
    Returns attribute value or sub_tag value.
    For example:
        <entity some_key="...."/>
        or
        <entity>
            <some_key>.....<some_key>
        </entity>
    """
    val = config.get(key)
    if val is None and hasattr(config, key):
        val = getattr(config, key)
    return val


def convert_single_or_list(value, process_fn, raise_exc=False):
    try:
        if isinstance(value, (list, tuple)):
            return [process_fn(item) for item in value]
        else:
            return process_fn(value)
    except ValueError:
        raise
        if raise_exc:
            raise
        return None


def process_primitive(strategy, raise_exc=True):
    def process(value, **kwargs):
        return convert_single_or_list(value, strategy, raise_exc) \
            if value is not None else None
    return process

def process_bool(value):
    val =  bool(strtobool(str(value)))
    return val
