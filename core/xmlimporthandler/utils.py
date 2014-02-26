from string import Template


class ParametrizedTemplate(Template):
    delimiter = '#'


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
