def iterchildren(config):
    for child_config in config.iterchildren():
        if child_config.tag != 'comment':
            yield child_config
