def process_bool(val=None):
    if val is None:
        return False
    if isinstance(val, basestring):
        try:
            from distutils.util import strtobool
            return bool(strtobool(val))
        except ValueError:
            return False
    else:
        return bool(val)
