from utils import ParametrizedTemplate

# Context:
from processors import composite_string, composite_python,\
    composite_readability, process_key_value


class ScriptManager(object):
    """
    Manages and executes python scripts.
    """
    def __init__(self):
        self.data = ''
        self.context = {}

    def add_python(self, script):
        exec(script, globals(), self.context)

    def execute_function(self, script, value, row_data=None):
        def update_strings(val):
            if isinstance(val, basestring):
                return "'%s'" % val
            return val

        row_data = row_data or {}
        params = {'value': update_strings(value)}
        params.update(row_data)
        text = ParametrizedTemplate(script).safe_substitute(params)
        return self._exec(text, row_data)

    def _exec(self, text, row_data=None):
        row_data = row_data or {}
        context = globals().copy()
        context.update(locals())
        return eval(text, context, self.context)
