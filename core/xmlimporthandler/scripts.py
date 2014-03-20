
from utils import ParametrizedTemplate


class ScriptManager(object):
    """
    Manages and executes javascript using V8.
    """
    def __init__(self):
        self.data = ''
        self.context = {}

    def add_python(self, script):
        exec(script, globals(), self.context)

    def execute_function(self, script, value, params={}):
        def update_strings(val):
            if isinstance(val, basestring):
                return "'%s'" % val
            return val

        params['value'] = update_strings(value)
        text = ParametrizedTemplate(script).safe_substitute(params)
        return self._exec(text)

    def _exec(self, text):
        return eval(text,  globals(), self.context)