import PyV8

from utils import ParametrizedTemplate


class ScriptManager(object):
    def __init__(self):
        self.context = PyV8.JSContext()
        self.context.enter()
        self.data = ''

    def add_js(self, js):
        self._exec(js)

    def execute_function(self, script, value, params={}):
        def update_strings(val):
            if isinstance(val, basestring):
                return "'%s'" % val
            return val

        params['value'] = update_strings(value)
        text = ParametrizedTemplate(script).safe_substitute(params)
        return self._exec(text)

    def _exec(self, text):
        return self.context.eval(text)
