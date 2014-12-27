from utils import ParametrizedTemplate

# Context:
from processors import composite_string, composite_python, \
    composite_readability, process_key_value
from exceptions import ImportHandlerException


class ScriptManager(object):
    """
    Manages and executes python scripts.
    """
    def __init__(self):
        self.data = ''
        self.context = {}

    def add_python(self, script):
        try:
            eval(compile(script, "<str>", "exec"), self.context, self.context)
        except Exception,  exc:
            raise ImportHandlerException(
                "Exception occurs while adding python script: {0}. {1}".format(
                    script[:100], exc))

    def execute_function(self, script, value,
                         row_data=None, local_vars={}):
        def update_strings(val):
            if isinstance(val, basestring):
                return "'%s'" % val
            return val

        row_data = row_data or {}
        params = {'value': update_strings(value)}
        params.update(row_data)
        params.update(local_vars)
        text = ParametrizedTemplate(script).safe_substitute(params)
        try:
            return self._exec(text, local_vars)
        except Exception,  exc:
            raise ImportHandlerException(
                "Exception occurs while executing script: {0}. {1}".format(
                    text[:100], exc))

    def _exec(self, text, row_data=None):
        row_data = row_data or {}
        context = globals().copy()
        context.update(locals())

        class ob(object):
            pass

        for k, v in row_data.iteritems():
            splited = k.split('.')
            if len(splited) == 1:
                context[k] = v
            elif len(splited) == 2:
                if not splited[0] in context:
                    context[splited[0]] = ob()
                setattr(context[splited[0]], splited[1], v)
            elif len(splited) == 3:
                if not splited[0] in context:
                    context[splited[0]] = ob()
                if not hasattr(context[splited[0]], splited[1]):
                    setattr(context[splited[0]], splited[1], ob())
                t = getattr(context[splited[0]], splited[1])
                setattr(t, splited[2], v)
        context.update(self.context)
        return eval(text, context, context)
