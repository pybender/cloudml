""" Exceptions processing """

# Author: Anna Lysak <annalysak@cloud.upwork.com>

import sys
import linecache
import traceback


def traceback_info():
    """
    Extracts backtrace and populates it with local variables for each line
    """
    max_recursion_deep = 40

    def format_line(line, locals=None):
        """
        Line to dictionary
        """
        def convert(data, ri):
            try:
                if ri > max_recursion_deep:
                    return str(data)
                import collections
                if isinstance(data, basestring):
                    return str(data)
                elif isinstance(data, (set, list, tuple)):
                    return [convert(item, ri + 1) for item in data]
                elif isinstance(data, dict):
                    d = dict()
                    for k, v in data.iteritems():
                        d[str(k)] = convert(v, ri + 1)
                    return d
                else:
                    return str(data)
            except Exception:
                return "Can't parse data"

        res = {'line': line}
        if locals:
            res['locals'] = {}
            for n, val in locals.iteritems():
                res['locals'][str(n)] = convert(val, 0)
        return res

    def process_tb_frame(tb):
        """
        Process traceback frame
        """
        f = tb.tb_frame
        co = f.f_code
        filename = co.co_filename
        modname = f.f_globals.get('__name__', filename)

        s = 'Module {2}. File {0}, in {3}, line {1}'\
            .format(co.co_filename, tb.tb_lineno, modname, co.co_name)
        line = linecache.getline(filename, tb.tb_lineno)
        if line:
            s = '{0}: {1}'.format(s, line.strip())
        res = [format_line(s, locals=f.f_locals)]
        return res

    try:
        t, v, tb = sys.exc_info()
        __stop_recursion__ = 1
        __stop_index__ = 0
        result = []
        if tb is None:
            return result
        result.append({'exception': '{}'.format(''.join(
            traceback.format_exception_only(t, v)))})
        result.append(format_line('Traceback (most recent call last):'))
        while tb is not None:
            if tb.tb_frame.f_locals.get('__stop_recursion__') \
                    or __stop_index__ > max_recursion_deep:
                break
            result.extend(process_tb_frame(tb))
            tb = tb.tb_next
            __stop_index__ += 1
        return result
    finally:
        del tb


class ChainedException(Exception):
    """
    Supports exception chaining and stores traceback on each step
    """
    def __init__(self, message, chain=None):
        super(ChainedException, self).__init__(message)
        self._traceback = traceback_info()
        self.chain = chain

    @property
    def traceback(self):
        tb = []
        obj = self
        # index to prevent deep cycle
        i = 0
        while obj is not None:
            if hasattr(obj, '_traceback') and obj._traceback:
                tb.append(obj._traceback)
            if hasattr(obj, 'chain') and obj.chain is not None:
                obj = obj.chain
            else:
                obj = None
        return tb

    def __str__(self):
        return self.message


def print_exception(e, with_colors=True, ret_value=False):
    """
     Prints exception in command line
    :return:
    """

    class bcolors:
        HEADER = 'HEADER'
        VARS = 'VARS'
        VAR_NAME = 'VAR_NAME'
        EXC_LINE = 'EXC_LINE'
        ENDC = 'ENDC'
        TB_LINE = 'TB_LINE'

        def cmd(self):
            self.HEADER = '\033[95m'
            self.VARS = '\033[94m'
            self.VAR_NAME = '\033[92m'
            self.EXC_LINE = '\033[93m'
            self.ENDC = '\033[0m'
            self.TB_LINE = '\033[1m'

        def disable(self):
            self.HEADER = ''
            self.VARS = ''
            self.VAR_NAME = ''
            self.EXC_LINE = ''
            self.ENDC = ''
            self.TB_LINE = ''

    c = bcolors()
    if not with_colors:
        c.disable()
    elif sys.stdout.isatty():
        c.cmd()

    ex = ChainedException("Got exception", e)
    tb_str = ''
    ind = ''
    nl = '\r\n'

    if not ex.traceback:
        return

    for tb in ex.traceback:
        if tb_str and tb:
            tb_str += "".join([nl, nl, ind[:-1], c.HEADER,
                               "CAUSED BY:", c.ENDC])
        for line in tb:
            exception = line.get('exception', None)
            lne = line.get('line', None)
            local_vars = line.get('locals', None)
            if exception:
                tb_str += "".join([nl, ind, c.EXC_LINE, exception, c.ENDC])
            elif lne:
                tb_str += "".join([nl, ind, c.TB_LINE, lne, c.ENDC])
            if local_vars:
                tb_str += "".join([nl, c.VARS, ind, "Local variables:",
                                   c.ENDC])
                for (k, v) in local_vars.iteritems():
                    tb_str += "".join([nl, ind, "\t", c.VAR_NAME, str(k),
                                       ": ", c.ENDC, str(v)])
        ind += "\t"

    if ret_value:
        return tb_str
    else:
        print tb_str
