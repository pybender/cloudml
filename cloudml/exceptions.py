import sys
import linecache
import traceback


def traceback_info():
    """
    Extracts backtrace and populates it with local variables for each line
    """
    def format_line(line, locals=None):
        """
        Line to dictionary
        """
        def convert(data):
            try:
                import collections
                if isinstance(data, basestring):
                    return str(data)
                elif isinstance(data, collections.Mapping):
                    return dict(map(convert, data.iteritems()))
                elif isinstance(data, collections.Iterable):
                    return type(data)(map(convert, data))
                else:
                    return str(data)
            except Exception as e:
                return str(data)

        res = {'line': line, 'locals': {}}
        if locals and len(locals) < 10 :
            for n, val in locals.iteritems():
                res['locals'][convert(n)] = convert(val)
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
            l = '{0}: {1}'.format(s, line.strip())
        res = [format_line(l, locals=f.f_locals)]

        return res

    try:
        t, v, tb = sys.exc_info()
        __stop_recursion__ = 1
        result = []
        if tb is None:
            return result
        result.append({'exception': '{}'.format(''.join(
            traceback.format_exception_only(t, v)))})
        result.append(format_line('Traceback (most recent call last):'))
        while tb is not None:
            if tb.tb_frame.f_locals.get('__stop_recursion__'):
                break
            result.extend(process_tb_frame(tb))
            tb = tb.tb_next
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
        self.traceback = self._get_traceback()

    def _get_traceback(self):
        """
        Returns full backtrace based on all exceptions chain
        """
        trace = {'trace': self._traceback}
        current = self
        # index to prevent deep cycle
        i = 0
        reasons = []
        while i<20 and current.chain is not None and \
                isinstance(current.chain, ChainedException):
            reasons.append(current.chain._traceback)
            current = current.chain
            i += 1

        trace['reasons'] = reasons
        return trace

    def __str__(self):
        return self.message


def print_exception(e):
    """
    Prints exception in command line
    :return:
    """

    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    obj = ChainedException("Got exception", e)
    i = 0
    ind = ""
    print
    while i < 20 and obj is not None:
        if hasattr(obj, '_traceback') and obj._traceback:
            for line in obj._traceback:
                exception = line.get('exception', None)
                lne = line.get('line', None)
                local_vars = line.get('locals', None)
                if exception:
                    print "".join([ind, bcolors.WARNING, exception,
                                   bcolors.ENDC])
                elif lne:
                    print "".join([ind, bcolors.BOLD, lne, bcolors.ENDC])
                if local_vars:
                    print "".join([bcolors.OKBLUE, ind, "Local variables:",
                                   bcolors.ENDC])
                    for (k, v) in local_vars.iteritems():
                        print "".join([ind, bcolors.OKGREEN, "    ", str(k),
                                       ": ", bcolors.ENDC, str(v)])
        if hasattr(obj, 'chain') and obj.chain is not None \
                and hasattr(obj.chain, 'traceback') and obj.chain._traceback:
            print
            print "".join([ind, bcolors.HEADER, "CAUSED BY:",
                           bcolors.ENDC])
            obj = obj.chain
            ind += "    "
            i += 1
        else:
            obj = None
