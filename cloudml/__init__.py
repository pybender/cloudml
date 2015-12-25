__version__ = '0.1.1'

__all__ = ['importhandler']


def traceback_info():
    from zope.exceptions import format_exception
    import sys
    try:
        t, v, tb = sys.exc_info()
        if tb:
            report = format_exception(t, v, tb, with_filenames=True)
            lines = []
            for line in report:
                line = line.replace('- __traceback_info__:', '**INFO**:')
                lines.append(line.replace(
                    'Traceback (most recent call last):', ''))
            return '\n'.join(lines)
        return ''
    finally:
        del tb


def full_stack():
    from zope.exceptions import extract_stack
    import sys
    lines = []
    for line in extract_stack(sys.exc_info()[2].tb_frame):
        lines.append(line.replace('- __traceback_info__:', '**INFO**:'))
    return '\n'.join(lines)


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
        trace = self._traceback
        current = self
        i = 0
        while hasattr(current, 'chain') and current.chain is not None \
                and i < 20:
            trace += '\n Caused by: {}\n'.format(current.chain.message)
            if current.chain._traceback:
                trace += current.chain._traceback
            current = current.chain
            i += 1

        return trace
