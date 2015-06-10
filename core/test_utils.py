import os


BASEDIR = 'testdata'


def get_iterator(dirname, filename, fmt='json'):
    from core.trainer.streamutils import streamingiterload
    with open(os.path.join(BASEDIR, dirname,
                           '{0}.{1}'.format(filename, fmt))) as fp:
        data = list(streamingiterload(
            fp.readlines(), source_format=fmt))
    print data
    return data
