# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>
import os
import json


BASEDIR = 'testdata'


def get_iterator(dirname, filename, fmt='json'):
    from cloudml.trainer.streamutils import streamingiterload
    with open(os.path.join(BASEDIR, dirname,
                           '{0}.{1}'.format(filename, fmt))) as fp:
        data = list(streamingiterload(
            fp.readlines(), source_format=fmt))
    return data


def db_row_iter_mock(*args, **kwargs):
    filename = kwargs.pop('filename', 'testdata/extractorxml/out.json')
    with open(filename, 'r') as fp:
        data = json.loads(fp.read())
    for r in data:
        yield r
