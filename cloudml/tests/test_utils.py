# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>
import os
import json
from placebo.pill import Pill, LOG, FakeHttpResponse, deserialize
from botocore.response import StreamingBody


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


class StreamPill(Pill):
    """
    This class support Streaming Body in response
    (Pill class supports only json responses)
    """
    def load_response(self, service, operation):
        LOG.debug('load_response: %s.%s', service, operation)
        response_file = self.get_next_file_path(service, operation)
        LOG.debug('load_responses: %s', response_file)
        with open(response_file, 'r') as fp:
            response_data = json.load(fp, object_hook=deserialize)
            if 'Body' in response_data['data']:
                import cStringIO
                self.stream = cStringIO.StringIO()
                self.stream.write(response_data['data']['Body'])
                self.stream.seek(0)
                response_data['data']['ContentLength'] = \
                    len(response_data['data']['Body'])
                response_data['data']['Body'] = \
                    StreamingBody(self.stream,
                                  len(response_data['data']['Body']))

        return (FakeHttpResponse(response_data['status_code']),
                response_data['data'])
