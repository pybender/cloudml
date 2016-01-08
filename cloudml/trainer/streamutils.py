"""
Json stream loading utilities.
"""

# Author: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>

import json
import csv


class JsonStreamReader(object):
    sbuffer = ''

    def __init__(self):
        self._decoder = json.JSONDecoder()

    def process_read(self, data):
        '''Parse out json objects'''
        self.sbuffer += data
        self.parsing = True
        while self.parsing:
            # Remove erroneous data in front of callback object
            index = self.sbuffer.find('{')
            if index is not -1 and index is not 0:
                self.sbuffer = self.sbuffer[index:]
                # Try to get a json object from the data stream
            try:
                obj, index = self._decoder.raw_decode(self.sbuffer)
            except Exception:
                self.parsing = False
                # If we got an object fire the callback infra
            if self.parsing:
                self.sbuffer = self.sbuffer[index:]
                return obj


def csviterload(stream):
    reader = csv.DictReader(
        stream,
        quotechar="'",
        quoting=csv.QUOTE_ALL
    )
    for obj in reader:
        for key, value in obj.items():
            # Try load json field
            strkey = str(obj[key])
            if strkey.startswith('{') or strkey.startswith('['):
                try:
                    obj[key] = json.loads(value)
                except Exception:
                    pass
        yield obj


def jsoniterload(stream):
    # TODO: Consider memory mapping file
    reader = JsonStreamReader()
    for line in stream:
        try:
            obj = reader.process_read(line)
            if obj is not None:
                yield obj
        except Exception, ex:
            raise Exception('Failed to read next line from the input stream. '
                            'Error: %s' % ex)


SOURCE_FORMATS = {
    'json': jsoniterload,
    'csv': csviterload,
}


def streamingiterload(stream, source_format='json'):
    loader = SOURCE_FORMATS.get(source_format)
    if not loader:
        loader = SOURCE_FORMATS['json']
    return loader(stream)
