'''
Created on Nov 16, 2012

@author: ifoukarakis
'''
import json


class StreamReader(object):
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
            except Exception, e:
                self.parsing = False
                # If we got an object fire the callback infra
            if self.parsing:
                self.sbuffer = self.sbuffer[index:]
                return obj


def streamingiterload(stream):
    ### TODO: Consider memory mapping file
    reader = StreamReader()
    for line in stream:
        try:
            obj = reader.process_read(line)
            if obj is not None:
                yield obj
        except Exception, ex:
            raise Exception('Failed to read next line from the input stream. '
                            'Error: %s' % ex)
