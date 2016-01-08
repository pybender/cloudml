# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>

import unittest
import os

from cloudml.trainer.streamutils import streamingiterload

BASEDIR = 'testdata'


class ParserTest(unittest.TestCase):

    def testLoadMultipleJSONSingleFile(self):
        f = open(os.path.join(BASEDIR, 'stream.data.json'))
        count = 0
        json_objects = []
        for o in streamingiterload(f.readlines()):
            json_objects.append(o)

        f.close()
        self.assertEquals(4, len(json_objects),
                          'Should have loaded 4 items from file (loaded %s)'
                          % (count,))
        self.assertEquals(1, json_objects[0]['id'],
                          'Invalid id for first JSON object')
        self.assertEquals('hire', json_objects[0]['class'],
                          'Invalid class for first JSON object')

        self.assertEquals(2, json_objects[1]['id'],
                          'Invalid id for second JSON object')
        self.assertEquals('hire', json_objects[1]['class'],
                          'Invalid class for second JSON object')

        self.assertEquals(3, json_objects[2]['id'],
                          'Invalid id for third JSON object')
        self.assertEquals('nohire', json_objects[2]['class'],
                          'Invalid class for third JSON object')

        self.assertEquals(4, json_objects[3]['id'],
                          'Invalid id for fourth JSON object')
        self.assertEquals('hire', json_objects[3]['class'],
                          'Invalid class for fourth JSON object')

    def testLoadMultipleCSVSingleFile(self):
        f = open(os.path.join(BASEDIR, 'stream.data.csv'))
        count = 0
        csv_objects = []
        for o in streamingiterload(f.readlines(), source_format='csv'):
            csv_objects.append(o)

        f.close()
        self.assertEquals(4, len(csv_objects),
                          'Should have loaded 4 items from file (loaded %s)'
                          % (count,))
        self.assertEquals('1', csv_objects[0]['id'],
                          'Invalid id for first CSV object')
        self.assertEquals('hire', csv_objects[0]['class'],
                          'Invalid class for first CSV object')

        self.assertEquals('2', csv_objects[1]['id'],
                          'Invalid id for second CSV object')
        self.assertEquals('hire', csv_objects[1]['class'],
                          'Invalid class for second CSV object')

        self.assertEquals('3', csv_objects[2]['id'],
                          'Invalid id for third CSV object')
        self.assertEquals('nohire', csv_objects[2]['class'],
                          'Invalid class for third CSV object')

        self.assertEquals('4', csv_objects[3]['id'],
                          'Invalid id for fourth CSV object')
        self.assertEquals('hire', csv_objects[3]['class'],
                          'Invalid class for fourth CSV object')

if __name__ == '__main__':
    unittest.main()
