#/usr/bin/env python
import sys
import re

if len(sys.argv) != 2:
    print 'Usage: ' + sys.argv[0] + ' <sql file>'
    sys.exit(1)

sql = ' '.join(filter(lambda x: not x.startswith('--'), 
               open(sys.argv[1], 'r').readlines()))

print re.sub('\\s+', ' ', sql).replace('"', '\\"')


