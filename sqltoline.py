# /usr/bin/env python
import sys
import re
import logging
import sqlparse

if len(sys.argv) != 2:
    print 'Usage: ' + sys.argv[0] + ' <sql file>'
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
sql = ''.join(open(sys.argv[1]).readlines())
for i, statement in enumerate(sqlparse.split(sql)):
    if not statement.strip():
        continue
    formatted = sqlparse.format(statement,
                                keyword_case='lower',
                                strip_comments=True,
                                reindent=True)
    print '\nStatement #%d' % (i + 1, )
    logging.debug(formatted)
    print re.sub('\\s+', ' ', formatted).replace('"', '\\"')
