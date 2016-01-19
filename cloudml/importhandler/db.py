"""
This module is a generic place used to hold helper functions
connect and query with databases.
"""

# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>

import psycopg2
import psycopg2.extras
import logging

from uuid import uuid4


def run_queries(queries, conn_string):  # pragma: no cover
    """
    Executes queries on PostgreSQL databse.

    queries: list of strings
        the SQL query to execute
    conn_string: string
        the connection string
    """
    from importhandler import ImportHandlerException
    if not queries:
        raise ImportHandlerException('Empty query list')

    conn = psycopg2.connect(conn_string)
    for query in queries:
        execute(conn.cursor(), query)
    conn.commit()


def postgres_iter(queries, conn_string):  # pragma: no cover
    """
    Iterator for iterating on a Postgres query using a named cursor.

    queries: list of strings
        the SQL query to execute
    conn_string: string
        the connection string

    """
    from importhandler import ImportHandlerException
    if not queries:
        raise ImportHandlerException('Empty query list')

    conn = psycopg2.connect(conn_string)
    for query in queries[:-1]:
        execute(conn.cursor(), query)
    cursor_name = 'cursor_cloudml_%s' % (uuid4(), )
    cursor = conn.cursor(cursor_name,
                         cursor_factory=psycopg2.extras.DictCursor)
    execute(cursor, queries[-1])
    for row in cursor:
        yield row


def check_table_name(tablename):
    """
    Check the given name for being syntactically valid,
    and usable without quoting
    """
    from string import letters
    NAMECHARS = frozenset(set(letters).union('._-'))
    if not isinstance(tablename, basestring):
        raise TypeError('%r is not a string' % (tablename, ))
    invalid = set(tablename).difference(NAMECHARS)
    if invalid:
        raise ValueError('Invalid chars: %s' % (tuple(invalid), ))
    for s in tablename.split('.'):
        if not s:
            raise ValueError('Empty segment in %r' % tablename)


def execute(cursor, query):
    logging.debug('Exec %s', query)
    if isinstance(query, dict):
        text = query.get('text')
        if not text:
            logging.warning('Trying to execute empty query')
            return

        parameters = query.get('parameters', {})
        cursor.execute(text, parameters)
    else:
        if not query:
            logging.warning('Trying to execute empty query')
            return

        cursor.execute(query)
