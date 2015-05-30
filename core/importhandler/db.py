"""
This module is a generic place used to hold helper functions
connect and query with databases.
"""

# Authors: Ioannis Foukarakis <ifoukarakis@upwork.com>
#          Nikolay Melnik <nmelnik@upwork.com>

import psycopg2
import psycopg2.extras

from uuid import uuid4
from time import time


def run_queries(queries, conn_string):
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
        cursor = conn.cursor().execute(query)
    conn.commit()


def postgres_iter(queries, conn_string):
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
        cursor = conn.cursor().execute(query)
    cursor_name = 'cursor_cloudml_%s' % (uuid4(), )
    cursor = conn.cursor(cursor_name,
                         cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(queries[-1])
    for row in cursor:
        yield row
