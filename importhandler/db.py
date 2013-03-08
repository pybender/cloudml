"""
Created on Dec 13, 2012

@author: ifoukarakis

"""
import psycopg2
import psycopg2.extras

from uuid import uuid4

from time import time


def postgres_iter(query, conn_string):
    """
    Iterator for iterating on a Postgres query using a named cursor.

    Keyword arguments:
    query -- the SQL query to execute
    conn_string -- the connection string

    """
    conn = psycopg2.connect(conn_string)
    cursor_name = 'cursor_cloudml_%s' % (uuid4(), )
    cursor = conn.cursor(cursor_name,
                         cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute(query)
    for row in cursor:
        yield row
