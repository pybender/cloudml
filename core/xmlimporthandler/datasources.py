from exceptions import ImportHandlerException
from core.importhandler.db import postgres_iter


class BaseDataSource(object):
    """
    Base class for any type of the datasource.
    """
    DB_ITERS = {
        'postgres': postgres_iter
    }

    def __init__(self, config):
        self.config = config
        self.name = config.get('name')  # unique
        self.type = config.tag

    def _get_iter(self, query=None):
        raise Exception('Not implemented')


class DbDataSource(BaseDataSource):
    """
    Database connection
    """
    def _get_iter(self, query):
        db_iter = self.DB_ITERS.get(self.config[0].attrib['vendor'])

        if db_iter is None:
            raise ImportHandlerException(
                'Database type %s not supported' % self.config['db']['vendor'])

        if 'host' not in self.config[0].attrib:
            raise ImportHandlerException(
                'No database connection details defined')

        conn_params = self.config[0].attrib
        conn_params.pop('name')
        conn_params.pop('vendor')
        conn_string = ' '.join(['%s=%s' % (k, v)
                                for k, v in conn_params.iteritems()])
        return db_iter(query, conn_string)


class PigDataSource(BaseDataSource):
    def _get_iter(self, query):
        pass


class DataSource(object):
    DATASOURCE_DICT = {
        'db': DbDataSource,
        'pig': PigDataSource,
    }

    @classmethod
    def factory(cls, config):
        return cls.DATASOURCE_DICT[config.tag](config)
