import json
from string import Template
from jsonpath import jsonpath

from exceptions import ProcessException


class SqlTemplate(Template):
    delimiter = '#'


def process_primitive(constructor):
    def process(value):
        return constructor(value) if value is not None else None
    return process


class Field(object):
    PROCESS_STRATEGIES = {
        'string': process_primitive(str),
        'float': process_primitive(float),
        'boolean': process_primitive(bool),
        'integer': process_primitive(int)
    }

    def __init__(self, config):
        self.config = config
        self.name = config.get('name')  # unique
        self.type = config.get('type', 'string')
        self.source = config.get('column', None)
        self.jsonpath = config.get('jsonpath', None)
        self.transform = config.get('transform', None)

    def process_field(self, row, entity):
        def convert_type(field, value):
            if value is not None and field.type is not None:
                strategy = self.PROCESS_STRATEGIES.get(field.type, None)
                if strategy is None:
                    raise ImportHandlerException(
                        'Unknown strategy %s' % field.type)

                value = strategy(value)
            return value

        item_value = row.get(self.source, None)
        result = {}
        if self.transform == 'json':  # Let's find inner entity with data
            # Parse JSON string
            data = load_json(item_value)
            for field in entity.json_datasources[self.name].fields.values():
                value = jsonpath(data, field.jsonpath)[0]
                result[field.name] = convert_type(field, value)
        else:
            result[self.name] = convert_type(self, item_value)
        return result


class Entity(object):
    def __init__(self, config):
        self.fields = {}
        self.json_datasources = {}  # entities, that used as json datasource.
        self.entities = []  # Nested entities with specific datasources.

        self.config = config
        self.datasource_name = config.attrib['datasource']
        self.name = config.attrib.get('name')
        self.query = config.attrib.get('query')
        if not self.query and hasattr(config, 'query'):
            self.query = config.query

        for field_config in config.xpath("field"):
            field = Field(field_config)
            self.fields[field.name] = field

        for entity_config in config.xpath("entity"):
            entity = Entity(entity_config)
            if entity.datasource_name in self.fields:
                self.json_datasources[entity.datasource_name] = entity
            else:
                self.entities.append(entity)

    def get_iter(self, datasource, params={}):
        query = prepare_query(self.query.text, params,
                              target=self.query.get('target'))
        return datasource._get_iter(query)

    def process_row(self, row):
        row_data = {}
        for field in self.fields.values():
            row_data.update(field.process_field(row, self))
        return row_data


def prepare_query(query, params, target=None):
    query = SqlTemplate(query).safe_substitute(params)
    query = [query]
    if target:
        query.append("SELECT * FROM %s;" % target)
    return query


def load_json(val):
    if isinstance(val, basestring):
        try:
            return json.loads(val)
        except:
            raise ProcessException('Couldn\'t parse JSON message')
    return val
