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
    """
    Represents entity field.
    """
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

    def process_value(self, value):
        if value is not None and self.type is not None:
            strategy = self.PROCESS_STRATEGIES.get(self.type, None)
            if strategy is None:
                raise ImportHandlerException(
                    'Unknown strategy %s' % self.type)

            value = strategy(value)
        return value


class Entity(object):
    """
    Represents import handler's import entity.
    """
    def __init__(self, config):
        self.fields = {}
        self.json_datasources = {}  # entities, that used as json datasource.
        self.entities = []  # Nested entities with another datasource.

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

    def build_query(self, params):
        target = self.query.get('target')
        query = SqlTemplate(self.query.text).safe_substitute(params)
        query = [query]
        if target:
            query.append("SELECT * FROM %s;" % target)
        return query


class EntityProcessor(object):
    def __init__(self, entity, import_handler, extra_params={}):
        self.import_handler = import_handler
        self.entity = entity

        params = {}
        params.update(import_handler.params)
        params.update(extra_params)
        self.params = params

        query = entity.build_query(params)
        self.datasource = import_handler.datasources.get(
            entity.datasource_name)
        self.iterator = self.datasource._get_iter(query)

    def process_next(self):
        row = self.iterator.next()
        row_data = {}
        for field in self.entity.fields.values():
            row_data.update(self.process_field(field, row))

        # Nested entities using a global datasource
        for nested_entity in self.entity.entities:
            nested_processor = EntityProcessor(
                nested_entity,
                self.import_handler,
                extra_params=row_data)
            # NOTE: Nested entity datasource should return only one row. Right?
            nested_row = nested_processor.process_next()
            row_data.update(nested_row)
        return row_data

    def process_field(self, field, row):
        item_value = row.get(field.source, None)
        result = {}
        if field.transform == 'json':  # Let's find inner entity with data
            # Parse JSON string
            data = load_json(item_value)
            json_datasource = self.entity.json_datasources[field.name]
            for sub_field in json_datasource.fields.values():
                value = jsonpath(data, sub_field.jsonpath)[0]
                result[sub_field.name] = sub_field.process_value(value)
        else:
            result[field.name] = field.process_value(item_value)
        return result


def load_json(val):
    if isinstance(val, basestring):
        try:
            return json.loads(val)
        except:
            raise ProcessException('Couldn\'t parse JSON message')
    return val
