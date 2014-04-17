"""
Classes to process XML Import Handler import section.
"""
from collections import OrderedDict
import json
import logging
from datetime import datetime
import re
from jsonpath import jsonpath

from exceptions import ProcessException, ImportHandlerException
from utils import get_key, ParametrizedTemplate, process_primitive


class FieldException(Exception):
    pass


class Field(object):
    """
    Represents entity field.
    """
    PROCESS_STRATEGIES = {
        'string': process_primitive(str),
        'float': process_primitive(float),
        # TODO: how do we need convert '1'?
        'boolean': process_primitive(bool),
        'json': lambda x: x,
        'integer': process_primitive(int)
    }

    def __init__(self, config):
        #self.config = config
        self.name = config.get('name')  # unique
        self.type = config.get('type', 'string')

        # if entity is using a DB or CSV datasource,
        # it will use data from this column
        self.column = config.get('column')

        # if entity is a JSON datasource,
        # it will use this jsonpath to extract data
        self.jsonpath = config.get('jsonpath')
        # concatenates values using the defined separator.
        # Used together with jsonpath only.
        self.join = config.get('join')

        # applies the given regular expression and
        # assigns the first match to the value
        self.regex = config.get('regex')
        # splits the value to an array of values using
        # the provided regular expression
        self.split = config.get('split')
        # transforms value to a date using the given date/time format
        self.dateFormat = config.get('dateFormat')
        # used to define a template for strings. May use variables.
        self.template = config.get('template')
        # call the Javascript defined in this element and assign the result
        # to this field. May use any of the built-in functions or any one
        # elements.
        if hasattr(config, 'script'):  # script is a child element
            self.script = config.script.text.strip()
        else:  # script is attribute
            self.script = config.get('script')
        # transforms this field to a datasource.
        self.transform = config.get('transform')
        # used only if transform="csv". Defines the header names for each item
        # in the CSV field.
        self.headers = config.get('headers')  # TODO:
        # Whether this field is required to have a value or not.
        # If not defined, default is false
        self.required = config.get('required') == 'true'

        self.validate_attributes()

    @property
    def is_datasource_field(self):
        """
        Determines whether it's a datasource field for a nested entity.
        """
        return self.transform in ('json', 'csv')

    def validate_attributes(self):  # TODO:
        """
        Validates field configuration.
        """
        if not self.type in self.PROCESS_STRATEGIES:
            types = ", ".join(self.PROCESS_STRATEGIES.keys())
            raise ImportHandlerException(
                'Type of the field %s is invalid: %s. Choose one of %s' %
                (self.name, self.type, types))

        if self.type != 'string':
            def _check_for_string(attr_name):
                if getattr(self, attr_name):
                    raise ImportHandlerException('Field %s declaration \
is invalid: use %s only for string fields' % (self.name, attr_name))
            _check_for_string('dateFormat')

    def process_value(self, value, script_manager, row=None, row_data=None,
                      datasource_type=None):
        """
        Processes value according to a field configuration.
        """
        convert_type = True
        row = row or {}
        row_data = row_data or {}

        if self.jsonpath:
            value = jsonpath(value, self.jsonpath)
            if value is False:
                value = None
            # if not self.join and isinstance(value, (list, tuple)) \
            #         and len(value) == 1:
            #     value = value[0]

        if self.regex:
            match = re.search(self.regex, value)
            if match:
                value = match.group(0)
            else:
                return None

        if self.script:
            data = {}
            data.update(row)
            data.update(row_data)
            value = script_manager.execute_function(self.script, value, data)
            convert_type = False

        if self.split and value:
            value = re.split(self.split, value)

        if value is not None and self.join:
            value = self.join.join(value)

        # TODO: could we use python formats for date?
        if self.dateFormat:  # TODO: would be returned datetime, Is it OK?
            value = datetime.strptime(value, self.dateFormat)
            convert_type = False

        if self.template:
            params = {'value': value}  # TODO: which params also we could use?
            value = ParametrizedTemplate(self.template).safe_substitute(params)

        if convert_type:
            strategy = self.PROCESS_STRATEGIES.get(self.type)
            value = strategy(value)

        # TODO: should it be here or before transformation?
        if self.required and not value:
            raise FieldException('Field {} is required'.format(self.name))

        return value


class Sqoop(object):

    def __init__(self, config):
        self.datasource_name = config.get('datasource')
        self.target = config.get('target')
        self.where = config.get('where')
        self.table = config.get('table')
        self.direct = config.get('direct')
        self.mappers = config.get('mappers', 1)
        self.query = config.text

    def build_query(self, params):
        """
        Returns query defined in the entity with applied parameters.
        """
        if not self.query:
            return None
        query = ParametrizedTemplate(self.query).safe_substitute(params)
        return query


class Entity(object):
    """
    Represents import handler's import entity.
    """
    def __init__(self, config):
        self.fields = OrderedDict()
        # entities, that used as json or csv field as datasource.
        self.nested_entities_field_ds = OrderedDict()
        # nested entities with another datasource.
        self.nested_entities_global_ds = []
        self.sqoop_imports = []

        self.datasource_name = config.get('datasource')
        self.name = config.get('name')

        if hasattr(config, 'query'):  # query is child element
            self.query_target = config.query.get('target')
            self.query = config.query.text
        else:  # query is attribute
            self.query = config.get('query')
            self.query_target = None

        self.load_fields(config)
        self.load_nested_entities(config)
        self.load_sqoop_imports(config)

    def build_query(self, params):
        """
        Returns query defined in the entity with applied parameters.
        """
        if not self.query:
            return None
        query = ParametrizedTemplate(self.query).safe_substitute(params)
        return query

    def load_fields(self, config):
        """
        Loads entity fields dictionary.
        """
        for field_config in config.xpath("field"):
            field = Field(field_config)
            self.fields[field.name] = field

    def load_sqoop_imports(self, config):
        for sqoop_config in config.xpath("sqoop"):
            self.sqoop_imports.append(Sqoop(sqoop_config))

    def load_nested_entities(self, config):
        """
        Loads nested entities with global datasource
        and that use csv/json field as datasource.
        """
        for entity_config in config.xpath("entity"):
            entity = Entity(entity_config)
            if self.uses_field_datasource(entity):
                self.nested_entities_field_ds[entity.datasource_name] = entity
            else:
                self.nested_entities_global_ds.append(entity)

    def uses_field_datasource(self, nested_entity):
        """
        Determines whether the nested entity use csv/json field as datasource.
        """
        return nested_entity.datasource_name in self.fields


class EntityProcessor(object):
    def __init__(self, entity, import_handler, extra_params={}):
        self.import_handler = import_handler
        self.entity = entity

        params = {}
        params.update(import_handler.params)
        params.update(extra_params)
        self.params = params

        # Building the iterator for the entity
        query = entity.build_query(params)
        self.datasource = import_handler.plan.datasources.get(
            entity.datasource_name)

        # Process sqoop imports
        logging.info('Process sqoop imports')
        for sqoop_import in self.entity.sqoop_imports:

            sqoop_import.datasource = import_handler.plan.datasources.get(
                sqoop_import.datasource_name)
            if sqoop_import.query:
                sqoop_query = sqoop_import.build_query(params)
                logging.info('Run query %s' % sqoop_query)
                sqoop_iter = sqoop_import.datasource.run_queries(sqoop_query)

        if self.datasource.config.tag == 'pig':
            self.datasource.run_sqoop_imports(self.entity.sqoop_imports)

        self.iterator = self.datasource._get_iter(
            query, self.entity.query_target)

    def process_next(self):
        """
        Returns entity's processed next row data.
        """
        row = self.iterator.next()
        row_data = {}
        for field in self.entity.fields.values():
            row_data.update(self.process_field(field, row, row_data))

        # Nested entities using a global datasource
        for nested_entity in self.entity.nested_entities_global_ds:
            nested_processor = EntityProcessor(
                nested_entity,
                self.import_handler,
                extra_params=row_data)
            # NOTE: Nested entity datasource should return only one row. Right?
            row_data.update(nested_processor.process_next())
        return row_data

    def process_field(self, field, row, row_data=None):
        row_data = row_data or {}
        item_value = row.get(field.column, None)
        result = {}
        kwargs = {
            'row': row,
            'row_data': row_data,
            'datasource_type': self.datasource.type,
            'script_manager': self.import_handler.plan.script_manager
        }
        if field.is_datasource_field:
            nested_entity = self._get_entity_for_datasource_field(field)
            if nested_entity is None:
                # No one use this field datasource
                return result

            if field.transform == 'json':
                data = load_json(item_value)
                for sub_field in nested_entity.fields.values():
                    result[sub_field.name] = sub_field.process_value(
                        data, **kwargs)
            elif field.transform == 'csv':
                raise Exception('Not implemented yet')  # TODO:
        else:
            result[field.name] = field.process_value(item_value, **kwargs)
        return result

    def _get_entity_for_datasource_field(self, field):
        """
        Returns nested entity, that use this csv/json field as datasource.
        """
        return self.entity.nested_entities_field_ds.get(field.name)


def load_json(val):
    if isinstance(val, basestring):
        try:
            return json.loads(val)
        except:
            raise ProcessException('Couldn\'t parse JSON message')
    return val
