"""
Classes to process XML Import Handler import section.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

from collections import OrderedDict
import logging
from datetime import datetime
import re
from jsonpath import jsonpath

from exceptions import ProcessException, ImportHandlerException
from utils import ParametrizedTemplate, PROCESS_STRATEGIES, \
    load_json
from datasources import DATASOURCES_REQUIRE_QUERY


__all__ = ['Field', 'Entity', 'EntityProcessor']


class Field(object):
    """
    Represents entity field.

    config: lxml.etree._Element
        parsed by lxml.objectify field definition tag.
    entity: Entity
        parent entity tag.

    Config contains attributes:
        name: string
            name of the field
        type: string, optional (default="string")
            type of the field: string, float, boolean, json, integer
        column: string, optional
            if entity is using a DB or CSV datasource, it will use
            data from this column
        jsonpath: string, optional
            if entity is a JSON datasource, it will use this jsonpath to
            extract data
        delimiter: string, optional
            concatenates values using the defined separator. Used together
            with jsonpath only.
        join: string
            concatenates values using the defined separator. Used together
            with jsonpath only.
        multipart: string, optional
            boolean (true/false), if the results of jsonpath is
            complex/multipart value or simple value, Used only with jsonpath
        key_path: string, optional
            a JSON path expression for identifying the keys of a map.
            Used together with value_path.
        value_path: string, optional
            JSON path expression for identifying the values of a map.
            Used together with key_path.
        script: string
            call the Javascript defined in this element and assign the
            result to this field. May use any of the built-in functions
            or any one defined in a Script element. Variables can also
            be used in script elements.
        regex: string, optional
            applies the given regular expression and assigns the first
            match to the value
        split: string, optional
            splits the value to an array of values
        dateFormat: string, optional
            transforms value to a date using the given date/time format
        template: string, optional
            used to define a template for strings. May use variables.
        transform: string, optional
            transforms this field to a datasource. For example, it can
            be used to parse JSON or CSV data stored in a DB column.
            Its values can be either json or csv.
        headers: string, optional
            used only if transform="csv". Defines the header names
            for each item in the CSV field.
        required: string, optional, (default='false')
            whether this field is required to have a value or not.
    """

    def __init__(self, config, entity):
        self.name = config.get('name')  # unique
        self.type = config.get('type', 'string')
        self.entity = entity

        # if entity is using a DB or CSV datasource,
        # it will use data from this column
        self.column = config.get('column')

        # if entity is a JSON datasource,
        # it will use this jsonpath to extract data
        self.jsonpath = config.get('jsonpath')
        # concatenates values using the defined separator.
        # Used together with jsonpath only.
        self.delimiter = config.get('delimiter', config.get('join'))
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
        self.multipart = config.get('multipart') == 'true'
        self.key_path = config.get('key_path', None)
        self.value_path = config.get('value_path', None)

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
        if self.type not in PROCESS_STRATEGIES:
            types = ", ".join(PROCESS_STRATEGIES.keys())
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
                      datasource_type=None, params=None):
        """
        Processes value according to a field configuration.
        """
        convert_type = True
        row = row or {}
        row_data = row_data or {}

        if self.jsonpath:
            self.jsonpath = \
                ParametrizedTemplate(self.jsonpath).safe_substitute(params)
            value = jsonpath(value, self.jsonpath)
            if value is False:
                value = None
            if self.key_path and self.value_path and value:
                # Treat as a dictionary
                keys = jsonpath(value[0], self.key_path)
                strategy = PROCESS_STRATEGIES.get(self.type)
                values = []
                for value in jsonpath(value[0], self.value_path):
                    try:
                        value = strategy(value)
                    except (ValueError, TypeError) as e:
                        if self.entity.log_msg_counter < 100:
                            self.entity.log_msg_counter += 1
                            logging.warning(
                                "Can't convert value '{0}' to {1} while "
                                "processing field {2}".format(value[:100],
                                                              self.type,
                                                              self.name))
                        value = None
                    values.append(value)
                convert_type = False
                if keys is not False and values is not False:
                    value = dict(zip(keys, values))
                else:
                    value = None
            if not self.delimiter and isinstance(value, (list, tuple)) \
                    and len(value) == 1 and not self.multipart:
                value = value[0]
            if isinstance(value, (list, tuple)):
                value = filter(None, value)

        if self.regex:
            if value is None:
                return None
            match = re.search(self.regex, value)
            if match:
                value = match.group(0)
                #print value
            else:
                return None

        if self.script:
            data = {}
            data.update(row)
            data.update(row_data)
            value = script_manager.execute_function(
                self.script, value, data, row_data)
            convert_type = False

        if self.split and value:
            value = re.split(self.split, value)

        if value is not None and self.delimiter:
            value = self.delimiter.join(value)

        # TODO: could we use python formats for date?
        if self.dateFormat:  # TODO: would be returned datetime, Is it OK?
            if value is None:
                return None
            value = datetime.strptime(value, self.dateFormat)
            convert_type = False

        if self.template:
            params = {'value': value}  # TODO: which params also we could use?
            value = ParametrizedTemplate(self.template).safe_substitute(params)

        if convert_type and value is not None:
            strategy = PROCESS_STRATEGIES.get(self.type)
            try:
                value = strategy(value)
            except ValueError, exc:
                if self.entity.log_msg_counter < 100:
                    self.entity.log_msg_counter += 1
                    logging.warning("Can't convert value '{0}' to {1} while \
processing field {2}".format(value[:100],
                             self.type,
                             self.name))
                value = None

        if self.required and not value:
            raise ProcessException('Field {} is required'.format(self.name))

        return value


class Sqoop(object):
    """
    Tag sqoop instructs import handler torun a Sqoop import.
    It should be used only on entities that have a pig datasource.

    config: lxml.etree._Element
        parsed by lxml.objectify sqoop definition tag.

    A sqoop tag may contain the following attributes:
    target: string
        the target file to save imported data on HDFS.
    datasource: string
        DB datasource name to use for importing the data
    table: string
        the name of the table to import its data.
    where: string, optional
        an expression that might be passed to the table for
        filtering the rows to import
    direct: boolean, optional
        whether to use direct import
    mappers: integer, optional (default=1)
        an integer number with the mappers to use for importing
        data. If table is a view or doesn't have a key it should
        be 1.
    options: string, optional
        extra options for sqoop import command.
    """
    def __init__(self, config):
        self.datasource_name = config.get('datasource')
        self.target = config.get('target')
        self.where = config.get('where')
        self.table = config.get('table')
        self.direct = config.get('direct')
        self.mappers = config.get('mappers', 1)
        self.options = config.get('options', '')
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

    config: lxml.etree._Element
        parsed by lxml.objectify field definition tag.

    The possible attributes of the element are the following:
    name: strinh
        a unique name to identify the entity
    datasource: string
        the datasource name to use for importing data
    query:string, optional
        a string that provides instructions on how to query
        a datasource.
    autoload_fields: boolean, optional (default=False)
        when setted we could not define fields. They would
        be loaded from the pig results.
    """
    def __init__(self, config):
        self.fields = OrderedDict()
        # entities, that used as json or csv field as datasource.
        self.nested_entities_field_ds = OrderedDict()
        # nested entities with another datasource.
        self.nested_entities_global_ds = []
        self.sqoop_imports = []
        self.log_msg_counter = 0

        self.datasource_name = config.get('datasource')
        self.name = config.get('name')
        self.autoload_fields = config.get('autoload_fields')
        self.fields_loaded = False

        if hasattr(config, 'query'):  # query is child element
            self.query_target = config.query.get('target')
            self.query = config.query.text
            self.autoload_sqoop_dataset = \
                config.query.get('autoload_sqoop_dataset', None) == 'true'
            self.sqoop_dataset_name = config.query.get('sqoop_dataset_name')
        else:  # query is attribute
            self.query = config.get('query')
            self.query_target = None

        if isinstance(config, dict):
            return

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
            field = Field(field_config, self)
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
    """
    Helper class that makes the logic of processing entity fields.
    """
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

        if self.datasource is None:
            raise ImportHandlerException(
                "Datasource or transformed field {0} not found, "
                "but it used in the entity {1}".format(
                    entity.datasource_name, entity.name))

        # Process sqoop imports
        for sqoop_import in self.entity.sqoop_imports:

            sqoop_import.datasource = import_handler.plan.datasources.get(
                sqoop_import.datasource_name)
            if sqoop_import.query:

                sqoop_query = sqoop_import.build_query(params)
                logging.info('Run query %s' % sqoop_query)
                # We running db datasource query to create a table
                sqoop_import.datasource.run_queries(sqoop_query)
            if self.entity.autoload_sqoop_dataset:
                from utils import SCHEMA_INFO_FIELDS, PIG_TEMPLATE, \
                    construct_pig_fields
                sql = """select * from {0} limit 1;
select {1} from INFORMATION_SCHEMA.COLUMNS where table_name = '{0}'
order by ordinal_position;""".format(sqoop_import.table,
                                     ','.join(SCHEMA_INFO_FIELDS))

                try:
                    iterator = sqoop_import.datasource._get_iter(sql)
                    fields_data = [{key: opt[i] for i, key in enumerate(
                                    SCHEMA_INFO_FIELDS)}
                                   for opt in iterator]
                except Exception, exc:
                    raise ValueError("Can't execute the query: {0}."
                                     "Error: {1}".format(sql, exc))

                fields_str = construct_pig_fields(fields_data)
                load_dataset_script = PIG_TEMPLATE.format(
                    self.entity.sqoop_dataset_name,
                    sqoop_import.target,
                    fields_str,
                    self.datasource.bucket_name)
                query = "{0}\n{1}".format(load_dataset_script, query)

        if self.datasource.type == 'pig':
            self.datasource.run_sqoop_imports(self.entity.sqoop_imports)
            self.datasource.set_import_handler(import_handler)

        if self.datasource.type == 'input' and query != 'any':
            query = import_handler.params[query]

        if self.datasource.type in DATASOURCES_REQUIRE_QUERY and \
                (query is None or not query.strip(' \t\n\r')):
            raise ImportHandlerException(
                "Query not specified in the entity {0}, but {1}"
                " datasource {2} require it".format(
                    self.entity.name,
                    self.datasource.type,
                    self.datasource.name))
        self.iterator = self.datasource._get_iter(
            query, self.entity.query_target, import_handler.params)

    def process_next(self):
        """
        Returns entity's processed next row data.
        """
        row = self.iterator.next()
        #print "row", row
        row_data = {}
        row_data.update(self.params)

        if self.entity.autoload_fields and not self.entity.fields_loaded:
            # We need to autoload fields on the first row processing
            logging.info('Auto load fields')
            from utils import autoload_fields_by_row
            autoload_fields_by_row(self.entity, row)

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
        fields = [i.name for i in self.entity.fields.values()]
        for p in self.params:
            if p not in fields:
                row_data.pop(p)
        return row_data

    def process_field(self, field, row, row_data=None):
        row_data = row_data or {}
        if field.column:
            item_value = row.get(field.column, None)
        else:
            item_value = row
        result = {}
        kwargs = {
            'row': row,
            'row_data': row_data,
            'datasource_type': self.datasource.type,
            'script_manager': self.import_handler.plan.script_manager,
            'params': self.params
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
                    kwargs['row_data'].update(result)
            elif field.transform == 'csv':
                # TODO: Implement Fields.transform=csv
                raise ImportHandlerException(
                    'Fields with transform=csv are not implemented yet')
        else:
            result[field.name] = field.process_value(item_value, **kwargs)
        return result

    def _get_entity_for_datasource_field(self, field):
        """
        Returns nested entity, that use this csv/json field as datasource.
        """
        return self.entity.nested_entities_field_ds.get(field.name)
