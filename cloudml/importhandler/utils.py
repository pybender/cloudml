# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import json
import os
from string import Template
from datetime import datetime
from ..utils import isint, isfloat, process_bool


class ParametrizedTemplate(Template):
    delimiter = '#'
    idpattern = r'[a-z][_a-z0-9]*(\.[a-z][_a-z0-9]*)*'


def iterchildren(config):
    for child_config in config.iterchildren():
        if child_config.tag != 'comment':
            yield child_config


def get_key(config, key):
    """
    Returns attribute value or sub_tag value.

    >>> from lxml import objectify
    >>> tag1 = objectify.fromstring('<entity key="value1"/>')
    >>> get_key(tag1, "key")
    'value1'
    >>> tag2 = objectify.fromstring("<entity><key>value2</key></entity>")
    >>> get_key(tag2, "key")
    'value2'
    """
    val = config.get(key)
    if val is None and hasattr(config, key):
        val = getattr(config, key)
    return val


def convert_single_or_list(value, process_fn):
    try:
        if isinstance(value, (list, tuple)):
            return [process_fn(item) for item in value]
        else:
            return process_fn(value)
    except ValueError:
        raise


def process_primitive(strategy):
    def process(value, **kwargs):
        return convert_single_or_list(value, strategy) \
            if value is not None else None
    return process


def process_date(value, format):
    return datetime.strptime(value, format)


PROCESS_STRATEGIES = {
    'string': process_primitive(str),
    'float': process_primitive(float),
    # TODO: how do we need convert '1'?
    'boolean': process_primitive(process_bool),
    'json': lambda x: x,
    'integer': process_primitive(int),
    'date': process_date
}


DIR = os.path.dirname(__file__)
with open(os.path.join(DIR, 'templates', 'pig_template.txt')) as fp:
    PIG_TEMPLATE = fp.read()

SCHEMA_INFO_FIELDS = [
    'column_name', 'data_type', 'character_maximum_length',
    'is_nullable', 'column_default']

PIG_FIELDS_MAP = {
    'integer': 'int',
    'smallint': 'int',
    'bigint': 'long',
    'character varying': 'chararray',
    'text': 'chararray',
    'double': 'double',
    'float': 'float',
    'decimal': 'double',
    'numeric': 'double',
    'boolean': 'boolean',
    'ARRAY': 'chararray',
    'json': 'chararray'
}


def get_pig_type(field):
    """
    Gets corresponding pig field type by sql-db field definition.

    field: dict
        declaration of the sql-table field, that contains data_type

    >>> get_pig_type({'column_name': 'title', 'data_type': 'integer'})
    'int'
    >>> get_pig_type({'column_name': 'created', 'data_type': 'timestamp'})
    'chararray'
    >>> get_pig_type({'column_name': 'accuracy', 'data_type': 'double-x'})
    'double'
    >>> get_pig_type({'column_name': 'created', 'data_type': 'custom'})
    'chararray'
    """
    type_ = field['data_type']
    if type_ in PIG_FIELDS_MAP:
        return PIG_FIELDS_MAP[type_]
    if type_.startswith('timestamp'):
        return 'chararray'
    if type_.startswith('double'):
        return 'double'
    return "chararray"


def construct_pig_fields(fields_data):
    """
    Constructs pig fields declaration by definition of the sql-db fields.

    >>> fields = [{'column_name': 'title', 'data_type': 'integer'}, \
                  {'column_name': 'created', 'data_type': 'timestamp'}, \
                  {'column_name': 'accuracy', 'data_type': 'double'}, \
                  {'column_name': 'created', 'data_type': 'custom'}]
    >>> construct_pig_fields(fields)
    'title:int\\n, created:chararray\\n, accuracy:double\\n, created:chararray'
    """
    fields_str = ""
    is_first = True
    for field in fields_data:
        if not is_first:
            fields_str += "\n, "
        fields_str += "{0}:{1}".format(field['column_name'],
                                       get_pig_type(field))
        is_first = False
    return fields_str


def autoload_fields_by_row(entity, row, prefix=''):
    """
    Autoloads entity fields from imported data row.

    entity: core.importhandler.Entity
        entity, where we need add fields and subentities.
    row: dict
        data, loaded from datasource.
    prefix:  string
        prefix to be added to the name.
    """
    def getjson(x):
        try:
            res = json.loads(x)
        except:
            return None
        return res

    from entities import Entity, Field
    for key, val in row.iteritems():
        data_type = 'string'
        if key not in entity.fields:
            if isint(val):
                data_type = 'integer'
            elif isfloat(val):
                data_type = 'float'
            else:
                item_dict = getjson(val)
                if item_dict:
                    entity.fields[key] = Field({
                        'name': key,
                        'column': key,
                        'transform': 'json'}, entity)
                    if key not in entity.nested_entities_field_ds:
                        json_entity = Entity(dict(name=key, datasource=key))
                        autoload_fields_by_row(
                            json_entity, item_dict, prefix='{0}.'.format(key))
                        entity.nested_entities_field_ds[key] = json_entity
                    continue

            if prefix:
                field_config = {
                    'name': prefix + key,
                    'jsonpath': '$.{0}'.format('.'.join(key.split('-')))}
            else:
                field_config = {
                    'name': key,
                    'type': data_type,
                    'column': key}
            entity.fields[key] = Field(field_config, entity)

    entity.fields_loaded = True


def load_json(val):
    if isinstance(val, basestring):
        try:
            return json.loads(val)
        except:
            from exceptions import ProcessException
            raise ProcessException(
                'Couldn\'t parse JSON message: {}'.format(val))
    return val
