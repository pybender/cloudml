from string import Template
from distutils.util import strtobool


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
    For example:
        <entity some_key="...."/>
        or
        <entity>
            <some_key>.....<some_key>
        </entity>
    """
    val = config.get(key)
    if val is None and hasattr(config, key):
        val = getattr(config, key)
    return val


def convert_single_or_list(value, process_fn, raise_exc=False):
    try:
        if isinstance(value, (list, tuple)):
            return [process_fn(item) for item in value]
        else:
            return process_fn(value)
    except ValueError:
        raise
        if raise_exc:
            raise
        return None


def process_primitive(strategy, raise_exc=True):
    def process(value, **kwargs):
        return convert_single_or_list(value, strategy, raise_exc) \
            if value is not None else None
    return process

def process_bool(value):
    val =  bool(strtobool(str(value)))
    return val


PIG_TEMPLATE = """register 's3://odesk-match-staging/pig/lib/elephant-bird-core-4.4.jar';
register 's3://odesk-match-staging/pig/lib/elephant-bird-pig-4.4.jar';
register 's3://odesk-match-staging/pig/lib/elephant-bird-hadoop-compat-4.4.jar';
register 's3://odesk-match-staging/pig/lib/piggybank-0.12.0.jar';

{0} = LOAD '${1}*' USING org.apache.pig.piggybank.storage.CSVExcelStorage(',', 'YES_MULTILINE') AS (
{2}
);"""


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
    type_ = field['data_type']
    if type_ in PIG_FIELDS_MAP:
        return PIG_FIELDS_MAP[type_]
    if type_.startswith('timestamp'):
        return 'chararray'
    if type_.startswith('double'):
        return 'double'
    return "chararray"


def construct_pig_sample(fields_data):
    fields_str = ""
    is_first = True
    for field in fields_data:
        if not is_first:
            fields_str += "\n, "
        fields_str += "{0}:{1}".format(field['column_name'],
                                       get_pig_type(field))
        is_first = False
    return fields_str


def isfloat(x):
    try:
        a = float(x)
    except :
        return False
    else:
        return True


def isint(x):
    try:
        a = float(x)
        b = int(a)
    except :
        return False
    else:
        return a == b
