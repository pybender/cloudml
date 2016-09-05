import sys
import logging
import coloredlogs
import re
import inspect
import json
from new import instancemethod

import numpy
from sklearn import *
from sklearn.base import ClassifierMixin, RegressorMixin


EXCLUDE = [
    'abc.NewBase',
    'sklearn.ensemble.forest.ForestClassifier',
    'sklearn.tree.tree.ExtraTreeClassifier',
    'sklearn.tree.tree.ExtraTreeRegressor',
    'sklearn.ensemble.forest.ForestRegressor',
    'sklearn.dummy.DummyClassifier']

PARAMS_TO_REMOVE = ['self']

TYPE_MAP = {
    'string': 'string',
    'str': 'string',
    'str or callable': 'string',
    'boolean': 'boolean',
    'bool': 'boolean',
    'float': 'float',
    'double': 'float',
    'integer': 'integer',
    'int': 'integer',
    'int seed': 'integer',
    'int, float, string or none': 'int_float_string_none',
    'integer or cross-validation generator': 'integer',
    'integer > 0': 'positive_integer',
    'float or array': 'float_or_list',
    'float | array': 'float_or_list',
    'boolean or string': 'boolean_or_string',
    'int or none': 'integer',
    'integer or none': 'integer',
    'int or str or array of shape = [n_outputs]': 'integer_or_string',
    'bool or int': 'boolean_or_integer',
    'bool or integer': 'boolean_or_integer',
    'boolean or integer': 'boolean_or_integer',
    'int or float': 'float_or_integer',
    'int or none': 'integer',
    'integer or none': 'integer',
    'float or none': 'float_or_none',
    'string or float': 'string_or_float',
    'array of floats': 'list',
    'dict': 'dict',
    'array-like': 'list',
    'array': 'list',
    'list of float': 'list',
    'list': 'list',
    'int or list': 'integer_or_list',
    'callabale': 'object',
    'object': 'object',
    'estimator object': 'object',
    'object or none': 'object_or_none',
    'auto_dict': 'auto_dict',
    "{dict, 'auto'}": 'auto_dict',
    'dict, {class_label: weight} or "auto"': 'auto_dict',
    # 'auto_dict': convert_auto_dict,
    # 'int_float_string_none': convert_int_float_string_none,
    # 'float_or_int': convert_float_or_int,
    # 'string_list_none': convert_string_list_none,
}

# Results
CLASSIFIERS_CONFIG = {}
NCLASSIFIER_MODELS = []
NREGRESSION_MODELS = []

classifiers_with_error = {}
not_defined_types = []

coloredlogs.install(level=logging.DEBUG)


def slugify(s):
    """
    Compatibility with old classifier names.
    """
    name_map = {
        'LogisticRegression': 'logistic regression',
        'SVR': 'support vector regression',
        'SGDClassifier': 'stochastic gradient descent classifier',
        'DecisionTreeClassifier': 'decision tree classifier',
        'DecisionTreeRegressor': 'decision tree regressor',
        'GradientBoostingClassifier': 'gradient boosting classifier',
        'ExtraTreesClassifier': 'extra trees classifier',
        'RandomForestClassifier': 'random forest classifier',
        'RandomForestRegressor': 'random forest regressor',
    }
    return name_map.get(s) or s


def full_name(cls):
    """
    Returns full path to the class.
    """
    return "{0}.{1}".format(cls.__module__,
                            cls.__name__).strip()


def params_to_dict(params, key='name'):
    """
    Converts parameters list in the config to dict: param name/param config
    """
    return dict([(item[key], item) for item in params])


def all_subclasses(cls):
    """
    Returns all subclasses of the specified class.
    """
    return cls.__subclasses__() + [g for s in cls.__subclasses__()
                                   for g in all_subclasses(s)]


### Methods related for docstring parsing

def parse_docstring(doc):
    """
    Parses class docstring. Returns class descriptions and parameters list.
    """
    descr = doc
    params = []
    if not doc:
        return descr, params

    splitted_doc = doc.split('Parameters\n    ----------\n')
    doc_len = len(splitted_doc)
    params_in_declaration = doc_len > 0
    if params_in_declaration:
        help_text = splitted_doc[0]  # first it's a class description.
        params_declaration = get_params_str(doc)
        params = parse_params(params_declaration)

    return prettify_help_text(help_text), params


NAME_TYPE_SEPARATOR = ': '
DEFAULT_TYPE = 'string'

def parse_params(params_decl):
    """
    Parses Parameters section of the docstring.
    """
    results = []
    lines = params_decl.split("\n")
    param_help = []
    p = {}
    for line in lines:
        is_new_param = NAME_TYPE_SEPARATOR in line
        if is_new_param:
            if p and param_help:
                p['help_text'] = '\n'.join(param_help)
            param_help = []
            p = parse_param(line)
            if p:
                results.append(p)
            else:
                logging.warning('Cannot parse: \n %s. Ignored', line)
        else:
            param_help.append(prettify_help_text(line))
    return results


def determine_type(type_decl):
    decl = type_decl.lower()
    possible_type = ''
    possible_type_str = ''
    for key, type_ in TYPE_MAP.iteritems():
        if key in decl:
            # looks like we found right type or partically right type
            if len(possible_type_str) < len(key):
                possible_type = type_
                possible_type_str = key

    return possible_type or DEFAULT_TYPE, type_decl.replace(possible_type_str, "").replace('optional', "")


def parse_param(decl):
    """
    Parses parameters string:
        name : type, [optional], default, choices
    """
    name, ptype_decl = decl.split(NAME_TYPE_SEPARATOR, 1)
    if name.endswith('_'):
        logging.warning('Param %s has not added', name)
        return

    type_, extra_decl = determine_type(ptype_decl)
    result = {'name': prettify_name(name), 'type': type_}
    if 'optional' in ptype_decl:
        result['required'] = False

    # locking for choices in {} brackets
    choices = []
    choices_in_brackets = re.search(r"\{['\"]+([A-Za-z0-9, _']+)['\"]+\}", ptype_decl)
    if choices_in_brackets:
        choices_str = choices_in_brackets.group(1)
        for ch in choices_str.split(','):
            choices.append(ch.strip('\' '))
    if choices:
        result['choices'] = choices
    else:
        # removing default
        extra_decl = re.sub('[\(]?default[= ]+[\'"]+[\w.]*[\'"]+[\)]?', '', decl)

        # looking for choices in extra
        or_choices = re.findall(r'[\'"]+([\w.-]*)[\'"]+', extra_decl)
        if or_choices:
            result['choices'] = or_choices
    return result



def get_params_str(doc):
    """
    Returning docstrings bettween "Parameters" section and next one section.
    """
    params_match = re.findall(r'Parameters[\n ]+[-]+\n([\w\W]*)[A-Za-z0-9 ]+[\n ]+[-]+\n', doc)
    if params_match:
        return params_match[0]


def prettify_name(str_):
    return str_.strip()


def prettify_help_text(str_):
    return str_.replace('\n', '').replace('  ', ' ').strip()

### Methods related for docstring parsing ends


def process(mixin, list_):
    classifiers = all_subclasses(mixin)
    i = 1
    for cl in classifiers:
        name = cl.__name__
        if isinstance(cl.__init__, instancemethod) and \
                not (name.startswith('_') or
                     name.startswith('Base') or full_name(cl) in EXCLUDE):
            clf_name = full_name(cl)
            # if clf_name not in ['sklearn.ensemble.forest.RandomForestRegressor']:
            #     continue

            slug = slugify(name)
            list_.append(slug)
            print "\n\n===========", i, "\t", clf_name, "========="

            # introspection:
            args, varargs, varkw, defaults = inspect.getargspec(cl.__init__)
            for p in PARAMS_TO_REMOVE:
                try:
                    args.remove(p)
                except:
                    pass

            defaults_key_map = dict(zip(args, defaults))
            descr, params = parse_docstring(cl.__doc__)
            clf_config = {
                'cls': clf_name,
                'help_text': descr,
                'parameters': [],
                'defaults': {}
            }

            # from docstrings:
            for p in params:
                default = defaults_key_map.get(p['name'])
                if not default is None:
                    if isinstance(default, numpy.ndarray):
                        default = default.tolist()
                    p['default'] = default
                    clf_config['defaults'][p['name']] = default
                if p['name'] in args:
                    clf_config['parameters'].append(p)
                else:
                    logging.warning('Line "%s" is ignored' % p['name'])
            docstr_param_names = [p['name'] for p in clf_config['parameters']]
            if set(args) != set(docstr_param_names):
                err = 'Some params are missing: actual %s, expected: %s' % \
                    (docstr_param_names, args)
                logging.error(err)
                classifiers_with_error[full_name(cl)] = err
            else:
                logging.info('%s added with params: %s',
                             full_name(cl), docstr_param_names)
            CLASSIFIERS_CONFIG[slug] = clf_config
            i += 1


def check_config(new_config):
    """
    Comparing new config to current cloudml's classifiers config.
    """
    from cloudml.trainer.classifier_settings import CLASSIFIERS
    print "\n-\tComparing new config to CLASSIFIERS dict..."

    CLF_DICT = dict([(cl['cls'], cl) for cl in CLASSIFIERS.values()])
    found_clfs = []

    for i, nconfig in enumerate(new_config.values()):
        name = nconfig['cls']
        if name in CLF_DICT:
            print i, name
            found_clfs.append(name)
            print "\n - Checking", name, "specification"
            config = CLF_DICT[name]
            if config.get('help_text'):
                if config.get('help_text') != nconfig.get('help_text'):
                    print "Classifier description are different."
            else:
                print "Classifier description added"

            defaults = config.get('defaults')
            ndefaults = nconfig.get('defaults')
            if defaults:
                if defaults != ndefaults:
                    print "defaults are different"
                    print "old:", defaults
                    print "new:", ndefaults
            else:
                print "defaults added"

            print '\n --- checking parameters'
            params = config['parameters']
            nparams = nconfig['parameters']
            if len(params) != len(nparams):
                print "Count of parameters are different."
                new = set([p['name'] for p in nparams])
                old = set([p['name'] for p in params])
                if old - new:
                    print old - new, 'is missing'
                if new - old:
                    print new - old, 'is added'
            nparams_dict = params_to_dict(nparams)
            for config in params:
                name = config['name']
                if name in nparams_dict:
                    # checking type
                    nconfig = nparams_dict[name]
                    type_ = config['type']
                    ntype = nconfig['type']
                    if type_ != ntype:
                        logging.error('%s has diff type: %s (old), %s (new)', name, type_, ntype)

                choices = config.get('choices')
                nchoices = nconfig.get('choices')
                if choices:
                    if nchoices:
                        if choices != nchoices:
                            logging.error("choices of %s are diff" % config['name'])
                            print '\t old:', choices
                            print '\t new:', nchoices
                    else:
                        logging.error('choices not parsed before')
                        nconfig['choices'] = choices
                else:
                    if nchoices:
                        print "Choices was added for param", config['name'], ':', nchoices


def write_to_file(filename='cloudml/trainer/auto_classifier_settings.py.bak'):
    fmt = """
CLASSIFIER_MODELS = %s
REGRESSION_MODELS = %s
CLASSIFIERS = %s
    """
    str_config = json.dumps(CLASSIFIERS_CONFIG, indent=4, sort_keys=True).replace('true', 'True').replace('false', 'False')
    result = fmt % (NCLASSIFIER_MODELS, NREGRESSION_MODELS, str_config)
    with open(filename,'w') as f:
        f.write(result)


def main():
    process(ClassifierMixin, NCLASSIFIER_MODELS)
    process(RegressorMixin, NREGRESSION_MODELS)

    print "************* RESULT *********"
    print "\n-\tChecking parameter names..."
    if len(classifiers_with_error.keys()):
        print "Following parameters are missing:"
        for clf, err in classifiers_with_error.iteritems():
            print "*", clf
            print "\t", err

    print "\n-\tChecking parameter types..."
    if len(not_defined_types):
        print "Following types was not parsed:"
        for i, tt in enumerate(set(not_defined_types)):
            print i, tt


    check_config(CLASSIFIERS_CONFIG)
    write_to_file()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.warn('keybord interrupt')
