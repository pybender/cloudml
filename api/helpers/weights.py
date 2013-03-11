import math


def calc_weights_css(weights, css_cls):
    """
    Determines tones of color dependly of weight value.
    """
    cmp_func = lambda a: abs(a['weight'])
    no_zero_weights = [w for w in weights if w['weight'] != 0]
    min_weight = min(no_zero_weights, key=cmp_func)['weight']
    weights = [{'name': item['name'],
                'weight': item['weight'],
                'transforment_weight': math.log(
                abs(item['weight'] / min_weight))
                if item['weight'] != 0 else 0}
               for item in weights]
    weights.sort(key=cmp_func)

    tones = ['lightest', 'lighter', 'light', 'dark', 'darker', 'darkest']
    if min_weight > 0:
        tones.reverse()

    tones_count = len(tones)
    cmp_func = lambda a: abs(a['transforment_weight'])
    wmax = abs(max(weights, key=cmp_func)['transforment_weight'])
    delta = round(wmax / tones_count)
    for i in xrange(tones_count):
        tone = tones[i]
        css_class = "%s %s" % (css_cls, tone)
        limit = i * delta
        for item in weights:
            if abs(item['transforment_weight']) >= limit:
                item['css_class'] = css_class
    return weights


def weights2tree(weights):
    """
    Converts weights list to tree dict.
    """
    tree = {}
    for item in weights:
        name = item['name']
        splitted = name.split(".")
        count = len(splitted)
        parent_node = tree
        for i, part in enumerate(splitted):
            if i == (count - 1):
                param = {'full_name': name,
                         'name': part,
                         'value': item['weight'],
                         'css_class': item['css_class']}
                parent_node[part] = param

            if not part in parent_node:
                parent_node[part] = {}

            parent_node = parent_node[part]
    return tree


def get_weighted_data(model, row):
    """
    Add weights and color tones to each parameter.
    """
    def get_weight_dict(weights):
        res = {}
        for item in weights:
            res[item['name']] = item
        return res

    weights = get_weight_dict(model.positive_weights)
    negative = get_weight_dict(model.negative_weights)
    weights.update(negative)

    result = {}
    for key, value in row.iteritems():
        result[key] = {'value': value}
        if key in weights:
            wdict = weights[key]
            result[key]['weight'] = wdict['weight']
            result[key]['css_class'] = wdict['css_class']
        else:
            if isinstance(value, basestring):
                value = value.strip()
                # try to find weight for {{ key.value }}
                concated_key = ("%s.%s" % (key, value)).lower()
                if concated_key in weights:
                    wdict = weights[concated_key]
                    result[key]['weight'] = wdict['weight']
                    result[key]['css_class'] = wdict['css_class']
                else:
                    # try to find each word from the value
                    splitted = value.split(' ')
                    for word in splitted:
                        word = word.lower()
                        if not 'weights' in result[key]:
                            result[key]['weights'] = {}
                        concated_key = "%s.%s" % (key, word)
                        if concated_key in weights:
                            wdict = weights[concated_key]
                            word_weight = {'weight': wdict['weight'],
                                           'css_class': wdict['css_class']}
                            result[key]['weights'][word] = word_weight
            elif isinstance(value, dict):
                for dkey, dvalue in value.iteritems():
                    concated_key = ("%s.%s=%s" % (key, dkey, dvalue))
                    if concated_key in weights:
                        wdict = weights[concated_key]
                        result[key]['weight'] = wdict['weight']
                        result[key]['css_class'] = wdict['css_class']
    return result
