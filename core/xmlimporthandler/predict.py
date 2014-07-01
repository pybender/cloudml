from exceptions import ImportHandlerException


class PredictModel(object):
    """
    Defines parameters for using a model with the data
    from the <import> part of the handler.
    """
    def __init__(self, config):
        self.name = config.get('name')
        self.value = config.get('value')
        self.script = config.get('script')
        if not (self.value or self.script):
            raise ImportHandlerException('Either value or script'
                                         ' attribute need to be defined')

        self.positive_label = PositiveLabel(config.xpath('positive_label[1]'))

        self.weights = []
        for weight in config.xpath('weight'):
            self.weights.append(Weight(weight))

    def __repr__(self):
        return 'Model "{0!s}": "{1!s}"'.format(self.name,
                                               self.value or self.script)


class PositiveLabel(object):
    """
    Allows overriding which label to use as positive label.
    """
    def __init__(self, config):
        if isinstance(config, list):
            config = config[0]
        if not config:
            self.value = 'true'
        else:
            self.value = config.get('value') or 'true'
        self.script = config.get('script')

    def __repr__(self):
        return self.value


class Weight(object):
    def __init__(self, config):
        self.label = config.get('label')
        self.script = config.get('script')


class PredictResult(object):
    """
    Defines how to formulate the response.
    """
    def __init__(self, config):
        pass


class Predict(object):
    """
    Container for prediction configuration elements.
    """
    def __init__(self, config):
        self.models = []
        self.results = []

        for model in config.xpath('model'):
            self.models.append(PredictModel(model))

        for result in config.xpath('result'):
            self.results.append(PredictResult(result))
