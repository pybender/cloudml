"""
Classes for definition inline predict section of XML import handlers.
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

from exceptions import ImportHandlerException

__all__ = ['Predict']


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
            raise ImportHandlerException(
                'Either value or script attribute need to be defined'
                ' for predict model {0}'.format(self.name))

        self.weights = []
        for weight in config.xpath('weight'):
            self.weights.append(Weight(weight))

    def __repr__(self):
        return 'Model "{0!s}": "{1!s}"'.format(self.name,
                                               self.value or self.script)


class Weight(object):
    def __init__(self, config):
        self.label = config.get('label')
        self.script = config.get('script')
        try:
            self.value = float(config.get('value', 1))
        except Exception, exc:
            raise ImportHandlerException(
                'Invalid predict model weight: {0}.'
                'Should be a float value.'.format(config.get('value')))


class ResultLabel(object):
    def __init__(self, config):
        self.model = config.get('model')
        self.script = config.get('script')


class ResultProbability(object):
    def __init__(self, config):
        self.model = config.get('model')
        self.script = config.get('script')
        self.label = config.get('label')


class PredictResult(object):
    """
    Defines how to formulate the response.
    """
    def __init__(self, config):
        self.label = ResultLabel(config.label)
        self.probability = ResultProbability(config.probability)


class Predict(object):
    """
    Container for prediction configuration elements.

    config: lxml.etree._Element
        parsed by lxml.objectify predict definition tag.

    Predict tag needs to have the following sub-elements:
        <model> - defines parameters for using a model with the
            data from the <import> part of the handler
        <result> - defines how to formulate the response. Predict
            should have only one result tag.
    """
    def __init__(self, config):
        self.models = []
        for model in config.xpath('model'):
            self.models.append(PredictModel(model))

        self.result = PredictResult(config.result)
