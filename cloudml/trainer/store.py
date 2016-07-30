"""
This module gathers utilities for pickling the model.
"""

# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>

import cPickle as pickle

from trainer import Trainer
from exceptions import InvalidTrainerFile
from . import __version__


class TrainerStorage(object):

    def __init__(self, trainer):
        self._classifier = trainer._classifier
        self._feature_model = trainer._feature_model
        self._features = trainer.features
        self._segments = trainer._segments
        self.visualization = trainer.visualization
        print __version__
        self.version = __version__

    @classmethod
    def load(cls, fp, load_classifier=True, load_visualisation=True):
        return cls._load(fp, load_classifier, load_visualisation)

    @classmethod
    def loads(cls, s, load_classifier=True, load_visualisation=True):
        return cls._load(s, load_classifier, load_visualisation)

    @classmethod
    def _load(cls, storage, load_classifier=True, load_visualisation=True):
        try:
            if isinstance(storage, basestring):
                storage = pickle.loads(storage)
            else:
                storage = pickle.load(storage)
        except (pickle.UnpicklingError, AttributeError), exc:
            raise InvalidTrainerFile("Could not unpickle trainer - %s" % exc)
        trainer = Trainer(storage._feature_model)
        if hasattr(storage, "_features"):
            if load_classifier:
                trainer.set_classifier(storage._classifier)
            trainer.set_features(storage._features)
            if hasattr(storage, "_segments"):
                trainer._segments = storage._segments
            if hasattr(storage, "visualization") and load_visualisation:
                trainer.visualization = storage.visualization
        else:
            trainer._feature_model.group_by = []
            if load_classifier:
                trainer.set_classifier({"default": storage._classifier})
            trainer.set_features({"default": storage._feature_model.features})

        return trainer

    def dump(self, fp):
        pickle.dump(self, fp)

    def dumps(self):
        return pickle.dumps(self)


def store_trainer(trainer, fp):
    storage = TrainerStorage(trainer)
    storage.dump(fp)


def load_trainer(fp, load_classifier=True, load_visualisation=True):
    return TrainerStorage.load(fp, load_classifier, load_visualisation)
