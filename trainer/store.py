__author__ = 'ifouk'

import pickle

from trainer import Trainer


def store_trainer(trainer, fp):
    pickle.dump(trainer, fp)


def load_trainer(fp):
    return pickle.load(fp)
