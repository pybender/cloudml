__author__ = 'ifouk'

import cPickle as pickle

from trainer import Trainer


def store_trainer(trainer, fp):
    # Unset data members of the trainer class to reduce its size 
    if hasattr(trainer, '_raw_data'):
        raw_data = trainer._raw_data
        trainer._raw_data = None
    if hasattr(trainer, '_vect_data'):
        vect_data = trainer._vect_data
        trainer._vect_data = None
    # Store class instance
    pickle.dump(trainer, fp)
    # Restore data members
    if hasattr(trainer, '_raw_data'):
        trainer._raw_data = raw_data
    if hasattr(trainer, '_vect_data'):
        trainer._vect_data = vect_data

def load_trainer(fp):
    return pickle.load(fp)
