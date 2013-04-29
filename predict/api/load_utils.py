import logging
import json
from os import listdir, makedirs
from os.path import isfile, join, exists, splitext

from core.trainer.store import load_trainer
from core.trainer.trainer import Trainer


def load_models(app):

    logging.info('Loading models')
    path = app.config['MODELS_FOLDER']
    if not exists(path):
        makedirs(path)
    for file in [f for f in listdir(path) if isfile(join(path, f))]:
        with open(join(path, file), 'r') as fp:
            logging.info('Loading %s...' % file)
            try:
                trainer = load_trainer(fp)
                app.models[splitext(file)[0]] = trainer
            except Exception, e:
                logging.info('File %s seems to be in invalid format' % file)
                logging.debug('Error: %s' % (str(e)))


def load_import_handlers(app):

    logging.info('Loading import handlers')
    path = app.config['IMPORT_HANDLERS_FOLDER']
    if not exists(path):
        makedirs(path)
    for file in [f for f in listdir(path) if isfile(join(path, f))]:
        with open(join(path, file), 'r') as fp:
            logging.info('Loading %s...' % file)
            try:
                app.import_handlers[splitext(file)[0]] = fp.read()
            except Exception, e:
                raise e
                logging.info('File %s seems to be in invalid format' % file)
                logging.debug('Error: %s' % (str(e)))
