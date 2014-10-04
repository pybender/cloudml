#! /usr/bin/env python
# encoding: utf-8
"""
Class for training classifier and evaluating trained model. Uses logistic
regression.


It defines class Trainer.

@author:     ifoukarakis

@copyright:  2013 odesk. All rights reserved.
"""

__author__ = 'ifouk'

import os
import sys
import logging

from core.trainer.config import FeatureModel, SchemaException
from core.trainer import __version__
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from core.importhandler.importhandler import ImportHandlerException, \
    ExtractionPlan, ImportHandler
from core.trainer.store import store_trainer
from core.trainer.streamutils import streamingiterload
from core.trainer.trainer import Trainer, list_to_dict, TransformerNotFound


def main(argv=None):
    program_name = os.path.basename(sys.argv[0])
    program_version = 'v%s' % __version__
    program_version_message = '%%(prog)s %s ' % (program_version, )
    program_shortdesc = __import__('__main__').__doc__.split('\n')[1]
    try:
        # Setup argument parser
        parser = ArgumentParser(
            description=program_shortdesc,
            formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument('-V', '--version', action='version',
                            version=program_version_message)
        parser.add_argument('-d', '--debug', dest='debug',
                            action='store_true',
                            help='store extracted data to given file.',
                            default=False)
        parser.add_argument('-o', '--output', dest='output',
                            help='store trained classifier to given file.',
                            metavar='output')
        parser.add_argument('-w', '--weights', dest='weights',
                            help='store feature weights to given file.',
                            metavar='weight-file')
        parser.add_argument('-s', '--store-vect', dest='store_vect',
                            help='store vectorized data to given file.',
                            metavar='store-vect-file')
        parser.add_argument('-i', '--input', dest='input',
                            help='read training data from input file.',
                            metavar='input-file')
        parser.add_argument('-t', '--test', dest='test',
                            help='read testing data from input file.',
                            metavar='test-file')
        parser.add_argument('-tp', '--test-percent', dest='test_percent',
                            help='specify what percentage of the training \
                            data would be used for testing and this part \
                            of the data would be excluded from the training \
                            set and considered only in the testing phase.',
                            metavar='test-percent')
        parser.add_argument('-e', dest='extraction',
                            help='read extraction plan from givenfile.',
                            metavar='extraction-plan-file')
        parser.add_argument('-I', dest='train_params',
                            help='user defined variable for training data. '
                                 'Must be in key=value format',
                            action='append', metavar='train-param')
        parser.add_argument('-T', dest='test_params',
                            help='user defined variable for test data. '
                                 'Must be in key=value format',
                            action='append', metavar='test-param')
        parser.add_argument('--skip-test', dest='skip_tests',
                            help='Skips testing.',
                            action='store_true', default=False)
        parser.add_argument('--transformer-path', dest='transformer_path',
                            help='Path to pretrained transformers.',
                            metavar='transformer_path')
        parser.add_argument(dest='path',
                            help='file containing feature model',
                            metavar='path')

        # Process arguments
        args = parser.parse_args()
        logging_level = logging.INFO
        if args.debug is True:
            logging_level = logging.DEBUG
        logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
                            level=logging_level)

        model = FeatureModel(args.path)
        trainer = Trainer(model)
        if args.transformer_path is not None:
            def get_transformers(name):
                from os import listdir, makedirs
                from os.path import isfile, join, exists, splitext
                import cPickle as pickle
                for f in listdir(args.transformer_path):
                    if isfile(join(args.transformer_path, f)) and splitext(f)[0] == name:
                        with open(join(args.transformer_path, f), 'r') as fp:
                            transformer = fp.read()
                            return pickle.loads(transformer)
                else:
                    raise TransformerNotFound
            trainer.set_transformer_getter(get_transformers)
        test_percent = int(args.test_percent or 0)
        if args.input is not None:
            # Read training data from file
            file_format = os.path.splitext(args.input)[1][1:]
            with open(args.input, 'r') as train_fp:
                trainer.train(streamingiterload(train_fp,
                                                source_format=file_format),
                              test_percent)
                if args.test_percent and args.skip_tests is False \
                   and args.test is None:
                    with open(args.input, 'r') as test_fp:
                        trainer.test(
                            streamingiterload(test_fp,
                                              source_format=file_format),
                            test_percent
                        )

            if args.test is not None and args.skip_tests is False:
                file_format = os.path.splitext(args.test)[1][1:]
                with open(args.test, 'r') as test_fp:
                    trainer.test(streamingiterload(test_fp,
                                                   source_format=file_format))

        elif args.extraction is not None:
            train_context = list_to_dict(args.train_params)
            plan = ExtractionPlan(args.extraction)
            train_handler = ImportHandler(plan, train_context)
            logging.info('Starting training with params:')
            for key, value in train_context.items():
                logging.info('%s --> %s' % (key, value))

            trainer.train(train_handler, test_percent)

            if args.skip_tests is False and args.test_percent\
               and args.test_params is None:
                test_handler = ImportHandler(plan, train_context)
                logging.info('Starting testing with params:')
                for key, value in train_context.iteritems():
                    logging.info('%s --> %s' % (key, value))

                trainer.test(test_handler, test_percent)

            if args.skip_tests is False and args.test_params is not None:
                test_context = list_to_dict(args.test_params)
                test_handler = ImportHandler(plan, test_context)
                logging.info('Starting testing with params:')
                for key, value in test_context.iteritems():
                    logging.info('%s --> %s' % (key, value))

                trainer.test(test_handler)
        else:
            logging.warn('You must define either an input file or '
                         'an extraction plan')
            return 0

        if args.weights is not None:
            logging.info('Storing feature weights to %s' % args.weights)
            with open(args.weights, 'w') as weights_fp:
                trainer.store_feature_weights(weights_fp)

        if args.store_vect is not None:
            logging.info('Storing vectorized data to %s' % args.store_vect)
            trainer.store_vect_data(args.store_vect)

        if args.output is not None:
            logging.info('Storing feature weights to %s' % args.weights)
            with open(args.output, 'w') as trainer_fp:
                store_trainer(trainer, trainer_fp)

    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except SchemaException, e:
        logging.warn('Invalid feature model: %s' % e.message)
        return 1
    except ImportHandlerException, e:
        logging.warn('Invalid extraction plan: %s' % e.message)
        return 1
    # except Exception, e:
    #     raise e
    #     indent = len(program_name) * ' '
    #     sys.stderr.write(program_name + ': ' + repr(e) + '\n')
    #     sys.stderr.write(indent + '  for help use --help')
    #     return 2

if __name__ == '__main__':
    sys.exit(main())
