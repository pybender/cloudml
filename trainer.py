#! /usr/bin/env python
# encoding: utf-8
"""
Model trainer command line utility.
"""

# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>

import os
import sys
import logging

from cloudml.trainer.config import FeatureModel, SchemaException
from cloudml.trainer import __version__
from cloudml.importhandler.exceptions import ImportHandlerException
from cloudml.importhandler.importhandler import ExtractionPlan, ImportHandler
from cloudml.trainer.store import store_trainer
from cloudml.trainer.streamutils import streamingiterload
from cloudml.trainer.trainer import Trainer, list_to_dict, TransformerNotFound
from cloudml.utils import init_logging, determine_data_format


# Status codes
DONE = 0
INVALID_FEATURE_MODEL = 1
INVALID_EXTRACTION_PLAN = 2
PARAMETERS_REQUIRED = 3
PERCENT_ERR_FORMAT = "Percent value '{0}' would be ignored. \
Should be value from 0 to 100."


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    init_logging(args.debug)

    try:
        model = FeatureModel(args.path)
    except IOError, exc:
        logging.warn("Can't load features file. {0!s}".format(exc))
        return INVALID_FEATURE_MODEL
    except SchemaException, exc:
        logging.warn('Invalid feature model: {0!s}'.format(exc))
        return INVALID_FEATURE_MODEL

    trainer = Trainer(model)

    if args.transformer_path is not None:
        # defines pretrained transformers path
        trainer.set_transformer_getter(
            transformer_getter(args.transformer_path))

    test_percent = parse_percent(args.test_percent)
    if args.input is not None:
        # Read training data from file
        file_format = determine_data_format(args.input)
        with open(args.input, 'r') as train_fp:
            logging.info("Training the model using input file dataset.")
            trainer.train(
                streamingiterload(train_fp, source_format=file_format),
                test_percent,
                store_vect_data=args.store_train_vect is not None)

            if args.store_train_vect is not None:
                logging.info('Storing train vectorized data to %s' %
                             args.store_train_vect)
                trainer.vect_data2csv(args.store_train_vect)

            if test_percent != 0 and args.skip_tests is False \
               and args.test is None:
                with open(args.input, 'r') as test_fp:
                    trainer.test(
                        streamingiterload(
                            test_fp, source_format=file_format),
                        test_percent
                    )

        if args.test is not None and args.skip_tests is False:
            file_format = os.path.splitext(args.test)[1][1:]
            with open(args.test, 'r') as test_fp:
                trainer.test(streamingiterload(test_fp,
                                               source_format=file_format))

    elif args.extraction is not None:
        train_context = list_to_dict(args.train_params)
        try:
            plan = ExtractionPlan(args.extraction)
            train_handler = ImportHandler(plan, train_context)
        except ImportHandlerException, e:
            logging.warn('Invalid extraction plan: %s' % e.message)
            return INVALID_EXTRACTION_PLAN

        logging.info('Starting training with params:')
        for key, value in train_context.items():
            logging.info('%s --> %s' % (key, value))

        trainer.train(train_handler, test_percent)

        if args.skip_tests is False:
            if test_percent != 0:
                if args.test_params is None:
                    test_handler = ImportHandler(plan, train_context)
                    logging.info('Starting testing with params:')
                    for key, value in train_context.iteritems():
                        logging.info('%s --> %s' % (key, value))

                    trainer.test(test_handler, test_percent)
                else:
                    logging.warn("Either test percent, either test "
                                 "parameters should be defined. Not both.")
                    return PARAMETERS_REQUIRED

            elif args.test_params is not None:
                test_context = list_to_dict(args.test_params)
                test_handler = ImportHandler(plan, test_context)
                logging.info('Starting testing with params:')
                for key, value in test_context.iteritems():
                    logging.info('%s --> %s' % (key, value))

                trainer.test(test_handler)
    else:
        logging.warn('You must define either an input file or '
                     'an extraction plan')
        parser.print_help()
        return PARAMETERS_REQUIRED

    if args.weights is not None:
        logging.info('Storing feature weights to %s' % args.weights)
        with open(args.weights, 'w') as weights_fp:
            trainer.store_feature_weights(weights_fp)

    if args.store_vect is not None:
        logging.info('Storing vectorized data to %s' % args.store_vect)
        if not hasattr(trainer, 'metrics'):
            logging.warn('Model was trained, but not evaluated. '
                         'You need to add --test or --test-percent param.')
            parser.print_help()
            return PARAMETERS_REQUIRED
        trainer.store_vect_data(
            trainer.metrics._true_data.values(), args.store_vect)

    if args.output is not None:
        logging.info('Storing feature weights to %s' % args.weights)
        with open(args.output, 'w') as trainer_fp:
            store_trainer(trainer, trainer_fp)

    return DONE


def create_parser():
    """ Setups argument parser """
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    program_shortdesc = __import__('__main__').__doc__
    program_version = 'v%s' % __version__
    program_version_message = '%%(prog)s %s ' % (program_version, )
    parser = ArgumentParser(
        description=program_shortdesc,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-V', '--version', action='version',
                        version=program_version_message)
    parser.add_argument('-d', '--debug', dest='debug',
                        action='store_true',
                        help='debug mode',
                        default=False)
    parser.add_argument('-o', '--output', dest='output',
                        help='store trained classifier to given file.',
                        metavar='output')
    parser.add_argument('-w', '--weights', dest='weights',
                        help='store feature weights to given file.',
                        metavar='weight-file')
    parser.add_argument('-s', '--store-vect', dest='store_vect',
                        help='store test vectorized data to given file.',
                        metavar='store-vect-file')
    parser.add_argument('-v', '--store-train-vect',
                        dest='store_train_vect',
                        help='store train vectorized data to given file.',
                        metavar='store-train-vect-file')
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
    return parser


def transformer_getter(transformer_path):
    def get_transformer(name):
        from os import listdir, makedirs
        from os.path import isfile, join, exists, splitext
        import cPickle as pickle
        for f in listdir(transformer_path):
            if isfile(join(transformer_path, f)) and splitext(f)[0] == name:
                with open(join(transformer_path, f), 'r') as fp:
                    transformer = fp.read()
                    return pickle.loads(transformer)
        else:
            raise TransformerNotFound('Transformer {0} not found'.format(name))
    return get_transformer


def parse_percent(percent_str):
    if percent_str is None:
        return 0

    try:
        percent = int(percent_str)
    except (ValueError, TypeError), exc:
        logging.warn(PERCENT_ERR_FORMAT.format(percent_str))
        return 0

    if percent < 0 or percent > 100:
        logging.warn(PERCENT_ERR_FORMAT.format(percent_str))
        return 0

    return percent


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.warn('keybord interrupt')
