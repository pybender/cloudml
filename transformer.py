#! /usr/bin/env python
# encoding: utf-8
"""
Command line util for train transformers
"""

# Author: Nikolay Melnik <nmelnik@cloud.upwork.com>

import os
import sys
import logging
import cPickle as pickle

from cloudml.importhandler.importhandler import ImportHandlerException, \
    ExtractionPlan, ImportHandler

from cloudml.trainer import __version__
from cloudml.transformers.transformer import Transformer, \
    TransformerSchemaException
from cloudml.trainer.trainer import list_to_dict
from cloudml.trainer.streamutils import streamingiterload
from cloudml.utils import init_logging

DONE = 0
INVALID_TRANSFORMER_CONFIG = 1
INVALID_EXTRACTION_PLAN = 2
PARAMETERS_REQUIRED = 3


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    init_logging(args.debug)

    try:
        transformer = Transformer(args.path)
    except TransformerSchemaException, e:
        logging.warn('Invalid feature model: %s' % e.message)
        return INVALID_TRANSFORMER_CONFIG

    if args.input is not None:
        file_format = os.path.splitext(args.input)[1][1:]
        with open(args.input, 'r') as train_fp:
            transformer.train(
                streamingiterload(train_fp, source_format=file_format))
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
        transformer.train(train_handler)
    else:
        logging.warn('You must define either an input file or '
                     'an extraction plan')
        parser.print_help()
        return PARAMETERS_REQUIRED

    if args.output is not None:
        logging.info('Storing transformer to %s' % args.output)
        with open(args.output, 'w') as trainer_fp:
            pickle.dump(transformer, trainer_fp)

    return DONE


def create_parser():
    """ Setups argument parser """
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    program_version = 'v%s' % __version__
    program_version_message = '%%(prog)s %s ' % (program_version, )
    program_shortdesc = __import__('__main__').__doc__
    parser = ArgumentParser(
        description=program_shortdesc,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-V', '--version', action='version',
                        version=program_version_message)
    parser.add_argument('-i', '--input', dest='input',
                        help='read training data from input file.',
                        metavar='input-file')
    parser.add_argument('-e', dest='extraction',
                        help='read extraction plan from givenfile.',
                        metavar='extraction-plan-file')
    parser.add_argument('-I', dest='train_params',
                        help='user defined variable for training data. '
                             'Must be in key=value format',
                        action='append', metavar='train-param')
    parser.add_argument('-o', '--output', dest='output',
                        help='store trained classifier to given file.',
                        metavar='output')
    parser.add_argument(dest='path',
                        help='file containing feature model',
                        metavar='path')
    parser.add_argument('-d', '--debug', dest='debug', action='store_true',
                        help='show debug logging messages.', default=False)
    return parser


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.warn('keybord interrupt')
