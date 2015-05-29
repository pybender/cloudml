#! /usr/bin/env python
# encoding: utf-8
"""
Command line util for train transformers

@author:     nmelnik

@copyright:  2014 odesk. All rights reserved.
"""

# Author: Nikolay Melnik <nmelnik@upwork.com>

import os
import sys
import logging
import cPickle as pickle
import colorer

from core.importhandler.importhandler import ImportHandlerException, \
    ExtractionPlan, ImportHandler

from core.trainer import __version__
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from core.transformers.transformer import Transformer, \
    TransformerSchemaException
from core.trainer.streamutils import streamingiterload


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

        # Process arguments
        args = parser.parse_args()
        transformer = Transformer(args.path)
        if args.input is not None:
            file_format = os.path.splitext(args.input)[1][1:]
            with open(args.input, 'r') as train_fp:
                transformer.train(
                    streamingiterload(train_fp, source_format=file_format))
        elif args.extraction is not None:
            train_context = list_to_dict(args.train_params)
            plan = ExtractionPlan(args.extraction)
            train_handler = ImportHandler(plan, train_context)
            logging.info('Starting training with params:')
            for key, value in train_context.items():
                logging.info('%s --> %s' % (key, value))
            transformer.train(train_handler)

        if args.output is not None:
            logging.info('Storing transformer to %s' % args.output)
            with open(args.output, 'w') as trainer_fp:
                pickle.dump(transformer, trainer_fp)

    except KeyboardInterrupt:
        # handle keyboard interrupt
        return 0
    except TransformerSchemaException, e:
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
