#! /usr/bin/env python
# encoding: utf-8
"""
importhandler-- extract values from DB according to extraction plan.
"""

# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>

import os
import sys
import logging

from cloudml.importhandler.importhandler import ImportHandlerException, \
    ExtractionPlan, ImportHandler
from cloudml.utils import init_logging


DONE = 0
INVALID_EXTRACTION_PLAN = 1


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    init_logging(args.debug)

    if args.user_params is not None:
        param_list = [x.split('=', 1) for x in args.user_params]
        context = dict((key, value) for (key, value) in param_list)
    else:
        context = {}

    logging.info('User-defined parameters:')
    for key, value in context.items():
        logging.info('%s --> %s' % (key, value))

    try:
        plan = ExtractionPlan(args.path)
        extractor = ImportHandler(plan, context)

    except ImportHandlerException, e:
        logging.warn('Invalid extraction plan: {}'.format(e.message))
        return INVALID_EXTRACTION_PLAN

    if args.output is not None:
        logging.info('Storing data to %s...' % args.output)
        getattr(extractor,
                'store_data_{}'.format(args.format),
                extractor.store_data_json)(args.output)

        logging.info('Total %s lines' % (extractor.count, ))
        logging.info('Ignored %s lines' % (extractor.ignored, ))
    return DONE


def create_parser():
    """ Setups argument parser """
    from cloudml.importhandler import __version__
    from argparse import ArgumentParser
    from argparse import RawDescriptionHelpFormatter
    program_version = 'v%s' % __version__
    program_version_message = '%%(prog)s %s ' % (program_version, )
    program_shortdesc = __import__('__main__').__doc__
    parser = ArgumentParser(
        description=program_shortdesc,
        formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('-o', '--output', dest='output',
                        help='store extracted data to given file.',
                        metavar='output')
    parser.add_argument('-d', '--debug', dest='debug',
                        action='store_true',
                        help='store extracted data to given file.',
                        default=False)
    parser.add_argument('-U', dest='user_params',
                        help='user defined variable', action='append',
                        metavar='user-param')
    parser.add_argument('-V', '--version', action='version',
                        version=program_version_message)
    parser.add_argument(
        '-f', '--format', dest='format',
        help='store extracted data using given format (json or csv).',
        metavar='format', default='json')
    parser.add_argument(dest='path',
                        help='file containing extraction plan',
                        metavar='path')
    return parser


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.warn('keyboard interrupt')
