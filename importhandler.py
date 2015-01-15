#! /usr/bin/env python
# encoding: utf-8
"""
importhandler-- extract values from DB according to extraction plan.

It defines classes ImportHandlerException, ExtractionPlan and ImportHandler

@author:     ifoukarakis

@copyright: 2013 oDesk. All rights reserved.

"""

import os
import sys
import logging
import colorer

from core.importhandler import __version__
from argparse import ArgumentParser
from argparse import RawDescriptionHelpFormatter

#from core.importhandler.importhandler import ImportHandlerException, ExtractionPlan, ImportHandler
from core.xmlimporthandler.importhandler import ImportHandlerException, ExtractionPlan, ImportHandler


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
        parser.add_argument('-f', '--format', dest='format',
                            help='store extracted data using given format (json or csv).',
                            metavar='format', default='json')
        parser.add_argument(dest='path',
                            help='file containing extraction plan',
                            metavar='path')

        # Process arguments
        args = parser.parse_args()

        logging_level = logging.INFO
        if args.debug is True:
            logging_level = logging.DEBUG
        logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
                            level=logging_level)

        if args.user_params is not None:
            param_list = [x.split('=', 1) for x in args.user_params]
            context = dict((key, value) for (key, value) in param_list)
        else:
            context = {}

        logging.info('User-defined parameters:')
        for key, value in context.items():
            logging.info('%s --> %s' % (key, value))
        # Read extraction plan
        plan = ExtractionPlan(args.path)
        # Create import handler
        extractor = ImportHandler(plan, context)

        if args.output is not None:
            logging.info('Storing data to %s...' % args.output)
            getattr(extractor,
                    'store_data_{}'.format(args.format),
                    extractor.store_data_json)(args.output)

            logging.info('Total %s lines' % (extractor.count, ))
            logging.info('Ignored %s lines' % (extractor.ignored, ))
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    # except ImportHandlerException, e:
    #     logging.warn(e.message)
    #     return 1
    # except Exception, e:
    #     indent = len(program_name) * ' '
    #     sys.stderr.write(program_name + ': ' + repr(e) + '\n')
    #     sys.stderr.write(indent + '  for help use --help')
    #     return 2

if __name__ == '__main__':
    sys.exit(main())
