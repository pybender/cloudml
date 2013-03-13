#! /usr/bin/env python
# encoding: utf-8
"""
Class for training classifier and evaluating trained model. Uses logistic
regression.


It defines class Trainer.

@author:     ifoukarakis

@copyright:  2013 odesk. All rights reserved.
"""
import logging
import os
import sklearn.metrics as metrics
import sys
import csv
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from core.importhandler.importhandler import ImportHandlerException,\
    ExtractionPlan, ImportHandler
from core.trainer.streamutils import streamingiterload
from core.trainer.store import load_trainer
from core.trainer.trainer import list_to_dict, Trainer


def roc(iterator, trainer, params):
    result = trainer.predict(iterator)
    probs = result['probs']
    fpr, tpr, thresholds = metrics.roc_curve(result['labels'], probs[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    logging.info('Area under the ROC curve: %s' % (roc_auc))


def dump_results_csv(iterator, trainer, params):
    out = params.get('out', 'result.csv')
    fields = []
    if 'fields' in params and len(params['fields']) > 0:
        fields = params['fields'].split(',')
    data = []

    #Callback function to store items
    def store_items(row_data):
        data_row = {}
        for field in fields:
            data_row[field] = row_data.get(field, None)
        data.append(data_row)

    result = trainer.predict(iterator, store_items)
    probs = result['probs']
    with open(out, 'w') as csv_fp:
        csv_out = csv.writer(csv_fp)
        for i in range(len(probs)):
            row = []
            row.extend([data[i].get(name, None) for name in fields])
            if result.get('labels') is not None:
                row.append(result['labels'][i])
            row.extend(probs[i])
            csv_out.writerow(row)

EVALUATION_METHODS = {
    'roc': roc,
    'csv': dump_results_csv
}


def main(argv=None):
    program_name = os.path.basename(sys.argv[0])
    program_version = 'v%s' % logging.__version__
    program_version_message = '%%(prog)s %s ' % (program_version, )
    program_shortdesc = __import__('__main__').__doc__.split('\n')[1]
    try:
        # Setup argument parser
        parser = ArgumentParser(
            description=program_shortdesc,
            formatter_class=RawDescriptionHelpFormatter)
        parser.add_argument(
            '-V', '--version', action='version',
            version=program_version_message)
        parser.add_argument(
            '-d', '--debug', dest='debug',
            action='store_true',
            help='store extracted data to given file.',
            default=False)
        parser.add_argument(
            '-o', '--output', dest='output',
            help='store trained classifier to given file.',
            metavar='output')
        parser.add_argument(
            '-m', '--method', dest='method',
            help='Evaluation method to use',
            choices=EVALUATION_METHODS.keys())
        parser.add_argument(
            '-P', dest='params',
            help='Add parameter for evaluation method',
            action='append', metavar='param')
        parser.add_argument(
            '-i', '--input', dest='input',
            help='read training data from input file.',
            metavar='')
        parser.add_argument(
            '-e', dest='extraction',
            help='read extraction plan from given file.',
            metavar='extraction-plan-file')
        parser.add_argument(
            '-U', dest='eval_params',
            help='user defined variable to use with extraction plan.'
                 'Must be in key=value format',
            action='append', metavar='eval-param')

        parser.add_argument(
            dest='path',
            help='file containing trained classifier',
            metavar='path')

        # Process arguments
        args = parser.parse_args()

        logging_level = logging.INFO
        if args.debug is True:
            logging_level = logging.DEBUG
        logging.basicConfig(format='[%(asctime)s] %(levelname)s - %(message)s',
                            level=logging_level)

        with open(args.path, 'r') as fp:
            trainer = load_trainer(fp)

        iterator = None
        if args.input is not None:
            # Read evaluation data from file.
            with open(args.test, 'r') as eval_fp:
                iterator = streamingiterload(eval_fp)
        elif args.extraction is not None:
            # Use import handler
            eval_context = list_to_dict(args.eval_params)
            plan = ExtractionPlan(args.extraction)
            eval_handler = ImportHandler(plan, eval_context)
            logging.info('Starting training with params:')
            for key, value in eval_context.items():
                logging.info('%s --> %s' % (key, value))

            iterator = eval_handler
        else:
            #TO DO: Add mutually exclusive group
            logging.info('Need to either specify -i or -e')
            parser.print_help()
            return 1

        eval_method = EVALUATION_METHODS.get(args.method)
        if eval_method is not None:
            eval_method(iterator, trainer, list_to_dict(args.params))

    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except ImportHandlerException, e:
        logging.warn('Invalid extraction plan: %s' % e.message)
        return 1
    except Exception, e:
        raise e
        indent = len(program_name) * ' '
        sys.stderr.write(program_name + ': ' + repr(e) + '\n')
        sys.stderr.write(indent + '  for help use --help')
        return 2

if __name__ == '__main__':
    sys.exit(main())
