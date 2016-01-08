#! /usr/bin/env python
# encoding: utf-8
"""
The predictor loads a trained classifier and its configuration, and attempts to
classify incoming data.
"""

# Authors: Ioannis Foukarakis <ifoukarakis@cloud.upwork.com>
#          Nikolay Melnik <nmelnik@cloud.upwork.com>

import logging
import os
import sys
import csv
import sklearn.metrics as metrics

from cloudml.importhandler.importhandler import ImportHandlerException, \
    ExtractionPlan, ImportHandler
from cloudml.trainer.streamutils import streamingiterload
from cloudml.trainer.store import load_trainer
from cloudml.trainer.trainer import list_to_dict
from cloudml.trainer.exceptions import InvalidTrainerFile
from cloudml.utils import init_logging, determine_data_format


DONE = 0
INVALID_EXTRACTION_PLAN = 1
INVALID_TRAINER = 2
PARAMETERS_REQUIRED = 3


def roc(iterator, trainer, params):
    result = trainer.predict(iterator, store_vect_data=True)

    # TODO: multiclass support
    probs = [p[1] for p in result['probs']]

    fpr, tpr, thresholds = metrics.roc_curve(result['true_labels'], probs)
    roc_auc = metrics.auc(fpr, tpr)
    logging.info('Area under the ROC curve: %s' % (roc_auc))


def dump_results_csv(iterator, trainer, params):
    out = params.get('out', 'result.csv')
    fields = []
    if 'fields' in params and len(params['fields']) > 0:
        fields = params['fields'].split(',')
    data = []

    # Callback function to store items
    def store_items(row_data):
        data_row = {}
        for field in fields:
            data_row[field] = row_data.get(field, None)
        data.append(data_row)

    result = trainer.predict(iterator, store_items, store_vect_data=True)
    probs = result['probs']
    with open(out, 'w') as csv_fp:
        csv_out = csv.writer(csv_fp)
        csv_out.writerow(['label'] + result['classes'].tolist() + fields)
        for i in range(len(probs)):
            row = []
            if result.get('labels') is not None:
                row.append(result['labels'][i])
            else:
                row.append(None)
            row.extend(probs[i])
            row.extend([data[i].get(name, None) for name in fields])
            csv_out.writerow(row)

EVALUATION_METHODS = {
    'roc': roc,
    'csv': dump_results_csv
}


def main(argv=None):
    parser = create_parser()
    args = parser.parse_args(argv)
    init_logging(args.debug)

    try:
        with open(args.path, 'r') as fp:
            trainer = load_trainer(fp)
    except InvalidTrainerFile, exc:
        logging.warn('Invalid trainer file: {0!s}'.format(exc))
        return INVALID_TRAINER

    iterator = None
    if args.input is not None:
        # Read evaluation data from file.
        eval_fp = open(args.input, 'r')
        file_format = determine_data_format(args.input)
        iterator = streamingiterload(eval_fp, source_format=file_format)
    elif args.extraction is not None:
        # Use import handler
        try:
            eval_context = list_to_dict(args.eval_params)
            plan = ExtractionPlan(args.extraction)
            eval_handler = ImportHandler(plan, eval_context)
        except ImportHandlerException, e:
            logging.warn('Invalid extraction plan: %s' % e.message)
            return INVALID_EXTRACTION_PLAN

        logging.info('Starting training with params:')
        for key, value in eval_context.items():
            logging.info('%s --> %s' % (key, value))

        iterator = eval_handler
    else:
        # TODO: Add mutually exclusive group
        logging.warn('Need to either specify -i or -e')
        parser.print_help()
        return PARAMETERS_REQUIRED

    eval_method = EVALUATION_METHODS.get(args.method)
    if eval_method is not None:
        eval_method(iterator, trainer, list_to_dict(args.params))
    if args.input is not None:
        eval_fp.close()

    if args.store_vect is not None:
        logging.info('Storing vectorized data to %s' % args.store_vect)
        trainer.store_vect_data(
            trainer.predict_data.values(), args.store_vect)

    return DONE


def create_parser():
    """ Setups argument parser """
    from argparse import ArgumentParser, RawDescriptionHelpFormatter
    from cloudml.trainer import __version__
    program_version = 'v%s' % __version__
    program_version_message = '%%(prog)s %s ' % (program_version, )
    program_shortdesc = __import__('__main__').__doc__
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
        '-s', '--store-vect', dest='store_vect',
        help='store vectorized data to given file.',
        metavar='store-vect-file')
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
    parser.add_argument(dest='path', metavar='path',
                        help='file containing trained classifier')
    return parser

if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.warn('keyboard interrupt')
