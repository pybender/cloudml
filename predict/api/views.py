import logging
import traceback

from flask import request, jsonify

from core.importhandler.importhandler import ExtractionPlan, \
    RequestImportHandler

from api import app

from api.utils import odesk_error_response, ERR_NO_SUCH_MODEL, \
    ERR_NO_SUCH_IMPORT_HANDLER, ERR_PREDICT, ERR_NO_MODELS


@app.route('/cloudml/model', methods=['GET'])
def list_models():
    """
    Lists all known trained models.
    """
    logging.info('Request to view available models')

    if len(app.models) == 0:
        return odesk_error_response(404, ERR_NO_MODELS, 'No models loaded')

    result = []
    for name, model in app.models.items():
        result.append({'name': name,
                       'schema-name': model._feature_model.schema_name,
                       'created': model.train_time})

    return jsonify({'models': result})


@app.route('/cloudml/import/handler', methods=['GET'])
def list_import_handlers():
    """
    Lists all known import handlers.
    """
    logging.info('Request to view available import handlers')

    if len(app.import_handlers) == 0:
        return odesk_error_response(404, ERR_NO_MODELS,
                                    'No import handlers loaded')

    result = []
    for name, handler in app.import_handlers.items():
        result.append({'name': name})

    return jsonify({'import_handlers': result})


@app.route('/cloudml/model/<regex("[\w\.]*"):model>/\
<regex("[\w\.]*"):import_handler>/predict', methods=['POST'])
def predict(model, import_handler):
    """
    Predict labels and probabilities
    """
    hndl = app.import_handlers.get(import_handler, None)
    if hndl is None:
        msg = "Import handler %s doesn\'t exist" % import_handler
        logging.error(msg)
        return odesk_error_response(404, ERR_NO_SUCH_IMPORT_HANDLER, msg)

    trainer = app.models.get(model, None)
    if trainer is None:
        msg = "Model %s doesn\'t exist" % model
        logging.error(msg)
        return odesk_error_response(404, ERR_NO_SUCH_MODEL, msg)

    data = [request.form, ]
    plan = ExtractionPlan(hndl, is_file=False)
    request_import_handler = RequestImportHandler(plan, data)

    try:
        probabilities = trainer.predict(request_import_handler,
                                        ignore_error=False)
    except Exception, exc:
        msg = "Predict error: %s:%s" % (exc.__class__.__name__, exc)
        logging.error(msg)
        return odesk_error_response(500, ERR_PREDICT,
                                    msg, traceback=traceback.format_exc())

    probs = probabilities['probs'][0]
    labels = probabilities['labels']
    probs_with_labels = []
    probs = probs.tolist() if not (probs is None) else []
    labels = labels.tolist() if not (labels is None) else []
    for label, prob in zip(labels, probs):
        probs_with_labels.append({'label': label, 'prob': prob})
    prob, label = sorted(zip(probs, labels),
                         lambda x, y: cmp(x[0], y[0]),
                         reverse=True)[0]
    return jsonify({'prediction': label, 'probs': probs_with_labels}), 201
