from datetime import datetime

from api import db
from api.utils import JSONEncodedDict
from api.serialization import Serializer


class Model(db.Model, Serializer):
    __public__ = ['id', 'name', 'created_on', 'import_params',
                  'importhandler', 'status']
    __all_public__ = ('id', 'name', 'created_on', 'status', 'import_params',
                      'positive_weights', 'negative_weights',
                      'positive_weights_tree', 'negative_weights_tree',
                      'importhandler', 'features', 'latest_test')
    STATUS_NEW = 'New'
    STATUS_QUEUED = 'Queued'
    STATUS_TRAINING = 'Training'
    STATUS_TRAINED = 'Trained'
    STATUS_ERROR = 'Error'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True)
    created_on = db.Column(db.DateTime)
    status = db.Column(db.String(10), default=STATUS_NEW)

    features = db.Column(db.Text)
    import_params = db.Column(JSONEncodedDict)
    # Import handler for tests
    importhandler = db.Column(db.Text)
    train_importhandler = db.Column(db.Text)

    # Trainer specific fields
    trainer = db.Column(db.PickleType)
    # some denormalization:
    positive_weights = db.Column(JSONEncodedDict)
    negative_weights = db.Column(JSONEncodedDict)
    positive_weights_tree = db.Column(JSONEncodedDict)
    negative_weights_tree = db.Column(JSONEncodedDict)

    tests = db.relationship('Test', backref='model',
                            lazy='dynamic')

    def __init__(self, name):
        self.name = name
        self.created_on = datetime.now()

    def get_import_handler(self, parameters=None, is_test=False):
        from core.importhandler.importhandler import ExtractionPlan, ImportHandler
        plan = ExtractionPlan(self.importhandler if is_test
                              else self.train_importhandler, is_file=False)
        handler = ImportHandler(plan, parameters)
        return handler

    def run_test(self, parameters=True):
        test_handler = self.get_import_handler(parameters, is_test=True)
        metrics = self.trainer.test(test_handler)
        raw_data = self.trainer._raw_data
        return metrics, raw_data

    def set_trainer(self, trainer):
        self.trainer = trainer
        self.set_weights(**trainer.get_weights())

    def set_weights(self, positive, negative):
        from helpers.weights import calc_weights_css, weights2tree
        self.positive_weights = calc_weights_css(positive, 'green')
        self.negative_weights = calc_weights_css(negative, 'red')
        self.negative_weights.reverse()
        self.positive_weights_tree = weights2tree(self.positive_weights)
        self.negative_weights_tree = weights2tree(self.negative_weights)

    @property
    def latest_test(self):
        test = self.tests.order_by('-id').first()
        if test:
            return test.to_brief_dict()

    def __repr__(self):
        return '<Model %r>' % self.name


class Test(db.Model, Serializer):
    __public__ = ('id', 'name', 'created_on', 'accuracy',
                  'parameters', 'data_count', 'status')
    __all_public__ = ('id', 'name', 'created_on', 'accuracy', 'parameters',
                      'classes_set', 'metrics', 'data_count',
                      'status')
    STATUS_QUEUED = 'Queued'
    STATUS_IN_PROGRESS = 'In Progress'
    STATUS_COMPLETED = 'Completed'
    STATUS_ERROR = 'Error'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    status = db.Column(db.String(10), default=STATUS_IN_PROGRESS)
    created_on = db.Column(db.DateTime)
    model_id = db.Column(db.Integer, db.ForeignKey('model.id'))

    # Import params
    parameters = db.Column(JSONEncodedDict)
    data = db.relationship('Data', backref='test', lazy='dynamic')
    classes_set = db.Column(JSONEncodedDict)

    accuracy = db.Column(db.Float)
    metrics = db.Column(JSONEncodedDict)

    def __init__(self, model):
        self.model_id = model.id
        self.name = Test.generate_name(model)
        self.created_on = datetime.now()

    def __repr__(self):
        return '<Test %r>' % self.name

    @classmethod
    def generate_name(cls, model, base_name='Test'):
        count = model.tests.count()
        return "%s-%s" % (base_name, count + 1)

    @property
    def data_count(self):
        return self.data.count()


class Data(db.Model, Serializer):
    __public__ = ['id', 'created_on', 'data_input']
    __all_public__ = ['id', 'created_on', 'data_input',
                      'weighted_data_input']

    id = db.Column(db.Integer, primary_key=True)
    created_on = db.Column(db.DateTime)
    data_input = db.Column(JSONEncodedDict)
    weighted_data_input = db.Column(JSONEncodedDict)
    result = db.Column(db.Text)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'))

    def __init__(self, data_input, test_id, weighted_data_input):
        self.data_input = data_input
        self.weighted_data_input = weighted_data_input
        self.created_on = datetime.now()
        self.test_id = test_id

    @classmethod
    def loads_from_raw_data(cls, model, test, raw_data):
        from helpers.weights import get_weighted_data
        for row in raw_data:
            weighted_data_input = get_weighted_data(model, row)
            data = cls(row, test.id, weighted_data_input)
            db.session.add(data)
        db.session.commit()
