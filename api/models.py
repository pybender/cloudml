from datetime import datetime

from api import db
from api.utils import Serializer, JSONEncodedDict


class Model(db.Model, Serializer):
    __public__ = ['id', 'name', 'created_on', 'import_params']
    __all_public__ = ('id', 'name', 'created_on', 'import_params',
                      'positive_weights', 'negative_weights',
                      'positive_weights_tree', 'negative_weights_tree')
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    features = db.Column(db.Text)
    created_on = db.Column(db.DateTime)
    trainer = db.Column(db.PickleType)
    import_handler = db.Column(db.Text)
    import_params = db.Column(JSONEncodedDict)
    tests = db.relationship('Test', backref='model',
                            lazy='dynamic')
    positive_weights = db.Column(JSONEncodedDict)
    negative_weights = db.Column(JSONEncodedDict)
    positive_weights_tree = db.Column(JSONEncodedDict)
    negative_weights_tree = db.Column(JSONEncodedDict)

    def __init__(self, name):
        self.name = name
        self.created_on = datetime.now()

    def set_weights(self, positive, negative):
        from helpers.weights import calc_weights_css, weights2tree
        self.positive_weights = calc_weights_css(positive, 'green')
        self.negative_weights = calc_weights_css(negative, 'red')
        self.negative_weights.reverse()
        self.positive_weights_tree = weights2tree(self.positive_weights)
        self.negative_weights_tree = weights2tree(self.negative_weights)

    def __repr__(self):
        return '<Model %r>' % self.name


class Test(db.Model, Serializer):
    __public__ = ('id', 'name', 'created_on', 'accuracy',
                  'parameters', 'data_count', )
    __all_public__ = ('id', 'name', 'created_on', 'accuracy', 'parameters',
                      'classes_set', 'metrics', 'data_count', )

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    parameters = db.Column(JSONEncodedDict)
    created_on = db.Column(db.DateTime)
    model_id = db.Column(db.Integer, db.ForeignKey('model.id'))
    data = db.relationship('Data', backref='test', lazy='dynamic')

    metrics = db.Column(JSONEncodedDict)
    accuracy = db.Column(db.Float)
    classes_set = db.Column(JSONEncodedDict)

    def __init__(self, name):
        self.name = name
        self.created_on = datetime.now()

    def __repr__(self):
        return '<Test %r>' % self.name

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
