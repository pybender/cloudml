from datetime import datetime

from api import db
from api.utils import Serializer, JSONEncodedDict, PickledValue


class Model(db.Model, Serializer):
    __public__ = ['id', 'name', 'created_on', 'import_params']

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))
    features = db.Column(db.Text)
    created_on = db.Column(db.DateTime)
    trainer = db.Column(db.PickleType)
    import_handler = db.Column(db.Text)
    import_params = db.Column(JSONEncodedDict)
    tests = db.relationship('Test', backref='model',
                            lazy='dynamic')

    def __init__(self, name):
        self.name = name
        self.created_on = datetime.now()

    def __repr__(self):
        return '<Model %r>' % self.name


class Test(db.Model, Serializer):
    __public__ = ('id', 'name', 'created_on', 'accuracy', 'parameters')
    __all_public__ = ('id', 'name', 'created_on', 'accuracy', 'parameters',
                      'classes_set', 'metrics')

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


class Data(db.Model, Serializer):
    __public__ = ['id', 'created_on', 'data_input']

    id = db.Column(db.Integer, primary_key=True)
    created_on = db.Column(db.DateTime)
    data_input = db.Column(JSONEncodedDict)
    result = db.Column(db.Text)
    test_id = db.Column(db.Integer, db.ForeignKey('test.id'))

    def __init__(self, data_input, test_id):
        self.data_input = data_input
        self.created_on = datetime.now()
        self.test_id = test_id

    @classmethod
    def loads_from_raw_data(cls, test, raw_data):
        for row in raw_data:
            data = cls(row, test.id)
            db.session.add(data)
        db.session.commit()
