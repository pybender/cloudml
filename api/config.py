SECRET_KEY = 'CHANGE_ME'
SQLALCHEMY_DATABASE_URI = 'sqlite:///cloudml.db'
STATIC_ROOT = None
UPLOAD_FOLDER = 'models'
MAX_CONTENT_LENGTH = 128 * 1024 * 1024
DEBUG = True

# Celery specific settings
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ENABLE_UTC = True
#BROKER_URL = 'sqla+sqlite:///celerydb.sqlite'
BROKER_URL = 'amqp://cloudml:cloudml@localhost:5672/cloudml1'
CELERY_RESULT_BACKEND = ''
