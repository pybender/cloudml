S3_LOG_URI = '/logs'
AMAZON_ACCESS_TOKEN = 'token'
AMAZON_TOKEN_SECRET = 'secret'
BUCKET_NAME = 's3-bucket-name'
DEFAILT_AMI_VERSION = '3.1.0'
DEFAULT_INSTANCE_TYPE = 'm1.small'
DEFAULT_NUM_INSTANCES = 1

try:
    from local_config import *
except:
    pass
