
import os

from fabdeploy.api import DefaultConf


DIRNAME = os.path.dirname(__file__)


class HereConf(DefaultConf):
    """Local development settings. Change these for your machine."""

    address = 'atmel@localhost'
    sudo_user = 'atmel'
    src_path = os.path.dirname(DIRNAME)
    env_path = os.path.join(DIRNAME, 've')
    django_dir = 'sna-client/sna_client'


class BaseConf(DefaultConf):
    """Base deployment settings."""

    # %(django_dir)s is set relative to %(project_path)s, which in our
    # case is repository root.
    # In this directory should reside:
    # - Django settings and manage.py files
    # - 500/50x/404 HTML pages (see nginx_apache.config and apache.config).
    django_dir = 'api'
    
    active_public_link = ['%(active_django_link)s', 'public']

    pip_req_path = ''
    pip_req_name = 'requirements.txt'
    
    supervisor_prefix = 'cloudml_'
    supervisord_config_lfile = 'supervisor/supervisord.conf'
    supervisord_config_file = ['%(supervisor_config_path)s', 'supervisord.conf']

    supervisor__log_path = ['%(var_path)s', 'log', 'supervisor']
    #supervisor_programs = ['celeryd', 'celerybeat', 'celerycam']
    
    rabbitmq_host = 'localhost'
    rabbitmq_port = 5672
    rabbitmq_user = '%(user)s'
    rabbitmq_password = '%(user)s'
    rabbitmq_vhost =  '%(user)s'

    broker_backend = 'amqplib'
    broker_host = 'localhost'
    broker_port = '%(rabbitmq_port)s'
    broker_user = '%(rabbitmq_user)s'
    broker_password = '%(rabbitmq_password)s'
    broker_vhost = '%(rabbitmq_vhost)s'

    db_name = 'cloudml'
    
    # Local settings file
    local_settings_file = 'local_settings.py'
    

class StagingConf(BaseConf):
    """Settings specific to production environment."""

    address = 'cloudml@172.27.67.106'

    sudo_user = 'nmelnik'

    # Code from this branch will be deployed.
    branch = 'master'

    server_name = 'cloudml.staging.match.odesk.com'
    # For Apache ServerAdmin directive
    server_admin = 'nmelnik@odesk.com'
    # Apache will serve WSGI on this port. (Nginx is front-end.)
    apache_port = 50005
    gunicorn_port = 8020

    # For pip extra index url config
    odeskps_pypi_user = 'nmelnik@odesk.com'
    odeskps_pypi_password = 'nmelnik'

    # Once on production, this file will replace %(local_settings_file)s
    # It should be a Jinja2 template, and can make use of fabdeploy config
    # variables.
    remote_settings_lfile = 'staging_settings.py.tpl'

    # ADMINS / MANAGERS (traces will be sent there)
    django_admins = (('Nikolay melnik', 'nmelnik@odesk.com'), )
