
import os

from fabdeploy.api import DefaultConf


DIRNAME = os.path.dirname(__file__)


class HereConf(DefaultConf):
    """Local development settings. Change these for your machine."""

    address = 'atmel@home'
    sudo_user = 'atmel'
    src_path = os.path.dirname(DIRNAME)
    env_path = os.path.join(DIRNAME, 've')
    django_dir = 'api'

    rabbitmq_host = 'localhost'
    rabbitmq_port = 5672
    rabbitmq_user = 'cloudml'
    rabbitmq_password = 'cloudml'
    rabbitmq_vhost =  'cloudml'


class BaseConf(DefaultConf):
    """Base deployment settings."""

    django_dir = 'api'

    ui_scripts_dir = ['home_path', 'ui', 'app', 'scripts']
    current_public_link = ['%(current_project_link)s', 'ui', '_public']
    current_docs_link = ['%(current_project_link)s', 'docs', 'build', 'html']

    pip_req_path = ''
    pip_req_name = 'requirements.txt'

    supervisor_prefix = 'cloudmlui_'
    supervisord_config_lfile = 'supervisor/supervisord.conf'
    supervisord_config_file = ['%(supervisor_config_path)s', 'supervisord.conf']

    supervisor__log_path = ['%(var_path)s', 'log', 'supervisor']
    supervisor_programs = ['celeryd', 'celerycam', 'gunicorn']

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
    local_settings_file = 'config.py'


class DevConf(BaseConf):
    """Settings specific to dev environment."""

    address = 'cloudml@172.27.68.147'

    sudo_user = 'nmelnik'
    home_path = '/webapps/cloudml'

    # Code from this branch will be deployed.
    branch = 'master'

    server_name = '172.27.68.147'
    # For Apache ServerAdmin directive
    server_admin = 'ifoukarakis@odesk.com'
    # Apache will serve WSGI on this port. (Nginx is front-end.)
    apache_port = 80

    remote_settings_lfile = 'prod_config.py.tpl'


class StagingConf(BaseConf):
    """Settings specific to production environment."""

    address = 'cloudml@172.27.67.106'

    sudo_user = 'nmelnik'
    home_path = '/webapps/cloudml'

    # Code from this branch will be deployed.
    branch = 'staging'

    server_name = 'cloudml.staging.match.odesk.com'
    # For Apache ServerAdmin directive
    server_admin = 'nmelnik@odesk.com'
    # Apache will serve WSGI on this port. (Nginx is front-end.)
    gunicorn_port = 8020

    # Once on production, this file will replace %(local_settings_file)s
    # It should be a Jinja2 template, and can make use of fabdeploy config
    # variables.
    remote_settings_lfile = 'staging_config.py.tpl'



class ProductionConf(BaseConf):
    """Settings specific to production environment."""

    address = 'cloudml@172.27.77.141'

    home_path = '/webapps/cloudmlui'

    sudo_user = 'papadimitriou'

    # Code from this branch will be deployed.
    branch = 'master'

    server_name = 'cloudml.match.odesk.com'
    # For Nginx ServerAdmin directive
    server_admin = 'nmelnik@odesk.com'
    # Gunicorn will serve WSGI on this port. (Nginx is front-end.)
    gunicorn_port = 5000

    remote_settings_lfile = 'prod_config.py.tpl'
