
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
    active_public_link = ['%(active_src_link)s', 'ui', '_public']
    active_docs_link = ['%(active_src_link)s', 'docs', 'build', 'html']

    pip_req_path = ''
    pip_req_name = 'requirements.txt'
    
    supervisor_prefix = 'cloudml_'
    supervisord_config_lfile = 'supervisor/supervisord.conf'
    supervisord_config_file = ['%(supervisor_config_path)s', 'supervisord.conf']

    supervisor__log_path = ['%(var_path)s', 'log', 'supervisor']
    supervisor_programs = ['celeryd', 'celerycam']
    
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


    # Once on production, this file will replace %(local_settings_file)s
    # It should be a Jinja2 template, and can make use of fabdeploy config
    # variables.
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
    apache_port = 5000
    gunicorn_port = 8020

    # For pip extra index url config
    odeskps_pypi_user = 'nmelnik@odesk.com'
    odeskps_pypi_password = 'nmelnik'

    # Once on production, this file will replace %(local_settings_file)s
    # It should be a Jinja2 template, and can make use of fabdeploy config
    # variables.
    remote_settings_lfile = 'staging_config.py.tpl'


class Production1Conf(BaseConf):
    """Settings specific to production environment."""

    #address = 'cloudml@172.27.77.242'
    address = 'cloudml@172.27.77.141'

    sudo_user = 'nmelnik'

    # Code from this branch will be deployed.
    branch = 'master'

    server_name = 'cloudml.match.odesk.com'
    # For Apache ServerAdmin directive
    server_admin = 'nmelnik@odesk.com'
    # Apache will serve WSGI on this port. (Nginx is front-end.)
    apache_port = 80

    # For pip extra index url config
    odeskps_pypi_user = 'nmelnik@odesk.com'
    odeskps_pypi_password = 'nmelnik'

    # Once on production, this file will replace %(local_settings_file)s
    # It should be a Jinja2 template, and can make use of fabdeploy config
    # variables.
    remote_settings_lfile = 'prod_config.py.tpl'


class ProductionConf(BaseConf):
    """Settings specific to production environment."""

    #address = 'cloudml@172.27.77.205'
    address = 'cloudml@172.27.77.141'

    sudo_user = 'nmelnik'
    django_dir = ''

    # Code from this branch will be deployed.
    branch = 'master'

    server_name = 'cloudml.match.odesk.com'
    # For Apache ServerAdmin directive
    server_admin = 'nmelnik@odesk.com'
    # Apache will serve WSGI on this port. (Nginx is front-end.)
    apache_port = 80

    # For pip extra index url config
    odeskps_pypi_user = 'nmelnik@odesk.com'
    odeskps_pypi_password = 'nmelnik'

    supervisor_programs = ['celeryd', 'celerycam']

    # Once on production, this file will replace %(local_settings_file)s
    # It should be a Jinja2 template, and can make use of fabdeploy config
    # variables.
    remote_settings_lfile = 'prod_config.py.tpl'
