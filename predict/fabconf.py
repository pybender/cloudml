import os

from fabdeploy.api import DefaultConf


DIRNAME = os.path.dirname(__file__)


class BaseConf(DefaultConf):
    """Base deployment settings."""

    django_dir = 'api'
    project_dir = 'predict'

    pip_req_path = ''
    pip_req_name = 'requirements.txt'

    supervisor_prefix = 'cloudml_'
    supervisord_config_lfile = 'supervisor/supervisord.conf'
    supervisord_config_file = ['%(supervisor_config_path)s',
                               'supervisord.conf']

    supervisor__log_path = ['%(var_path)s', 'log', 'supervisor']
    supervisor_programs = ['gunicorn']

    # Local settings file
    local_settings_file = 'config.py'


class Production1Conf(BaseConf):
    """Settings specific to production environment."""

    address = 'cloudml@172.27.85.243'
    home_path = '/webapps/cloudml'
    sudo_user = 'nmelnik'

    # Code from this branch will be deployed.
    branch = 'master'

    server_name = 'cloudml1.match.odesk.com'
    # For Nginx ServerAdmin directive
    server_admin = 'papadimitriou@odesk.com'
    gunicorn_port = 5000

    remote_settings_lfile = 'prod_config.py.tpl'
