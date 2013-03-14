try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from fabdeploy import monkey
monkey.patch_all()
from fabric.api import task, env, settings, local, run, sudo
from fabric.contrib import files
from fabdeploy.api import *


setup_fabdeploy()


@task
def here(**kwargs):
    fabd.conf.run('here')

@task
def staging(**kwargs):
    fabd.conf.run('staging')

@task
def install():
    users.create.run()
    ssh.push_key.run(pub_key_file='~/.ssh/id_rsa.pub')

    fabd.mkdirs.run()

    #system.setup_backports.run()
    #system.install_common_software.run()

    #rabbitmq.install()
    #nginx.install.run()
    apache.install.run()
    #postgres.install.run()

    for app in ['supervisor']:
        pip.install.run(app=app)

@task
def setup():
    fabd.mkdirs.run()
#
#    with settings(warn_only=True):
#        postgres.create_user.run()
#        postgres.create_db.run()
#        postgres.grant.run()
#
    # with settings(warn_only=True):
    #     rabbitmq.add_user.run()
    #     rabbitmq.add_vhost.run()
    # rabbitmq.set_permissions.run()

    #apache.wsgi_push.run()
    push_apache_config.run()
    apache.graceful.run()

   # pip.push_config.run()

    #supervisor.push_configs.run()
    #supervisor.d.run()

@task
def qdeploy():
    version.work_on.run(0)
    deploy.run()

@task
def deploy():
    fabd.mkdirs.run()
    
    version.create.run()
    git.init.run()
    git.push.run()
    
    #supervisor.push_configs.run()
    apache.wsgi_push.run()

    virtualenv.create.run()
    virtualenv.pip_install_req.run()
    virtualenv.make_relocatable.run()

    # django.syncdb.run()
    # django.migrate.run()

    
    #run('%(env_path)s/bin/python %(django_path)s/manage.py '
    #     'collectstatic --noinput;' % env.conf)

    version.activate.run()

    #supervisor.update.run()
    # supervisor.restart_program.run(program='celeryd')
    # supervisor.restart_program.run(program='celerycam')
    # supervisor.restart_program.run(program='celerybeat')
    # supervisor.reload.run()

    apache.wsgi_touch.run()

from fabdeploy.apache import PushConfig as StockPushApacheConfig
from fabdeploy.utils import upload_config_template

class PushApacheConfig(StockPushApacheConfig):
    def do(self):
        # Instead of appending Listen directive, just upload the whole
        # ports.conf template to make sure it has the right contents.
        # We assume it doesn't contain anything useful anyway.
        upload_config_template(
            'apache_ports.config',
            self.conf.ports_filepath,
            context=self.conf,
            use_sudo=True)
        upload_config_template(
            'apache.config',
            self.conf.config_filepath,
            context=self.conf,
            use_sudo=True)
        sudo('a2ensite %(instance_name)s' % self.conf)

push_apache_config = PushApacheConfig()