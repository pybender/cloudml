try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from fabdeploy import monkey
monkey.patch_all()
import os
import posixpath
from fabric.api import task, env, settings, local, run, sudo, prefix
from fabric.contrib import files
from fabdeploy.api import *


setup_fabdeploy()


from contextlib import contextmanager

@contextmanager
def shell_env(**env_vars):
    orig_shell = env['shell']
    env_vars_str = ' '.join('{0}={1}'.format(key, value)
                           for key, value in env_vars.items())
    env['shell']='{0} {1}'.format(env_vars_str, orig_shell)
    yield
    env['shell']= orig_shell



@task
def here(**kwargs):
    fabd.conf.run('here')

@task
def staging(**kwargs):
    fabd.conf.run('staging')

@task
def prod(**kwargs):
    fabd.conf.run('production')


@task
def dev(**kwargs):
    fabd.conf.run('dev')


@task
def install():
    users.create.run()
    ssh.push_key.run(pub_key_file='~/.ssh/id_rsa.pub')

    fabd.mkdirs.run()

    rabbitmq.install()
    apache.install.run()
    #postgres.install.run()

    for app in ['supervisor']:
        pip.install.run(app=app)

    pip.install.run(app='virtualenv', upgrade=True)
    aptitude_install.run(packages='liblapack-dev gfortran')


@task
def push_key():
    ssh.push_key.run(pub_key_file='~/.ssh/id_rsa.pub')

@task
def setup():
    fabd.mkdirs.run()

    apache.wsgi_push.run()
    apache.push_config.run(update_ports=False)
    apache.graceful.run()  
    supervisor.push_init_config.run()
    supervisor.push_configs.run()
    supervisor.d.run()

    # pip.push_config.run()
    # with settings(warn_only=True):
    #     postgres.create_user.run()
    #     postgres.create_db.run()
    #     postgres.grant.run()
    with settings(warn_only=True):
        rabbitmq.add_user.run()
        rabbitmq.add_vhost.run()
    rabbitmq.set_permissions.run()

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

    supervisor.push_configs.run()
    apache.wsgi_push.run()
    push_flask_config.run()

    virtualenv.create.run()
    with shell_env(LAPACK='/usr/lib/liblapack.so',
         ATLAS='/usr/lib/libatlas.so', BLAS='/usr/lib/libblas.so'):
        virtualenv.pip_install_req.run()
    virtualenv.make_relocatable.run()

    # django.syncdb.run()
    # django.migrate.run()

    
    #run('cd %(project_path)s/ui; ./scripts/production.sh')
    run('%(env_path)s/bin/python %(project_path)s/manage.py '
         'createdb;' % env.conf)

    version.activate.run()

    supervisor.update.run()
    supervisor.restart_program.run(program='celeryd')
    supervisor.restart_program.run(program='celerycam')
    # supervisor.restart_program.run(program='celerybeat')
    #supervisor.reload.run()

    apache.wsgi_touch.run()

from fabdeploy.apache import PushConfig as StockPushApacheConfig
from fabdeploy.utils import upload_config_template

class PushAnjularConfig(Task):
    @conf
    def from_file(self):
        return os.path.join(
            self.conf.django_dir, self.conf.remote_anjsettings_lfile)

    @conf
    def to_file(self):
        return posixpath.join(
            self.conf.django_path, self.conf.local_anjsettings_file)

    def do(self):
        files.upload_template(
            self.conf.from_file,
            self.conf.to_file,
            context=self.conf,
            use_jinja=True)

push_anj_config = PushAnjularConfig()

class PushFlaskConfig(Task):
    @conf
    def from_file(self):
        return os.path.join(
            self.conf.django_dir, self.conf.remote_settings_lfile)

    @conf
    def to_file(self):
        return posixpath.join(
            self.conf.django_path, self.conf.local_settings_file)

    def do(self):
        files.upload_template(
            self.conf.from_file,
            self.conf.to_file,
            context=self.conf,
            use_jinja=True)

push_flask_config = PushFlaskConfig()

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