try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from fabdeploy import monkey
monkey.patch_all()
from fabric.api import task, env, settings, local, run
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
    #apache.install.run()
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

    # apache.wsgi_push.run()
    # apache.push_config.run()
    # apache.push_nginx_config.run()
    # nginx.restart.run()
    # apache.graceful.run()

   # pip.push_config.run()

    #supervisor.push_configs.run()
    #supervisor.d.run()

@task
def deploy():
    fabd.mkdirs.run()
    
    version.create.run()
    git.init.run()
    git.push.run()
    
    #supervisor.push_configs.run()
    #apache.wsgi_push.run()

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

    #gunicorn.reload_with_supervisor.run()

    #apache.wsgi_touch.run()