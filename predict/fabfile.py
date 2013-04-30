from fabdeploy import monkey
monkey.patch_all()

from fabric.api import task, prefix
from fabdeploy.api import *

setup_fabdeploy()


@task
def prod1(**kwargs):
    fabd.conf.run('production1')


@task
def prod2(**kwargs):
    fabd.conf.run('production2')


@task
def setup():

    # TODO: chef
#    users.create.run()
#    ssh.push_key.run(pub_key_file='~/.ssh/id_rsa.pub')
#   system.package_install.run(packages='python-dev python-pip liblapack-dev gfortran \
#libpq-dev python-dev')
#   pip.install.run(app='supervisor', upgrade=True)
#   pip.install.run(app='virtualenv', upgrade=True)
#    nginx.install.run() 
#    gunicorn.push_nginx_config.run()
#    nginx.restart.run()
    # TODO: end chef

    #uwsgi.install_deps.run()
    #uwsgi.install.run()


    fabd.mkdirs.run()
    supervisor.push_d_config.run()
    supervisor.push_configs.run()
    supervisor.d.run()


@task
def qdeploy():
    release.work_on.run(0)
    deploy.run()


@task
def deploy():
    fabd.mkdirs.run()
    release.create.run()
    git.init.run()
    git.push.run()

    supervisor.push_configs.run()
    flask.push_flask_config.run()
    gunicorn.push_config.run()

    virtualenv.create.run()
    with prefix('export LAPACK=/usr/lib/liblapack.so'):
        with prefix('export ATLAS=/usr/lib/libatlas.so'):
            with prefix('export BLAS=/usr/lib/libblas.so'):
                virtualenv.pip_install.run(app='numpy')
                virtualenv.pip_install.run(app='scipy')
    virtualenv.pip_install_req.run()
    virtualenv.make_relocatable.run()

    release.activate.run()
    supervisor.update.run()
    supervisor.restart_program.run(program='gunicorn')
    supervisor.restart_program.run(program='uwsgi')
