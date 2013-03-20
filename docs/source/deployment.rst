Deployment
==========

We are going to deploy using:

- Apache
- Fabdeploy
- Virtualenv
- Supervisor

Create virtual env::

    $ virtualenv --no-site-packages ve
    $ . ve/bin/activate
    $ pip install -r requirements.txt

Create settings::

    $ cp fabconf.py.def fabconf.py

Read fabfile.py tasks to be aware of changes that will be made to your system.

Install packages, create user::

    $ fab staging install

Setup software::

    $ fab staging setup

Deploy::

    Commit changes to staging branch
    $ fab staging deploy


For first starting supervisor please run::

    $ fab staging supervisor.d

Run supervisorctl::

    $ fab staging supervisor.ctl

Get list of available tasks::

    $ fab -l