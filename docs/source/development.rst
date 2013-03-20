Development
===========

Backend
-------

Create virtual env and install requirements::

    $ virtualenv --no-site-packages ve
    $ . ve/bin/activate
    $ pip install -r requirements.txt

Run api dev server::

    $ python runserver.py

Run celery:

    $ ./runcelery.sh


Frontend
--------

Install nodejs and nmp:

    $ sudo aptget install nodejs nmp

Init ui dev enviropment::
    
    $ cd ui
    $ ./scripts/init.sh

Run ui dev server::

    $ cd ui
    $ ./scripts/server.sh
