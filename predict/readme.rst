Cloudml-predict
===============

Api
---

Methods:

* GET /cloudml/import/handler/ - Get list of allowed import handlers
* GET /cloudml/models/ - Get list of allowed models
* POST /cloudml/model/:ModelName/:HandlerName/predict - Predict


Development
-----------

Create virtual env and install requirements::

    $ virtualenv --no-site-packages ve
    $ . ve/bin/activate
    $ pip install -r requirements.txt

Run dev server::

    $ python manage.py runserver

Run tests::

    $ python manage.py test


Deploy
------

Create virtual env and install deploy requirements::

    $ virtualenv --no-site-packages ve
    $ . ve/bin/activate
    $ pip install -r deploy_requirements.txt

Copy fabconf file::
    
    $ cp fabconf.py.def fabconf.py

Change sudo_user property to your user.

For setup project to instance run::

    $ fab prod1 setup

For deploy new release(will create new env) run::

    $ fab prod1 deploy

For deploy to current release(withouth create new release and env) run::

    $ fab prod1 qdeploy

Managed supervisor::

    $ fab prod1 supervisor.ctl

After that we will can put model's and import_handler's files to::

    /webapps/cloudml/shared/var/models
    /webapps/cloudml/shared/var/import_handlers
(I can create fab command for simple upload model. Is it necessary?)

Reload gunicorn after upload files::
    
    $ fab prod1 gunicorn.reload_with_supervisor


