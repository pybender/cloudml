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



