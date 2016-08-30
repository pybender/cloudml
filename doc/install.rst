============
Installation
============

There are different methods to install CloudML:

  * :ref:`Installing an official release <install_official_release>`. This
    is the preferred option for users who require a stable version number
    and are not concerned about running a slightly older version of
    CloudMl.

  * :ref:`Installing the latest development version
    <install_development_version>`. This is the preferred installation option for users who require new and the very latest features, and for those who do not mind running "brand-new" code.


.. _install_official_release:

Installing an official release
==============================

CloudML requires:

- Python (>= 2.6)
 
Ubuntu 
------

1. Install system requirements:

.. code-block:: console

    $ sudo apt-get install -y build-essential git python-pip python-dev libxml2-dev libxslt1-dev liblapack-dev gfortran libpq-dev libevent-dev

2. Install numpy and scipy:

.. code-block:: console

    $ export LAPACK=/usr/lib/liblapack.so
    $ export ATLAS=/usr/lib/libatlas.so
    $ export BLAS=/usr/lib/libblas.so
    $ pip install numpy==1.10.04


3. Install CloudML with pip:

.. code-block:: console

    $ pip install cloudml

In order to build CloudML from the source package, download the source package from http://pypi.python.org/pypi/cloudml, unpack the sources into a directory, cd into this directory and run:

.. code-block:: console

    $ python setup.py install


Mac OSX
-------

Install CloudML with pip:

.. code-block:: console

    $ pip install cloudml


Windows
-------

This instruction assumes that you do not have Python already installed on your machine.

1. Install `Python <http://www.python.org>`_.
2. Install `NumPy 1.7.1 and SciPy 0.12.0 <http://www.scipy.org/install.html#individual-packages>`_.
3. Install `psycopg2 <http://www.stickpeople.com/projects/python/win-psycopg/index.2.4.6.html>`_.
4. Install `NLTK 2.0.4 <https://pypi.python.org/pypi/nltk>`_.
5. Install `cloudml <http://pypi.python.org/pypi/cloudml>`_.

.. note:: Here we suggest to use pre-compiled packages, since it works fine and makes the installation easier.

.. _install_development_version:

Install the latest development version
======================================

.. code-block:: console

    $ git clone https://github.com/odeskdataproducts/cloudml.git


.. _testing:

Testing
=======

Testing requires `nose`, `coverage`, `moto`, `mock` libraries:

.. code-block:: console

    $ pip install nose coverage moto==0.3.3 mock==1.0.1

To start testing, cd to a source directory and execute:

.. code-block:: console

    $ python setup.py test
