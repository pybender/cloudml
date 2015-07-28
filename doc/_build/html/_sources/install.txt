============
Installation
============

There are different ways to get cloudml installed:

  * :ref:`Install an official release <install_official_release>`. This
    is the best approach for users who want a stable version number
    and aren't concerned about running a slightly older version of
    cloudml.

  * :ref:`Install the latest development version
    <install_development_version>`. This is best for users who want the
    latest-and-greatest features and aren't afraid of running
    brand-new code.


.. _install_official_release:

Installing an official release
==============================

Cloudml requires:

- Python (>= 2.6)
 
Ubuntu 
------

1. Install system requirements::

    $ sudo apt-get install -y build-essential git python-pip python-dev libxml2-dev libxslt1-dev liblapack-dev gfortran libpq-dev libevent-dev

2. Install numpy and scipy::

    $ export LAPACK=/usr/lib/liblapack.so
    $ export ATLAS=/usr/lib/libatlas.so
    $ export BLAS=/usr/lib/libblas.so
    $ pip install numpy==1.7.1
    $ pip install scipy==0.12.0


3. Install cloudml with pip::

    $ pip install cloudml

To build cloudml from the source package download the source package from http://pypi.python.org/pypi/cloudml, unpack the sources into a directory, cd into this directory and run::

    $ python setup.py install


Mac OSX
-------

Windows
-------


.. _install_development_version:

Install the latest development version
======================================

git clone https://github.com/odeskdataproducts/cloudml.git


.. _testing:

Testing
=======

Testing requires having the `nose`, `coverage`, `moto`, `mock` libraries::
    
    $ pip install nose coverage moto==0.3.3 mock==1.0.1

To start testing cd to a source directory and execute::

    $ python setup.py test
