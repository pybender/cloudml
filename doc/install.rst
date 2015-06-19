=======
Install
=======

There are different ways to get cloudml installed:

  * :ref:`Install an official release <install_official_release>`. This
    is the best approach for users who want a stable version number
    and aren't concerned about running a slightly older version of
    cloudml.

  * :ref:`Install the latest development version
    <install_development_version>`. This is best for users who want the
    latest-and-greatest features and aren't afraid of running
    brand-new code.


Installing an official release
==============================

Cloudml requires:

- Python (>= 2.6)
 
Ubuntu installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install setuptools http://pypi.python.org/pypi/setuptools

2. Install pip::

    $ easy_install pip

3. Make sure that you have the following dev packages::

    $ sudo apt-get install python-dev libxml2-dev libxslt1-dev

4.1. Check that you have LAPACK, BLAS and ATLAS libraries, if not::

    $ sudo apt-get install liblapack-dev gfortran libpq-dev libevent-dev

4.2. Install Numpy::

    $ pip install -U numpy

4.3. Install SciPy::

    $ pip install scipy

or simply::

4. Install SciPy and NumPy from repositories (WARNING: can be outdated)::

    $ sudo apt-get install python-numpy python-scipy


5. Install cloudml with pip::

    $ pip install cloudml

To build cloudml from the source package download the source package from http://pypi.python.org/pypi/cloudml, unpack the sources into a directory, cd into this directory and run::

    $ python setup.py install


.. _install_development_version:

Install the latest development version
=============

git clone https://github.com/odeskdataproducts/cloudml.git


.. _testing:

Testing
=======

Testing requires having the `nose <http://somethingaboutorange.com/mrl/projects/nose/>`_ module.
To start testing cd outside of a source directory and execute::

    $ python setup.py test
