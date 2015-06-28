=======
Install
=======

There are two ways to get CloudML installed:

  * `Install the official release <install_official_release>`_. This
    is the best approach for those who want the latest stable version of CloudML.

  * :ref:`Install the latest development version
    <install_development_version>`. This is the best solution for users who want the
    latest-and-greatest features and aren't afraid of running brand-new code.

.. _install_official_release:

Installing an official release
==============================



CloudML requires:

- Python (>= 2.6)

 .. note:: You have to have **Psycopg2 2.4.6**, **Numpy 1.7.1**, and **SciPy 0.12.0** pre-installed (for instructions see below).

Linux installation
~~~~~~~~~~~~~~~~~~~

1. Install `setuptools <http://pypi.python.org/pypi/setuptools>`_

2. Install pip::

    $ easy_install pip

3. Make sure that you have the required dev packages::

    $ sudo apt-get install python-dev libxml2-dev libxslt1-dev

4.1. Check that you have LAPACK, BLAS and ATLAS libraries, if not::

    $ sudo apt-get install liblapack-dev gfortran libpq-dev libevent-dev

4.2. Install Numpy::

    $ pip install -U numpy

4.3. Install SciPy::

    $ pip install scipy

or simply::

4. Install SciPy and NumPy from repositories (WARNING: the packages can be outdated)::

    $ sudo apt-get install python-numpy python-scipy


5. Install cloudml with pip::

    $ pip install cloudml

To build cloudml from the source package download the source package from http://pypi.python.org/pypi/cloudml, unpack the sources into a directory, cd into this directory and run::

    $ python setup.py install


Windows installation
~~~~~~~~~~~~~~~~~~~~
This instruction assumes that you do not have Python already installed on your machine.

1. Install `Python <http://www.python.org>`_.
2. Install `NumPy 1.7.1 and SciPy 0.12.0 <http://www.scipy.org/install.html#individual-packages>`_.
3. Install `psycopg2 <http://www.stickpeople.com/projects/python/win-psycopg/index.2.4.6.html>`_.
4. Install `NLTK 2.0.4 <https://pypi.python.org/pypi/nltk>`_.
5. Install `cloudml <http://pypi.python.org/pypi/cloudml>`_.

.. note:: Here we suggest to use pre-compiled packages, since it works fine and makes the installation easier. If you want to compile all the packages yourself, you may want to check the following links: `1 <http://blog.ionelmc.ro/2014/12/21/compiling-python-extensions-on-windows/>`_, `2 <http://docs.scipy.org/doc/numpy/user/install.html#building-from-source>`_.

MacOS installation
~~~~~~~~~~~~~~~~~~


.. _install_development_version:

Install the latest development version
======================================

To get the latest version of the source code run::

    git clone https://github.com/odeskdataproducts/cloudml.git

Follow the appropriate instruction from the previous section. But after the penultimate step cd to the directory with the source code and run::

    python setup.py install

.. _testing:

Testing
=======

Testing requires having the `nose <http://somethingaboutorange.com/mrl/projects/nose/>`_ module installed.
To start testing cd to the source directory and execute::

    $ python setup.py test


