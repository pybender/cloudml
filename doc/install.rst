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
 
Install cloudml with pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is usually the fastest way to install or upgrade to the latest stable
release::

    pip install cloudml


From source package
~~~~~~~~~~~~~~~~~~~

Download the source package from http://pypi.python.org/pypi/cloudml/
, unpack the sources and cd into the source directory.

This packages uses distutils, which is the default way of installing
python modules. The install command is::

    python setup.py install


.. _install_development_version:

Install the latest development version
=============

git clone https://github.com/odeskdataproducts/cloudml.git


.. _testing:

Testing
=======

Testing requires having the `nose
<http://somethingaboutorange.com/mrl/projects/nose/>`_ library. After
installation, the package can be tested by executing *from outside* the
source directory::

    $ python setup.py test
