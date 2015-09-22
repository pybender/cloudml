=================
Developer's Guide
=================

------------------
Bootsrap on Ubuntu
------------------

Install system requirements:

.. code-block:: console

	$ sudo apt-get install -y build-essential git python-pip python-dev libxml2-dev libxslt1-dev liblapack-dev gfortran libpq-dev libevent-dev python-virtualenv

Clone ClowdML repo:

.. code-block:: console

	$ git clone https://github.com/odeskdataproducts/cloudml.git
	$ cd cloudml

Create virtual env:

.. code-block:: console

    $ virtualenv --no-site-packages ve
    $ source ve/bin/activate

Install numpy and scipy:

.. code-block:: console

    $ export LAPACK=/usr/lib/liblapack.so
    $ export ATLAS=/usr/lib/libatlas.so
    $ export BLAS=/usr/lib/libblas.so
    $ pip install numpy==1.7.1
    $ pip install scipy==0.12.0

Install memory-profiler:

.. code-block:: console

	$ memory-profiler==0.27

Install python requirements:

.. code-block:: console

    $ pip install -r ./requirements.txt

Create local config:

.. code-block:: console

    $ cp local_config.py.tpl local_config.py


------------------
Bootsrap on MacOS
------------------

Clone cloduml repo:

.. code-block:: console

	$ git clone https://github.com/odeskdataproducts/cloudml.git
	$ cd cloudml

Create virtual env:

.. code-block:: console

    $ virtualenv --no-site-packages ve
    $ source ve/bin/activate

Install numpy and scipy:

.. code-block:: console

    $ export LAPACK=/usr/lib/liblapack.so
    $ export ATLAS=/usr/lib/libatlas.so
    $ export BLAS=/usr/lib/libblas.so
    $ pip install numpy==1.7.1
    $ pip install scipy==0.12.0

Install memory-profiler:

.. code-block:: console

	$ memory-profiler==0.27

Ensure python version 2.7 or above is being used. By default, if python installed on the Mac is an older version 2.6, then follow this procedure:

.. code-block:: console

   $ brew install python 
   $ mv /usr/bin/python /usr/bin/python.orig
   $ ln -s /opt/local/bin/python /usr/bin/python
   $ port select --set python python27 

Ensure that pip2.7 or newer is being used. If not, undertaken the following steps:

.. code-block:: console

    $ cd /usr/local/bin/
    $ mv pip pip.orig
    $ ln -s pip2.7 pip

Ensure that easy_install version 2.7 and older is being used. If not, undertake the following steps:

.. code-block:: console

    $ cd /usr/bin/
    $ mv easy_install easy_install.orig
    $ ln -s easy_install-2.7 easy_install

Install nltk with easy install:

.. code-block:: console

	$sudo easy_install nltk==3.0.3

Install jsonpath:

.. code-block:: console

    $ Download jsonpath from http://www.ultimate.com/phil/python/download/jsonpath-0.54.tar.gz 
    $ cd ~/Downloads/jsonpath-0.54
    $ sudo python setup.py install

Install python requirements:

.. code-block:: console

    $ cd cloudml
    $ pip install -r ./requirements.txt

Create local config:

.. code-block:: console

    $ cp local_config.py.tpl local_config.py


-----------------------
Bootstrap using vagrant
-----------------------

Before using CloudML, kindly `install the latest version of Vagrant <http://docs.vagrantup.com/v2/installation/>`_. Since `VirtualBox <http://www.virtualbox.org/>`_ will be used as the provider for the obtaining the start guide, kindly install the same.

Clone cloduml repo:

.. code-block:: console

	$ git clone https://github.com/odeskdataproducts/cloudml.git

For booting the Vagrant environment. Run the following:

.. code-block:: console

	$ cd cloudml
	$ vagrant up

In approximately 20-30 minutes time, this command will complete running and a virtual machine running Ubuntu with all installed dependencies will be available.

For connecting to the machine run the following:

.. code-block:: console

	$ vagrant ssh

For running a test, kindly go to `/vagrant` directory:

.. code-block:: console

	$ cd /vagrant
	$ python setup.py test

Once finished with tweaking the machine, run `vagrant destroy` back on the host machine, and Vagrant will remove all traces of the virtual machine.

Effectively, a `vagrant suspend` saves the state of the machine in 'real time' or 'as-it-is', so that once it is resumed later, it begins running from that suspended point, instead of undertaking a full boot.


----------
Build docs
----------

For build docs, kindly install:

.. code-block:: console

    $ sudo pip install Sphinx==1.3.1

Build html doc:

.. code-block:: console

	$ cd doc
	$ make html

View doc in ./doc/_build/html directory.


-------------
Run tests
-------------

For undertaking a test run, kindly install::

	$ pip install nose coverage moto==0.3.3 mock==1.0.1

Run all tests:

.. code-block:: console

	$ python setup.py test

Run tests with coverage:

.. code-block:: console

	$ python setup.py coverage

Run single test:

.. code-block:: console

	$ nosetests cloudml.tests.trainer_tests:TrainerTestCase.test_features_not_found