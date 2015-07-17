=================
Developer's guide
=================

------------------
Bootsrap on Ubuntu
------------------

Install system requirements::

	$ sudo apt-get install -y build-essential git python-pip python-dev libxml2-dev libxslt1-dev liblapack-dev gfortran libpq-dev libevent-dev python-virtualenv

Clone cloduml repo::

	$ git clone https://github.com/odeskdataproducts/cloudml.git
	$ cd cloudml

Create virtual env::

    $ virtualenv --no-site-packages ve
    $ source ve/bin/activate

Install numpy and scipy::

    $ export LAPACK=/usr/lib/liblapack.so
    $ export ATLAS=/usr/lib/libatlas.so
    $ export BLAS=/usr/lib/libblas.so
    $ pip install numpy==1.7.1
    $ pip install scipy==0.12.0

Install memory-profiler::

	$ memory-profiler==0.27

Install python requirements::

    $ pip install -r ./requirements.txt

Create local config::

    $ cp local_config.py.tpl local_config.py


------------------
Bootsrap on MacOS
------------------

Clone cloduml repo::

	$ git clone https://github.com/odeskdataproducts/cloudml.git
	$ cd cloudml

Create virtual env::

    $ virtualenv --no-site-packages ve
    $ source ve/bin/activate

Install numpy and scipy::

    $ export LAPACK=/usr/lib/liblapack.so
    $ export ATLAS=/usr/lib/libatlas.so
    $ export BLAS=/usr/lib/libblas.so
    $ pip install numpy==1.7.1
    $ pip install scipy==0.12.0

Install memory-profiler::

	$ memory-profiler==0.27

Make sure you are using python version 2.7 and above. If the default python in your mac is an older version 2.6, then do the following::

   $ brew install python 
   $ mv /usr/bin/python /usr/bin/python.orig
   $ ln -s /opt/local/bin/python /usr/bin/python
   $ port select --set python python27 

Make sure you are using pip2.7 or newer. If not, do the following::

    $ cd /usr/local/bin/
    $ mv pip pip.orig
    $ ln -s pip2.7 pip

Make sure you are using easy_install version 2.7 and older. Else do the following::

    $ cd /usr/bin/
    $ mv easy_install easy_install.orig
    $ ln -s easy_install-2.7 easy_install

Install nltk with easy install::

	$sudo easy_install nltk==3.0.3

Install jsonpath::

    $ Download jsonpath from http://www.ultimate.com/phil/python/download/jsonpath-0.54.tar.gz 
    $ cd ~/Downloads/jsonpath-0.54
    $ sudo python setup.py install

Install python requirements::

    $ cd cloudml
    $ pip install -r ./requirements.txt

Create local config::

    $ cp local_config.py.tpl local_config.py


-----------------------
Bootstrap using vagrant
-----------------------

Before diving into cloudml, please `install the latest version of Vagrant <http://docs.vagrantup.com/v2/installation/>`_. And because we'll be using `VirtualBox <http://www.virtualbox.org/>`_ as our provider for the getting started guide, please install that as well.

Clone cloduml repo::

	$ git clone https://github.com/odeskdataproducts/cloudml.git

For boot your Vagrant environment. Run the following::

	$ cd cloudml
	$ vagrant up

In 20-30 minutes, this command will finish and you'll have a virtual machine running Ubuntu with installed all dependencies.

For connect to machine run::

	$ vagrant ssh

For run test please go to `/vagrant` directory::

	$ cd /vagrant
	$ python setup.py test

When you're done fiddling around with the machine, run `vagrant destroy` back on your host machine, and Vagrant will remove all traces of the virtual machine.

A `vagrant suspend` effectively saves the exact point-in-time state of the machine, so that when you resume it later, it begins running immediately from that point, rather than doing a full boot.


----------
Build docs
----------

For build docs please install::

    $ sudo pip install Sphinx==1.3.1

Build html doc::

	$ cd doc
	$ make html

View doc in ./doc/_build/html directory.


-------------
Run tests
-------------

For run test please install::

	$ pip install nose coverage moto==0.3.3 mock==1.0.1

Run all tests::

	$ python setup.py test

Run tests with coverage::

	$ python setup.py coverage

Run single test::

	$ nosetests cloudml.tests.trainer_tests:TrainerTestCase.test_features_not_found