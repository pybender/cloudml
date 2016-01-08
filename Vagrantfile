# -*- mode: ruby -*-
# vi: set ft=ruby :

Vagrant.configure(2) do |config|

  config.vm.box = "hashicorp/precise32"

  config.vm.provider :virtualbox do |vb|
    vb.customize ["modifyvm", :id, "--natdnshostresolver1", "on"]
  end

  config.vm.provision "shell", inline: <<-SHELL
     sudo apt-get update
     sudo apt-get install -y build-essential git python-pip python-dev libxml2-dev libxslt1-dev liblapack-dev gfortran libpq-dev libevent-dev rabbitmq-server

     sudo rabbitmqctl add_user cloudml cloudml
     sudo rabbitmqctl add_vhost cloudml
     sudo rabbitmqctl set_permissions cloudml cloudml ".*" ".*" ".*"


     sudo apt-get install postgresql
     sudo -u postgres createuser -D -A -P cloudml
     sudo -u postgres createdb -O cloudml cloudml


     export LAPACK=/usr/lib/liblapack.so
     export ATLAS=/usr/lib/libatlas.so
     export BLAS=/usr/lib/libblas.so
     sudo easy_install pip==1.1
     sudo pip install -U numpy==1.7.1
     sudo pip install scipy==0.12.0
     sudo pip install memory-profiler==0.27
     sudo pip install Sphinx==1.3.1
     sudo pip install nose coverage moto==0.3.3 mock==1.0.1
     cd /vagrant
     pip install -r ./requirements.txt
  SHELL
end
