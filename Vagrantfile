Vagrant.configure("2") do |config|
	config.vm.box = "ubuntu/xenial64"
	config.vm.provider "virtualbox" do |v|
		v.memory = 1024
		v.cpus = 1
	config.vm.network "forwarded_port", guest: 8888, host: 3000
	end
end
