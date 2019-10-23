#!/bin/bash

# track of built steps. 
# will need to be run via Makefile as a separate
# target or directly via shell script. Takes too long ~14hrs
# to compile build_pip_package

# Checkout tensorflow from git and switch to last branch
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
git describe --tags
git checkout r2.0
git branch
 
# load needed odules
module load bazel/0.26.1 
module load cuda/10.1.243_418.87.00 
module load computecpp/1.1.4 
module load tensorRT/6.0.1.5 

# these are executed from tensorflow/ (git repo)
# bazel build takes forever
nohup bazel build --local_ram_resources=4096 //tensorflow/tools/pip_package:build_pip_package >  ../build-out &

file bazel-bin/tensorflow/tools/pip_package/build_pip_package
ll bazel-bin/tensorflow/tools/pip_package/build_pip_package

# build whl file
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow.pkg

#  copy resulting wheel to sources
cp /tmp/tensorflow.pkg/tensorflow-2.0.0-cp36-cp36m-linux_x86_64.whl  ../sources/

# use yaml file to create RPM
make tensorflow.pkg


