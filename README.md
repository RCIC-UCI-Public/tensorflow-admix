## tensorflow-admix
Building tensorflow. 

See generic step instructions to compile from source https://www.tensorflow.org/install/source. 
Details of specific build instrucionts is in `yamlspecs/README-build-tensorflow*`

### Download source

Download a source release https://github.com/tensorflow/tensorflow/archive/v<version>.tar.gz
This is used with bazel build to create python whl package.

### Prerequisites

Build/install all the prerequisites.

1. **Python prerequisites**

   Using the latest python PYVERSION.
   Python packages prerequisites are specified in tensorflow source
   in tensorflow-VERSION/tensorflow/tools/pip_package/setup.py

   Also check tensorflow-VERSION/tensorflow/tools/ci_build/release/requirements_common.txt

   The packages and their specific versions are listed in `packages.yaml`, `set*yaml`  and `versions*yaml`

   **Note:** gast 0.3+ no longer has attribute `Num`, which is required by tensroflow. The built
   tensorflow package fails on import with `AttributeError: module 'gast' has no attribute 'Num'`
   Use older gast verison 0.2.2 for tensorflow 2.0.0.

1. **Bazel**

   Install bazel from source. See https://docs.bazel.build/versions/master/install.html for details.
   Make sure to use a supported bazel version: any version between _TF_MIN_BAZEL_VERSION and 
   _TF_MAX_BAZEL_VERSION as specified in tensorflow/configure.py. 
   
   When building bazel verify java version. Specific versions of bazel require specific versions of java
   This is handled via `versions*yaml` files.

1. **GPU support**  

   See https://www.tensorflow.org/install/gpu for details

   **NOTE:** most packaghes are related by verison of CUDA and cuDNN. See specific versions
   in `versions.yaml`

   The following NVIDIA® software must be installed:
   - CUDA® Toolkit —TensorFlow supports CUDA 10.0 
     - NVIDIA® GPU drivers CUDA 10.0 requires 410.x or higher.
     - CUPTI ships with the CUDA Toolkit.
   - cuDNN SDK (>= 7.4.1). Download RPMs from NVIDIA for CUDA 10, current version X=7.6.4.38-1.cuda10.1
     - libcudnn7-X
     - libcudnn7-devel-X
     - libcudnn7-doc-X
1. TensorRT to improve latency and throughput for inference on some models.
   Download source from NVIDIA Developer (have to login), specific versions:
   for CUDA 10 and cuDNN 7  use 6.0 GA (Generic Availability)
       https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/6.0/GA_6.0.1.5/tars
   for CUDA 11.4 and cuDNN 8  use 8.2 GA (Generic Availability) Update 2
       https://developer.nvidia.com/nvidia-tensorrt-8x-download
   for CUDA 11.7 and cudNN 8 use 8.4 GA (Generic Availability) Update 1 
       https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.2/tars/tensorrt-8.4.2.4.linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
       While the file name has cuda11.6 the web page claims it works with cuda 11.7
       See https://developer.nvidia.com/nvidia-tensorrt-8x-download

   Compile python whl files and install packaged  contents. Requires cuDNN
   Install instructions https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-tar

1. **Optional ComputeCPP**
  
   Moved ComptuteCPP to parallel-admix 2020-01-24
  
   If want to run the OpenCL™ version of TensorFlow™ using ComputeCpp, a SYCL™ implementation
   need to install ComputeCpp.  Register at codeplay in order to download the source distro.
   Current version is ComputeCpp v1.2.0: Changes to Work-item Mapping Optimization
   See Guides:
   - https://developer.codeplay.com/products/computecpp/ce/guides
   - https://developer.codeplay.com/products/computecpp/ce/guides?render=pdf
   
   When configuring tensroflow build the following question determines if OpenCL support will be enabled: 
   - Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 
   
   Generic instruction for building tensorflow with OpenCL support are 
   https://developer.codeplay.com/products/computecpp/ce/guides/tensorflow-guide/tensorflow-generic-setup

   **NOTE:** installed ComputeCPP but NOT USING with tensorflow build

### Patch tensorflow source.

The Tensorflow v.2.0.0 had a few bugs that require fixes.

1. Bazel build fails with `cannot convert ‘std::nullptr_t’ to ‘Py_ssize_t {aka long int}’ in initialization`
   See https://github.com/tensorflow/tensorflow/issues/33543.
   In Python 3.8, the reserved `tp_print` slot was changed from a function pointer to a number, 
   `Py_ssize_t tp_vectorcall_offset`. Search for `tp_print` in the source files and change nullptr to 0 (or NULL):
   ```txt
   from:
   nullptr, /* tp_print */ 
   to:
   NULL, /* tp_print */ 
   ```
   Create a patch for files:
   - tensorflow/python/eager/pywrap_tensor.cc 
   - tensorflow/python/eager/pywrap_tfe_src.cc
   - tensorflow/python/lib/core/bfloat16.cc
   - tensorflow/python/lib/core/ndarray_tensor_bridge.cc

1. TensorFlow on Python 3.8 logger issue #33953
   See https://github.com/tensorflow/tensorflow/pull/33953/commits/ea3063c929c69f738bf65bc99dad1159803e772f
   Create a patch for file:
   - tensorflow.orig/python/platform/tf_logging.py

Create a single patch file for all 5 patches and apply while in top extraceted source dir:
```bash
patch  -p0 < ../tensorflow-v.2.0.0-python.3.8.0.patch
```

### Configure the build

Load modules to setup environment and configure system build via the `./configure` at the root of the source tree. 
This script prompts for the location of TensorFlow dependencies and asks for additional build configuration options.
See yamlspecs/README-build-tensorflow-<VERSION> for a complete set of questions and commands.

**NOTE:** need git that understands `-C` flag (git provided by foundation module is 2.23).

**NOTE:** For compilation optimization flags, the default (`-march=native`) optimizes the generated code 
for the machine's CPU type where the build is run. For building TensorFlow for a different CPU type, 
need a more specific optimization flag. See the GCC manual for examples:
https://gcc.gnu.org/onlinedocs/gcc-4.5.3/gcc/i386-and-x86_002d64-Options.html

After running `./configure` a configuration file `.tf_configure.bazelrc` is created in the
current directory and it will be used by bazel build.

### Build the bazel package

#### Run bazel build command
Specify memory resources as an option so that the build does not run out of memory.
```bash
nohup bazel build --local_ram_resources=4096 \
                  --verbose_failures //tensorflow/tools/pip_package:build_pip_package > ../build-out &
```

Bazel build on a VM with 2 cores and 8Gb memory takes ~14hrs.

#### Run build_pip_package command
The bazel build command creates an executable `build_pip_package`, this is the program that builds the 
needed `.whl` package. Run the executable as:
```bash
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

This command generates a `.whl` file  in `/tmp/tensorflow_pkg/`.
The filename of the generated `.whl` file depends on the TensorFlow version and on 
the local platform, for example, `tensorflow-2.0.0-cp38-cp38-linux_x86_64.whl`

Copy the resulting `.whl` file to `tensorflow-admix/sources/`

### Install the package

Use `yamlspecs/tensorflow.yaml` to generate RPM from the `.whl` file.

## Keras

A new version (2.8.0) install from source fails hence using whl file. 
To build from source need bazel AND already built tensorflow.  Tried commands:

```bash
cd keras-2.8.0
module load  bazel/VERSION
module load tensorflow/2.8.0
bazel build //keras/tools/pip_package:build_pip_package
# next command would be 
# ./bazel-bin/keras/tools/pip_package/build_pip_package /tmp/keras_pkg
```
Failed with

``` txt
[178 / 187] Compiling src/google/protobuf/descriptor.cc [for host]; 9s local ... (7 actions running)
ERROR: /export/repositories/tensorflow-admix/yamlspecs/keras-2.8.0/keras/api/BUILD:143:19: Executing genrule //keras/api:keras_python_api_gen failed: (Exit 1): bash failed: error executing command /bin/bash -c ... (remaining 1 argument(s) ski
pped)
Traceback (most recent call last):
  File "/root/.cache/bazel/_bazel_root/6473a4b21803cec7d4edad8916105edd/execroot/org_keras/bazel-out/k8-opt-exec-2B5CBBC6/bin/keras/api/create_keras_api_1_keras_python_api_gen.runfiles/org_keras/keras/api/create_python_api_wrapper.py", line 2
6, in <module>
    import keras  # pylint: disable=unused-import
  File "/root/.cache/bazel/_bazel_root/6473a4b21803cec7d4edad8916105edd/execroot/org_keras/bazel-out/k8-opt-exec-2B5CBBC6/bin/keras/api/create_keras_api_1_keras_python_api_gen.runfiles/org_keras/keras/__init__.py", line 21, in <module>
    from tensorflow.python import tf2
ModuleNotFoundError: No module named 'tensorflow'
Target //keras/tools/pip_package:build_pip_package failed to build
Use --verbose_failures to see the command lines of failed build steps.
ERROR: /export/repositories/tensorflow-admix/yamlspecs/keras-2.8.0/keras/tools/pip_package/BUILD:37:10 Middleman _middlemen/keras_Stools_Spip_Upackage_Sbuild_Upip_Upackage-runfiles failed: (Exit 1): bash failed: error executing command /bin/b
ash -c ... (remaining 1 argument(s) skipped)
INFO: Elapsed time: 59.379s, Critical Path: 13.59s
INFO: 231 processes: 20 internal, 211 local.
FAILED: Build did NOT complete successfully
FAILED: Build did NOT complete successfully
```
On a command line per loaded modules can start python and do import `from tensorflow.python import tf2`

