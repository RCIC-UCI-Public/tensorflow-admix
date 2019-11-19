## tensorflow-admix
Building tensorflow. 

See generic step instructions to compile from source https://www.tensorflow.org/install/source. 
Details of specific build instrucionts is in `yamlspecs/README-build-tensorflow`

### Download source

Download a release https://github.com/tensorflow/tensorflow/releases/tag/v2.0.0

### Prerequisites

Build/install all the prerequisites.

1. **Python prerequisites**

   Using the latest python 3.8.0.
   Python packages prerequisites are specified in tensorflow source
   in tensorflow-VERSION/tensorflow/tools/pip_package/setup.py
   For tensorflow v 2.0.0: 
   - absl-py >= 0.7.0
   - astor >= 0.6.0
   - backports.weakref >= 1.0rc1
   - gast == 0.2.2
   - google_pasta >= 0.1.8
   - keras_applications >= 1.0.8
   - keras_preprocessing >= 1.1.0
   - numpy >= 1.16.0 < 2.0
   - opt_einsum >= 2.3.2
   - protobuf >= 3.8.0
   - tensorboard >= 2.0.0 < 2.1.0
   - tensorflow_estimator >= 2.0.0 < 2.1.0
   - termcolor >= 1.1.0
   - wrapt >= 1.11.1
   - wheel >= 0.26
   - six >= 1.12.0

   The packages and their specific versions are listed in `packages.yaml` and `versions.yaml`

   **Note:** gast 0.3+ no longer has attribute `Num`, which is required by tensroflow. The built
   tensorflow package fails on import with `AttributeError: module 'gast' has no attribute 'Num'`
   Use older gast verison 0.2.2. 

1. **Bazel**

   Install bazel from source. See https://docs.bazel.build/versions/master/install.html for details.
   Make sure to use a supported bazel version: any version between _TF_MIN_BAZEL_VERSION and 
   _TF_MAX_BAZEL_VERSION as specified in tensorflow/configure.py. Currently supported is 0.26.1

1. **GPU support**  

   See https://www.tensorflow.org/install/gpu for details
   **NOTE:** most packaghes are related by verison of CUDA and cuDNN. See specific versions
   in `versions.yaml`

   The following NVIDIA® software must be installed:
   - CUDA® Toolkit —TensorFlow supports CUDA 10.0 
     - NVIDIA® GPU drivers —CUDA 10.0 requires 410.x or higher.
     - CUPTI ships with the CUDA Toolkit.
   - cuDNN SDK (>= 7.4.1). Download RPMs from NVIDIA for CUDA 10, current version X=7.6.4.38-1.cuda10.1
     - libcudnn7-X
     - libcudnn7-devel-X
     - libcudnn7-doc-X
   - Optional TensorRT to improve latency and throughput for inference on some models.
     Download source from NVIDIA Developer, specific version for CUDA 10 and cuDNN 7.6. Compile python
     whl files and install packaged  contents. Requires cuDNN
     Install instructions https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html#installing-tar

1. **Optional ComputeCPP**
  
   If want to run the OpenCL™ version of TensorFlow™ using ComputeCpp, a SYCL™ implementation
   need to install ComputeCpp.  Register at codeplay in order to download the source distro.
   Current version is ComputeCpp v1.1.6: Changes to Work-item Mapping Optimization
   See https://codeplay.com/portal/11-18-19-computecpp-v1-1-6-changes-to-work-item-mapping-optimization
   Guides:
   - https://developer.codeplay.com/products/computecpp/ce/guides
   - https://developer.codeplay.com/products/computecpp/ce/guides?render=pdf
   
   When configuring tensroflow build the following question determines if OpenCL support will be enabled: 
   - Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: 
   
   Generic instruction for building tensorflow with OpenCL support are 
   https://developer.codeplay.com/products/computecpp/ce/guides/tensorflow-guide/tensorflow-generic-setup

   **NOTE:** installed ComputeCPP but NOT USING with tensorflow build

### Patch tensorflow source.

The Tensorflwo v.2.0.0 had a few bugs that require fixes.

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

Create a single patch file for all 5 patches and apply whil in top extraceted source dir:
```bash
patch  -p0 < ../tensorflow-v.2.0.0-python.3.8.0.patch
```

### Configure the build

Load modules to setup environment and configure system build via the `./configure` at the root of the source tree. 
This script prompts for the location of TensorFlow dependencies and asks for additional build configuration options.
```bash
module load bazel/0.26.1 
module load cuda/10.1.243
module load computecpp/1.1.6 
module load tensorRT/6.0.1.5 
module load foundation
./configure
```

**NOTE:** need git that understands `-C` flag (provided by foundation module is 2.23).
See yamlspecs/README-build-tensorflow for a complete set of questions.

**NOTE:** For compilation optimization flags, the default (`-march=native`) optimizes the generated code 
for the machine's CPU type where the build is run. For building TensorFlow for a different CPU type, 
need a more specific optimization flag. See the GCC manual for examples:
https://gcc.gnu.org/onlinedocs/gcc-4.5.3/gcc/i386-and-x86_002d64-Options.html

Option used for current build is `-march=core-avx2`

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
`.whl` package. Run the executable in the /tmp/tensorflow_pkg directory to build from a release branch:
```bash
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
```

This command generates a `.whl` file  in `/tmp/tensorflow_pkg/`.
The filename of the generated .whl file depends on the TensorFlow version and the local platform,
for example, `tensorflow-2.0.0-cp38-cp38-linux_x86_64.whl`

Copy the resulting `.whl` file to `tensorflow-admix/sources`

### Install the package

Use `yamlspecs/tensorflow.yaml` to generate RPM from the `.whl` file.
