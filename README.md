# VSI PJRT Plugin

This is a PJRT client implementation for the VeriSilicon NPU/GPU platform as a dynamic plugin.

## Building

Currently, the dynamic PJRT plugin registration API is not yet available in release version of PyTorch/XLA (See [pytorch/xla#6242](https://github.com/pytorch/xla/issues/6242)). To use our PJRT plugin for PyTorch, the user needs to build PyTorch/XLA from source of the master branch.

### Install PyTorch nightly build from pip

```shell
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
```

Although we installed the pre-built PyTorch package from pip, in order to build the PyTorch/XLA, we still need a local PyTorch built from source.

### Build PyTorch/XLA from source

Clone the PyTorch repo:

```shell
git clone --recursive https://github.com/pytorch/pytorch pytorch
```

Build PyTorch:

```shell
cd pytorch
_GLIBCXX_USE_CXX11_ABI=0 USE_CUDA=0 BUILD_TEST=0 python3 setup.py develop --prefix=$HOME/.local
```

Clone the PyTorch/XLA repo:

```shell
git clone --recursive https://github.com/pytorch/xla.git torch_xla
```

Modify `torch_xla/bazel/dependencies.bzl` to locate the PyTorch directory:

```shell
PYTORCH_LOCAL_DIR = "../pytorch"
```

Build PyTorch/XLA:

```shell
cd torch_xla
CXX_ABI=0 XLA_CUDA=0 BAZEL_VERBOSE=1 python3 setup.py develop --prefix=$HOME/.local
```

### Build PJRT plugin

You can build the plugin dynamic library using bazel:

```shell
bazel build //xla:vsi_pjrt_plugin
```

The built plugin library is located at `bazel-bin/xla/pjrt/c/pjrt_c_api_vsi_plugin.so`, for development purpose, you can create a symlink to the built plugin library:

```shell
cd torch_xla_vsi_plugin/lib
ln -s ../../bazel-bin/xla/pjrt/c/pjrt_c_api_vsi_plugin.so pjrt_c_api_vsi_plugin.so
```

And add the `torch_xla_vsi_plugin` dir to `PYTHONPATH`.

Or you can build and bundle the plugin as a pip wheel.

```shell
# Build wheel.
pip wheel . -v
# Or install directly.
pip install . -v
```

### Generate compilation database

This command will generate a `compile_commands.json` in current workspace for Clang linter tools.

```shell
bazel run :refresh_compile_commands
```

## Usage

### Set environment variables

```shell
# Locate the vsi unified driver.
VIVANTE_SDK_DIR=${VIV_SDK_INSTALL_PATH}
LD_LIBRARY_PATH=${VIVANTE_SDK_DIR}/[lib|lib64|drivers]
# Need to specify hardware PID if using simulator driver.
VSIMULATOR_CONFIG=VIP9000ULSI_PID0XBA

XLA_STABLEHLO_COMPILE=1
XLA_USE_32BIT_LONG=1
```

### Load plugin dynamically

```python
from torch_xla.experimental import plugins
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from torch_xla_vsi_plugin import VsiPlugin

# Use dynamic PJRT plugin.
vsi_plugin = VsiPlugin()
plugins.use_dynamic_plugins()
plugins.register_plugin("vsi", vsi_plugin)
xr.set_device_type("vsi")

# Now you can use the npu device for PyTorch modules and tensors.
xla_device = xm.xla_device()

```

See more examples in `examples/`.
