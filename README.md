# VSI PJRT Plugin

This is a PJRT client implementation for the VeriSilicon NPU/GPU platform as a dynamic plugin.

## Building

### Install PyTorch

```shell
pip3 install torch~=2.3.0 torchvision~=0.18.0 --index-url https://download.pytorch.org/whl/cpu
```

### Install PyTorch/XLA

```shell
pip3 install torch_xla~=2.3.0
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

# Map PyTorch Long type to HLO s32 type.
XLA_USE_32BIT_LONG=1
# Since there's no StableHLO -> HLO conversion of Q/DQ ops,
# need to disable HLO -> StableHLO -> HLO roundtrip.
XLA_STABLEHLO_COMPILE=0
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
