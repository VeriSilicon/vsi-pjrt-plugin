import copy
import os

import torch
from torch import nn

from torch_xla.core import xla_model as xm
from torch_xla import runtime as xr
from torch_xla.experimental import plugins as xp

from torch_xla_vsi_plugin import VsiPlugin


class TestModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv_0 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False
        )
        self.bn_0 = nn.BatchNorm2d(32)
        self.relu_0 = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0)
        )

        self.conv_1 = nn.Conv2d(
            in_channels=32,
            out_channels=16,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )
        self.relu_1 = nn.ReLU(inplace=True)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(
            output_size=(1, 1)
        )

        self.fc = nn.Linear(
            in_features=16,
            out_features=10
        )

    def forward(self, input_nhwc: torch.Tensor) -> torch.Tensor:
        input_nchw = torch.permute(input_nhwc, (0, 3, 1, 2))
        x = self.conv_0(input_nchw)
        x = self.bn_0(x)
        x = self.relu_0(x)
        x = self.max_pool(x)
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.global_avg_pool(x)
        x = torch.squeeze(x, dim=(2, 3))
        x = self.fc(x)
        out = torch.sigmoid(x)
        return out


if __name__ == "__main__":
    # Log device type.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
    os.environ["TF_CPP_VMODULE"] = "pjrt_registry=5"

    # Load dynamic VSI PJRT plugin.
    vsi_plugin = VsiPlugin()
    xp.use_dynamic_plugins()
    xp.register_plugin("vsi", vsi_plugin)
    xr.set_device_type("vsi")

    device_type = xr.device_type()
    assert device_type == "vsi"

    xla_device = xm.xla_device()

    input_tensor = torch.randn(size=(1, 64, 64, 3), dtype=torch.float32)
    model = TestModel().eval()

    sample_inputs = (input_tensor, )
    with torch.no_grad():
        output_torch = model(*sample_inputs)

    # Move model and inputs to XLA device.
    model_xla = copy.deepcopy(model).to(xla_device)
    sample_inputs_xla = tuple(
        tensor.to(xla_device) for tensor in sample_inputs
    )

    model_xla_compiled = torch.compile(model_xla, backend="openxla")

    print(f"[================ PID: {os.getpid()} ================]")
    with torch.no_grad():
        # Run inference for multiple times.
        for _ in range(3):
            output_xla = model_xla_compiled(*sample_inputs_xla)
            xm.mark_step(wait=True)

            output_xla = output_xla.cpu()
            assert torch.allclose(output_torch, output_xla, rtol=1e-03, atol=1e-04)
