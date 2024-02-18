from typing import Any, List
import os
import copy
from pathlib import Path
from argparse import ArgumentParser

import torch
from torch import nn
from torch import fx

from torchvision.io import image, ImageReadMode
from torchvision.transforms import v2 as transforms_v2

from torch_xla.core import xla_model as xm
from torch_xla import runtime as xr
from torch_xla.experimental import plugins as xp

from torch_xla_vsi_plugin import VsiPlugin


def get_args() -> Any:
    parser = ArgumentParser()
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="resnet18",
        choices=[
            "resnet18",
            "regnet_x_400mf",
            "mobilenet_v2",
            "mobilenet_v3_small",
            "squeezenet1_1",
            "shufflenet_v2_x0_5",
            "efficientnet_v2_s",
            "mnasnet0_5",
            "vit_b_16",
            # "swin_v2_t", # Not supported for now.
        ],
        help="Model name."
    )
    parser.add_argument(
        "--image", "-i",
        type=Path,
        required=True,
        help="Path to image file."
    )
    parser.add_argument(
        "--labels", "-l",
        type=Path,
        required=True,
        help="Path to classification labels."
    )
    parser.add_argument(
        "--input_size", "-s",
        type=int,
        default=224,
        help="Input size."
    )
    parser.add_argument(
        "--fold_bn",
        action="store_true",
        help="Whether to fold BatchNorm in models."
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether run models in FP16 precision."
    )
    return parser.parse_args()


def get_model(model_name: str) -> nn.Module:
    model: nn.Module
    if model_name == "resnet18":
        from torchvision.models.resnet import resnet18, ResNet18_Weights
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    elif model_name == "regnet_x_400mf":
        from torchvision.models.regnet import regnet_x_400mf, RegNet_X_400MF_Weights
        model = regnet_x_400mf(weights=RegNet_X_400MF_Weights.DEFAULT)
    elif model_name == "mobilenet_v2":
        from torchvision.models.mobilenet import mobilenet_v2, MobileNet_V2_Weights
        model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    elif model_name == "mobilenet_v3_small":
        from torchvision.models.mobilenet import mobilenet_v3_small, MobileNet_V3_Small_Weights
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    elif model_name == "squeezenet1_1":
        from torchvision.models.squeezenet import squeezenet1_1, SqueezeNet1_1_Weights
        model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
    elif model_name == "shufflenet_v2_x0_5":
        from torchvision.models.shufflenetv2 import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
        model = shufflenet_v2_x0_5(weights=ShuffleNet_V2_X0_5_Weights.DEFAULT)
    elif model_name == "efficientnet_v2_s":
        from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights
        model = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
    elif model_name == "mnasnet0_5":
        from torchvision.models.mnasnet import mnasnet0_5, MNASNet0_5_Weights
        model = mnasnet0_5(weights=MNASNet0_5_Weights.DEFAULT)
    elif model_name == "vit_b_16":
        from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    elif model_name == "swin_v2_t":
        from torchvision.models.swin_transformer import swin_v2_t, Swin_V2_T_Weights
        model = swin_v2_t(weights=Swin_V2_T_Weights.DEFAULT)
    else:
        raise NotImplementedError(f"Unsupported model name: {model_name}.")
    return model


if __name__ == "__main__":
    args = get_args()
    model_name: str = args.model
    img_path: Path = args.image
    labels_path: Path = args.labels
    input_size: int = args.input_size
    is_fold_bn: bool = args.fold_bn
    is_fp16: bool = args.fp16

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

    model = get_model(model_name)
    model.eval()

    # Fuse Conv + BN.
    if is_fold_bn:
        import torch.fx.experimental.optimization as fx_opt
        model: fx.GraphModule = fx_opt.fuse(model)

    if is_fp16:
        model = model.to(torch.float16)

    # Load input image.
    img_chw = image.read_image(path=str(img_path), mode=ImageReadMode.RGB)
    preprocess = transforms_v2.Compose([
        transforms_v2.Resize(
            size=input_size,
            antialias=True,
        ),
        transforms_v2.CenterCrop(input_size),
        transforms_v2.ConvertImageDtype(
            torch.float16 if is_fp16 else torch.float32),
        transforms_v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        transforms_v2.Lambda(lambda img_chw: torch.unsqueeze(img_chw, dim=0))
    ])
    img_nchw: torch.Tensor = preprocess(img_chw)

    # Load classification labels.
    cls_labels: List[str] = []
    with open(labels_path, mode="r") as f:
        for label in f:
            cls_labels.append(label.strip())

    # Run inference with torch to get the groundtruth result.
    with torch.no_grad():
        output_torch = model(img_nchw)

    # Move model and inputs to XLA device.
    model_xla = copy.deepcopy(model).to(xla_device)
    img_nchw_xla = img_nchw.to(xla_device)

    model_xla_compiled = torch.compile(model_xla, backend="openxla")

    print(f"[================ PID: {os.getpid()} ================]")
    with torch.no_grad():
        output_xla = model_xla_compiled(img_nchw_xla)
        xm.mark_step(wait=True)

    output_xla = output_xla.cpu()
    if is_fp16:
        assert torch.allclose(output_torch, output_xla, rtol=1e-01, atol=1e-01)
    else:
        assert torch.allclose(output_torch, output_xla, rtol=1e-03, atol=1e-04)

    cls_torch: int = torch.argmax(output_torch, dim=1).squeeze().item()
    cls_xla: int = torch.argmax(output_xla, dim=1).squeeze().item()
    assert cls_torch == cls_xla

    print("[Classification]")
    print(f"torch: {cls_labels[cls_torch]}, xla: {cls_labels[cls_xla]}")
