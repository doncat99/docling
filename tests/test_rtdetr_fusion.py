# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

import pytest
import torch
from torch import nn
from transformers.models.rt_detr.modeling_rt_detr_resnet import (
    RTDetrResNetConvLayer,
    RTDetrResNetShortCut,
)
from transformers.models.rt_detr_v2.modeling_rt_detr_v2 import (
    RTDetrV2FrozenBatchNorm2d,
)

from docling.models.inference_engines.object_detection.rtdetr_fusion import (
    fuse_rtdetr_resnet_layers,
)


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("shortcut", [False, True])
@pytest.mark.parametrize("device", ["cpu", "mps"])
@torch.no_grad()
def test_folding_preserves_loaded_resnet_outputs(
    frozen: bool, shortcut: bool, device: str
) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("requires Apple MPS")
    torch.manual_seed(31)
    layer = (
        RTDetrResNetShortCut(4, 8, stride=2)
        if shortcut
        else RTDetrResNetConvLayer(4, 8, kernel_size=3, stride=2)
    )
    if frozen:
        layer.normalization = RTDetrV2FrozenBatchNorm2d(8)
    normalization = layer.normalization
    normalization.weight.copy_(torch.rand(8) + 0.1)
    normalization.bias.copy_(torch.randn(8))
    normalization.running_mean.copy_(torch.randn(8))
    normalization.running_var.copy_(torch.rand(8) + 0.2)
    layer.eval().to(device)
    image = torch.randn(2, 4, 17, 19, device=device)
    reference = layer(image)
    assert fuse_rtdetr_resnet_layers(layer) == 1
    torch.testing.assert_close(layer(image), reference, rtol=2e-5, atol=2e-6)
    assert fuse_rtdetr_resnet_layers(layer) == 0
    torch.testing.assert_close(layer(image), reference, rtol=2e-5, atol=2e-6)


def test_folding_rejects_training_and_leaves_other_models_unchanged() -> None:
    model = nn.Sequential(nn.Conv2d(4, 8, 3), nn.BatchNorm2d(8))
    with pytest.raises(ValueError, match="eval"):
        fuse_rtdetr_resnet_layers(model)
    assert fuse_rtdetr_resnet_layers(model.eval()) == 0
    assert isinstance(model[1], nn.BatchNorm2d)
