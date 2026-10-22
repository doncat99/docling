# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Eval-only folding for Transformers RT-DETR's ResNet backbone."""

from __future__ import annotations

import copy

from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from transformers.models.rt_detr.modeling_rt_detr_resnet import (
    RTDetrResNetConvLayer,
    RTDetrResNetShortCut,
)
from transformers.models.rt_detr_v2.modeling_rt_detr_v2 import (
    RTDetrV2FrozenBatchNorm2d,
)


def fuse_rtdetr_resnet_layers(model: nn.Module) -> int:
    """Fold fixed normalization into its preceding convolution during eval.

    Only the two explicit RT-DETR ResNet layer types are changed. Activations,
    strides, groups, and residual connections remain in their original owners.
    Call after checkpoint loading and eval(), before inference or compilation.
    The transformed model is for inference, not training or checkpoint export.
    """
    if model.training:
        raise ValueError("RT-DETR convolution fusion requires eval mode")

    count = 0
    for layer in model.modules():
        if not isinstance(layer, (RTDetrResNetConvLayer, RTDetrResNetShortCut)):
            continue
        convolution = layer.convolution
        normalization = layer.normalization
        if isinstance(normalization, nn.BatchNorm2d):
            fused = fuse_conv_bn_eval(convolution, normalization)
        elif isinstance(normalization, RTDetrV2FrozenBatchNorm2d):
            fused = copy.deepcopy(convolution)
            # Transformers' frozen normalization forward fixes epsilon at 1e-5.
            fused.weight, fused.bias = fuse_conv_bn_weights(
                convolution.weight,
                convolution.bias,
                normalization.running_mean,
                normalization.running_var,
                1e-5,
                normalization.weight,
                normalization.bias,
            )
        else:
            continue
        layer.convolution = fused
        layer.set_submodule("normalization", nn.Identity())
        count += 1
    return count
