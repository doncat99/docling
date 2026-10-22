# SPDX-FileCopyrightText: The Docling Contributors
# SPDX-License-Identifier: MIT

"""Eval-only reparameterization of explicit Transformers RT-DETR layers."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.fusion import fuse_conv_bn_eval, fuse_conv_bn_weights
from transformers.models.rt_detr.modeling_rt_detr_resnet import (
    RTDetrResNetConvLayer,
    RTDetrResNetShortCut,
)
from transformers.models.rt_detr_v2.modeling_rt_detr_v2 import (
    RTDetrV2ConvNormLayer,
    RTDetrV2FrozenBatchNorm2d,
    RTDetrV2Model,
    RTDetrV2RepVggBlock,
)


@torch.no_grad()
def fuse_rtdetr_layers(model: nn.Module) -> int:
    """Fold fixed normalization and linear RepVGG branches during eval.

    Only explicit RT-DETR owners are changed; arbitrary Sequential layers are
    untouched. Activations, strides, and residual connections are preserved.
    Call after checkpoint loading and eval(), before inference or compilation.
    The transformed model is for inference, not training or checkpoint export.
    Return the number of folded normalizations and merged branch pairs.
    """
    if model.training:
        raise ValueError("RT-DETR convolution fusion requires eval mode")

    count = 0
    for layer in list(model.modules()):
        if isinstance(layer, RTDetrV2ConvNormLayer) and isinstance(
            layer.norm, nn.BatchNorm2d
        ):
            layer.conv = fuse_conv_bn_eval(layer.conv, layer.norm)
            layer.norm = nn.Identity().eval()
            count += 1
        if isinstance(layer, RTDetrV2Model):
            for projection in [*layer.encoder_input_proj, *layer.decoder_input_proj]:
                if isinstance(projection[1], nn.BatchNorm2d):
                    projection[0] = fuse_conv_bn_eval(projection[0], projection[1])
                    projection[1] = nn.Identity().eval()
                    count += 1
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
        layer.set_submodule("normalization", nn.Identity().eval())
        count += 1

    for name, layer in list(model.named_modules()):
        if not name or not isinstance(layer, RTDetrV2RepVggBlock):
            continue
        # A non-linear branch cannot be merged into a single convolution.
        if not (
            isinstance(layer.conv1.activation, nn.Identity)
            and isinstance(layer.conv2.activation, nn.Identity)
        ):
            continue
        fused = copy.deepcopy(layer.conv1.conv)
        fused.weight = nn.Parameter(
            fused.weight + F.pad(layer.conv2.conv.weight, (1, 1, 1, 1))
        )
        fused.bias = nn.Parameter(fused.bias + layer.conv2.conv.bias)
        model.set_submodule(
            name, nn.Sequential(fused, copy.deepcopy(layer.activation)).eval()
        )
        count += 1
    return count
