# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from detr.util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels


    def forward(self, tensor):
        xs = self.body(tensor)
        return xs

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class BackboneTMVDN(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone_rgb = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        backbone_depth = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        backbone_depth.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=Fals)
        num_channels = 512
        super().__init__(backbone_rgb, backbone_depth, train_backbone, num_channels, return_interm_layers)


class BackboneBaseTMVDN(nn.Module):

    def __init__(self, backbone_rgb: nn.Module, backbone_depth: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        # if return_interm_layers:
        return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body_rgb = IntermediateLayerGetter(backbone_rgb, return_layers=return_layers)
        self.body_depth = IntermediateLayerGetter(backbone_depth, return_layers=return_layers)
        self.num_channels = num_channels
        self.mffsa = MFFSA()

    def forward(self, tensor_list):
        rgbs = self.body_rgb(tensor_list[0])
        depths = self.body_depth(tensor_list[1])

        x = self.mffsa(rgbs, depths)
        return x

class FeatureFusion(nn.Module):
    def __init__(self, in_channels):
        super(FeatureFusion, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_feat, depth_feat):
        # 调整维度以便计算
        b, c, h, w = rgb_feat.shape
        rgb_reshaped = rgb_feat.view(b, c, -1).transpose(1, 2)  # [b, h*w, c]
        depth_reshaped = depth_feat.view(b, c, -1)  # [b, c, h*w]

        # 计算交互矩阵
        interaction = torch.bmm(depth_reshaped, rgb_reshaped)  # [b, h*w, h*w]
        interaction = self.sigmoid(interaction)

        # 生成深度特征的权重并融合
        depth_weighted = torch.bmm(interaction, depth_reshaped)  # [b, c, h*w]
        depth_weighted = depth_weighted.view(b, c, h, w)
        fused = rgb_feat + depth_weighted
        return fused



class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        pool = torch.cat([max_pool, avg_pool], dim=1)
        out = self.conv(pool)
        out = self.sigmoid(out)
        return x * out


class ConvSA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvSA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        shortcut = x
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.sa(x)
        x = x + shortcut
        x = self.relu(x)
        return x


class MFFSA(nn.Module):
    def __init__(self):
        super(MFFSA, self).__init__()

        # FF1
        self.ff1 = FeatureFusion(128)
        self.conv_sa11 = ConvSA(128, 256)
        self.conv_sa12 = ConvSA(256, 512)

        # FF2
        self.ff2 = FeatureFusion(256)
        self.conv_sa2 = ConvSA(256, 512)

        # FF3
        self.ff3 = FeatureFusion(512)


        # concatenate
        self.sa_final = SpatialAttention()
        self.down_conv = nn.Conv2d(1536, 512, kernel_size=1)  # 1536 -> 512
        # self.final_sa =  ConvSA(512, 512)
        self.final_conv = nn.Conv2d(512, 512, kernel_size=1)

    def forward(self, rgb: list, depth: list):

        rgb_128, rgb_256, rgb_512 = rgb
        depth_128, depth_256, depth_512 = depth

        # first stage
        fused1 = self.ff1(rgb_128, depth_128)
        fused1 = self.conv_sa11(fused1)
        fused1 = self.conv_sa12(fused1)

        # second stage
        fused2 = self.ff2(rgb_256, depth_256)
        fused2 = self.conv_sa2(fused2)

        # third stage
        fused3 = self.ff3(rgb_512, depth_512)

        # 拼接特征
        concat_feat = torch.cat([fused1, fused2, fused3], dim=1)
        # 1x1 卷积调整通道
        down_feat = self.down_conv(concat_feat)
        # 空间注意力
        sa_out = self.sa_final(down_feat)

        out = self.final_conv(sa_out)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)


    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_TMVDN_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = True
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

if __name__ == "__main__":
    rgbs = [torch.rand([4, 128, 128, 128]), torch.rand([4, 256, 64, 64]), torch.rand([4, 512, 32, 32])]
    depths = [torch.rand([4, 128, 128, 128]), torch.rand([4, 256, 64, 64]), torch.rand([4, 512, 32, 32])]
    ms = MFFSA()
    print(ms(rgbs, depths))
