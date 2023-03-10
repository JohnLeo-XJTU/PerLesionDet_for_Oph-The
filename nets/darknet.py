#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import torch
from torch import nn
import math

class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
'''ATT'''
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class SE(nn.Module):
    def __init__(self, c1, r=16):
        super(SE, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.l1 = nn.Linear(c1, c1 // r, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.l2 = nn.Linear(c1 // r, c1, bias=False)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.l1(y)
        y = self.relu(y)
        y = self.l2(y)
        y = self.sig(y)
        y = y.view(b, c, 1, 1)
        return x * y
'''???????????????'''



class SiLU(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = SiLU()
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left  = x[...,  ::2,  ::2]
        patch_bot_left  = x[..., 1::2,  ::2]
        patch_top_right = x[...,  ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)

class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super().__init__()
        pad         = (ksize - 1) // 2
        self.conv   = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias)
        self.bn     = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act    = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(in_channels, in_channels, ksize=ksize, stride=stride, groups=in_channels, act=act,)
        self.pconv = BaseConv(in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)

class SPPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), activation="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1      = BaseConv(in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m          = nn.ModuleList([nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2) for ks in kernel_sizes])
        conv2_channels  = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2      = BaseConv(conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x

#--------------------------------------------------#
#   ??????????????????????????????????????????
#--------------------------------------------------#
# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
#         super().__init__()
#         hidden_channels = int(out_channels * expansion)
#         Conv = DWConv if depthwise else BaseConv
#         #--------------------------------------------------#
#         #   ??????1x1???????????????????????????????????????????????????50%
#         #--------------------------------------------------#
#         self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
#         #--------------------------------------------------#
#         #   ??????3x3?????????????????????????????????????????????????????????
#         #--------------------------------------------------#
#         self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
#         self.use_add = shortcut and in_channels == out_channels
#
#     def forward(self, x):
#         y = self.conv2(self.conv1(x))
#         if self.use_add:
#             y = y + x
#         return y


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, in_channels, out_channels, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels
        '''???????????????'''
        self.se = SE(hidden_channels)
    def forward(self, x):
        '''???????????????'''
        y = self.conv2(self.se(self.conv1(x)))
        # y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class CSPLayer(nn.Module):
    def __init__(self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, depthwise=False, act="silu",):
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  
        #--------------------------------------------------#
        #   ???????????????????????????
        #--------------------------------------------------#
        self.conv1  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #--------------------------------------------------#
        #   ????????????????????????????????????
        #--------------------------------------------------#
        self.conv2  = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        #-----------------------------------------------#
        #   ???????????????????????????????????????
        #-----------------------------------------------#
        self.conv3  = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)

        #--------------------------------------------------#
        #   ?????????????????????????????????Bottleneck????????????
        #--------------------------------------------------#
        module_list = [Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, depthwise, act=act) for _ in range(n)]
        self.m      = nn.Sequential(*module_list)

    def forward(self, x):
        #-------------------------------#
        #   x_1???????????????
        #-------------------------------#
        x_1 = self.conv1(x)
        #-------------------------------#
        #   x_2????????????????????????
        #-------------------------------#
        x_2 = self.conv2(x)

        #-----------------------------------------------#
        #   ????????????????????????????????????????????????????????????
        #-----------------------------------------------#
        x_1 = self.m(x_1)
        #-----------------------------------------------#
        #   ????????????????????????????????????????????????
        #-----------------------------------------------#
        x = torch.cat((x_1, x_2), dim=1)
        #-----------------------------------------------#
        #   ???????????????????????????????????????
        #-----------------------------------------------#
        return self.conv3(x)

# class CSPDarknet(nn.Module):
#     def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu",):
#         super().__init__()
#         assert out_features, "please provide output features of Darknet"
#         self.out_features = out_features
#         Conv = DWConv if depthwise else BaseConv
#
#         #-----------------------------------------------#
#         #   ???????????????640, 640, 3
#         #   ????????????????????????64
#         #-----------------------------------------------#
#         base_channels   = int(wid_mul * 64)  # 64
#         base_depth      = max(round(dep_mul * 3), 1)  # 3
#
#         #-----------------------------------------------#
#         #   ??????focus??????????????????????????????
#         #   640, 640, 3 -> 320, 320, 12 -> 320, 320, 64
#         #-----------------------------------------------#
#         self.stem = Focus(3, base_channels, ksize=3, act=act)
#
#         #-----------------------------------------------#
#         #   ?????????????????????320, 320, 64 -> 160, 160, 128
#         #   ??????CSPlayer?????????160, 160, 128 -> 160, 160, 128
#         #-----------------------------------------------#
#         self.dark2 = nn.Sequential(
#             Conv(base_channels, base_channels * 2, 3, 2, act=act),
#             CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
#         )
#
#         #-----------------------------------------------#
#         #   ?????????????????????160, 160, 128 -> 80, 80, 256
#         #   ??????CSPlayer?????????80, 80, 256 -> 80, 80, 256
#         #-----------------------------------------------#
#         self.dark3 = nn.Sequential(
#             Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
#             CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
#         )
#
#         #-----------------------------------------------#
#         #   ?????????????????????80, 80, 256 -> 40, 40, 512
#         #   ??????CSPlayer?????????40, 40, 512 -> 40, 40, 512
#         #-----------------------------------------------#
#         self.dark4 = nn.Sequential(
#             Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
#             CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
#         )
#
#         #-----------------------------------------------#
#         #   ?????????????????????40, 40, 512 -> 20, 20, 1024
#         #   ??????SPP?????????20, 20, 1024 -> 20, 20, 1024
#         #   ??????CSPlayer?????????20, 20, 1024 -> 20, 20, 1024
#         #-----------------------------------------------#
#         self.dark5 = nn.Sequential(
#             Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
#             SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
#             CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise, act=act),
#         )
#
#     def forward(self, x):
#         outputs = {}
#         x = self.stem(x)
#         outputs["stem"] = x
#         x = self.dark2(x)
#         outputs["dark2"] = x
#         #-----------------------------------------------#
#         #   dark3????????????80, 80, 256???????????????????????????
#         #-----------------------------------------------#
#         x = self.dark3(x)
#         outputs["dark3"] = x
#         #-----------------------------------------------#
#         #   dark4????????????40, 40, 512???????????????????????????
#         #-----------------------------------------------#
#         x = self.dark4(x)
#         outputs["dark4"] = x
#         #-----------------------------------------------#
#         #   dark5????????????20, 20, 1024???????????????????????????
#         #-----------------------------------------------#
#         x = self.dark5(x)
#         outputs["dark5"] = x
#         return {k: v for k, v in outputs.items() if k in self.out_features}

class CSPDarknet(nn.Module):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu", ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # -----------------------------------------------#
        #   ??????cbam???????????????
        # -----------------------------------------------#
        # self.cbam1 = CBAM(base_channels * 4)
        # self.cbam2 = CBAM(base_channels * 8)
        # self.cbam3 = CBAM(base_channels * 16)
        # self.cbam1 = SE(base_channels * 4)
        # self.cbam2 = SE(base_channels * 8)
        # self.cbam3 = SE(base_channels * 16)
        self.cbam1 = ECA(base_channels * 4)
        self.cbam2 = ECA(base_channels * 8)
        self.cbam3 = ECA(base_channels * 16)


        self.stem = Focus(3, base_channels, ksize=3, act=act)

        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, depthwise=depthwise, act=act),
        )

        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, depthwise=depthwise, act=act),
        )

        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False, depthwise=depthwise,
                     act=act),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        # ---------------------------------------------------------#
        #   dark3????????????80, 80, 256????????????????????????????????????cbam??????
        # ---------------------------------------------------------#
        x = self.dark3(x)
        x1 = self.cbam1(x)
        outputs["dark3"] = x1
        # --------------------------------------------------------#
        #   dark4????????????40, 40, 512????????????????????????????????????cbam??????
        # --------------------------------------------------------#
        x = self.dark4(x)
        x2 = self.cbam2(x)
        outputs["dark4"] = x2
        # --------------------------------------------------------#
        #   dark5????????????20, 20, 1024????????????????????????????????????cbam??????
        # --------------------------------------------------------#
        x = self.dark5(x)
        x3 = self.cbam3(x)
        outputs["dark5"] = x3
        return {k: v for k, v in outputs.items() if k in self.out_features}


if __name__ == '__main__':
    print(CSPDarknet(1, 1))