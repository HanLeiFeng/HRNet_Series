# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from Hamburger_Decoder.decoder import HamDecoder
from nets.STL import STL

import yaml
with open('Hamburger_Decoder/config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(SegFormerHead, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # print("Stage4_MPL_UP_Shape:", _c4.shape)
        # print("Stage3_MPL_UP_Shape:", _c3.shape)
        # print("Stage2_MPL_UP_Shape:", _c2.shape)
        # print("Stage1_MPL_UP_Shape:", _c1.shape)

        x = torch.cat([_c4, _c3, _c2], dim=1)

        # x = self.dropout(_c)
        # x = self.linear_pred(x)

        return x

class SegFormer(nn.Module):
    def __init__(self, num_classes = 21, phi = 'b0', pretrained = False):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = SegFormerHead(num_classes, self.in_channels, self.embedding_dim)
        self.decoder = HamDecoder(num_classes, config, 3072)
        self.STL = STL(64, 768)


    def forward(self, inputs):
        inputs = inputs.to(torch.device('cuda'))
        H, W = inputs.size(2), inputs.size(3)

        variable = inputs

        # 检查是否为PyTorch张量
        # print("原始输入图像大小：", inputs.shape)
        x = self.backbone.forward(inputs)
        # print("SegFormer网络经过4个Stages后输出图像的大小：")
        # for i in range(4):
        #     print("{:<5} {:<1} {:<3} {:<20}".format('Stage', i, "的大小为:torch.Size([", ', '.join(str(dim) for dim in x[i].shape)))
        stl_layer = self.STL(x[0])
        # print("SegFormer网络Stage_01输出图像经过统计纹理增强STL后的大小：", stl_layer.shape)
        x = self.decode_head.forward(x)
        # print("后三个Stage输出的特征图经过多层感知机后拼接输出的特征图大小:", x.shape)
        x = torch.cat([stl_layer, x], dim=1)
        # print("将经过统计纹理增强后的特征图与SegFormer网络Backbone后三个阶段的特征图进行拼接后的大小:", x.shape)
        x = self.decoder(x)
        # print("MLP输出的特征图经过Hambergure解码器后的大小:", x.shape)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        # print("经过上采样操作后输出的特征图大小: ",x.shape)
        return x
