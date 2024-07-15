# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from nets.resnet import resnet50
from nets.xception import xception
from nets.STL import STL
from Hamburger_Decoder.decoder import HamDecoder
import yaml
with open('Hamburger_Decoder/config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

class Resnet(nn.Module):
    def __init__(self, dilate_scale=8, pretrained=True):
        super(Resnet, self).__init__()
        from functools import partial
        model = resnet50(pretrained)

        #--------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,1024和30,30,2048
        #--------------------------------------------------------------------------------------------#
        if dilate_scale == 8:
            model.layer3.apply(partial(self._nostride_dilate, dilate=2))
            model.layer4.apply(partial(self._nostride_dilate, dilate=4))
        elif dilate_scale == 16:
            model.layer4.apply(partial(self._nostride_dilate, dilate=2))

        # self.conv1 = model.conv1[0]
        # self.bn1 = model.conv1[1]
        # self.relu1 = model.conv1[2]
        # self.conv2 = model.conv1[3]
        # self.bn2 = model.conv1[4]
        # self.relu2 = model.conv1[5]
        # self.conv3 = model.conv1[6]
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x):
        # x = self.relu1(self.bn1(self.conv1(x)))
        # x = self.relu2(self.bn2(self.conv2(x)))
        # x = self.relu3(self.bn3(self.conv3(x)))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        # print("x.shape_1")
        # print(x.shape)
        # x = self.layer2(x)
        # print("x.shape_2")
        # print(x.shape)
        # x = self.layer3(x)
        # print("x.shape_3")
        # print(x.shape)
        # x = self.layer4(x)
        # print("x.shape_4")
        # print(x.shape)
        return x

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
        
        # _c4 = c4.permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # _c3 = c3.permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        # _c2 = c2.permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = c1

        # _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = torch.cat([_c4, _c3, _c2, _c1], dim=1)


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
        # self.resnet50 = Resnet(dilate_scale=8, pretrained=True)
        # self.STL = STL(256, 128)
        # self.conv = nn.Conv2d(128+768, 128, 1, 1, 0, bias=False)
        # self.bn = nn.BatchNorm2d(128, momentum=0.0003)
        # self.relu = nn.ReLU(inplace=True)
        self.decoder = HamDecoder(num_classes, config, 1024)
        # self.linear_pred = nn.Conv2d(768, num_classes, kernel_size=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, inputs):
        # inputs = inputs.to(self.device)  # device应是cuda:0或其他GPU设备索引
        inputs = inputs.to(self.device)

        H, W = inputs.size(2), inputs.size(3)
        # x_oux = self.resnet50(inputs)
        # print("x_oux.shape_1")
        # print(x_oux.shape)
        # x_oux = self.STL(x_oux)
        # print("x_oux.shape_2")
        # print(x_oux.shape)
        x = self.backbone.forward(inputs)
        for i in range(4):
            x[i] = x[i].to(self.device)
        x = self.decode_head.forward(x)
        x = x.to(self.device)
        # x = torch.cat([x, x_oux],dim=1)
        # x = self.conv(x)
        # x = self.bn(x)
        # x = self.relu(x)
        # print("x.shape_2")
        # print(x.shape)
        # x = self.linear_pred(x)
        x = self.decoder(x)
        x = x.to(self.device)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)


        return x
