# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from nets.flatten_pvt import FLatten_CSWin_64_24181_tiny_224

class SegFormer0(nn.Module):
    def __init__(self, pretrained=False, num_classes=21):
        super(SegFormer0, self).__init__()
        self.model = FLatten_CSWin_64_24181_tiny_224(pretrained=pretrained, num_classes=num_classes)

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        # print("Flatten20",inputs.shape)
        x = self.model(inputs)
        # print("Flatten22", x.shape)
        x = x.permute(0, 3, 1, 2)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        # print("Flatten24", x.shape)
        return x

# # 创建模型实例
# model   = SegFormer0(pretrained=True)
#
# # 定义输入形状
# batch_size = 2
# channels = 3
# height = 224
# width = 224
#
# # 生成随机输入张量
# input_tensor = torch.randn(batch_size, channels, height, width)
# print("input_tensor输出结果形状:", input_tensor.shape)
# # 将输入张量传递给模型进行验证
# output = model(input_tensor)
#
# # 输出结果形状
# print("输出结果形状:", output.shape)