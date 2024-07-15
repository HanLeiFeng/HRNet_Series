import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BN_MOMENTUM, hrnet_classification
from .ma import ma_block

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

class HRnet_Backbone(nn.Module):
    def __init__(self, backbone = 'hrnetv2_w48', pretrained = True):
        super(HRnet_Backbone, self).__init__()
        self.model    = hrnet_classification(backbone = backbone, pretrained = pretrained)
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier
        

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        
        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.model.stage2(x_list)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage3(x_list)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.model.stage4(x_list)
        
        return y_list

class HRnet(nn.Module):
    def __init__(self, num_classes = 21, backbone = 'hrnetv2_w18', pretrained = False):
        super(HRnet, self).__init__()
        self.backbone       = HRnet_Backbone(backbone = backbone, pretrained = pretrained)

        last_inp_channels   = int(np.sum(self.backbone.model.pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

        self.linear_c0 = MLP(input_dim=48,  embed_dim=48)
        # self.linear_c1 = MLP(input_dim=96,  embed_dim=16)
        # self.linear_c2 = MLP(input_dim=192, embed_dim=32)
        # self.linear_c3 = MLP(input_dim=384, embed_dim=64)

        channel_1 = 48
        channel_2 = 96
        channel_3 = 192
        channel_4 = 384

        self.ma_1 = ma_block(channel_1, channel_2, 16)
        self.ma_2 = ma_block(channel_1, channel_3, 16)
        self.ma_3 = ma_block(channel_1, channel_4, 16)
        self.ma_4 = ma_block(channel_2, channel_3, 16)
        self.ma_5 = ma_block(channel_2, channel_4, 16)
        self.ma_6 = ma_block(channel_3, channel_4, 16)


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)

        x0 = self.linear_c0(x[0]).permute(0,2,1).reshape(x[0].shape[0], -1, x[0].shape[2], x[0].shape[3])
        # x1 = self.linear_c1(x[1]).permute(0,2,1).reshape(x[1].shape[0], -1, x[1].shape[2], x[1].shape[3])
        # x2 = self.linear_c2(x[2]).permute(0,2,1).reshape(x[2].shape[0], -1, x[2].shape[2], x[2].shape[3])
        # x3 = self.linear_c3(x[3]).permute(0,2,1).reshape(x[3].shape[0], -1, x[3].shape[2], x[3].shape[3])
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]

        # com_1 = [x0, x1]
        # com_2 = [x0, x2]
        com_3 = [x0, x3]
        # com_4 = [x1, x2]
        # com_5 = [x1, x3]
        # com_6 = [x2, x3]

        # com_1 = self.ma_1(com_1)
        # com_2 = self.ma_2(com_2)
        com_3 = self.ma_3(com_3)
        # com_4 = self.ma_4(com_4)
        # com_5 = self.ma_5(com_5)
        # com_6 = self.ma_6(com_6)

        # x0 = torch.cat([com_1[0], com_2[0], com_3[0]], dim= 1)
        # x1 = torch.cat([com_1[1], com_4[0], com_5[0]], dim= 1)
        # x2 = torch.cat([com_2[1], com_4[1], com_6[0]], dim= 1)
        # x3 = torch.cat([com_3[1], com_5[1], com_6[1]], dim= 1)
        x0 = com_3



        x0_h, x0_w = x0.size(2), x0.size(3)
        x1 = F.interpolate(x1, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)

        x = torch.cat([x0, x1, x2, x3], 1)

        x = self.last_layer(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
