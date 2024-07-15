import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BN_MOMENTUM, hrnet_classification
from MSCN.MSCA import MSCANet


class HRnet_Backbone(nn.Module):
    def __init__(self, backbone='hrnetv2_w48', pretrained=True):
        super(HRnet_Backbone, self).__init__()
        self.model = hrnet_classification(backbone=backbone, pretrained=pretrained)
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

class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class HRnet(nn.Module):
    def __init__(self, num_classes=21, backbone='hrnetv2_w18', pretrained=False):
        super(HRnet, self).__init__()
        self.backbone = HRnet_Backbone(backbone=backbone, pretrained=pretrained)

        last_inp_channels = int(np.sum(self.backbone.model.pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1,
                      padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        self.msca_2 = MSCANet(in_channnels=192, embed_dims=[192, 384],
                              ffn_ratios=[4, 4, 4, 4], depths=[5, 3],
                              num_stages=2, ls_init_val=1e-2, drop_path=0.0)

        self.msca_3 = MSCANet(in_channnels=384, embed_dims=[384, 1],
                              ffn_ratios=[4, 4, 4, 4], depths=[5, 3],
                              num_stages=1, ls_init_val=1e-2, drop_path=0.0)



        self.linear_fuse_2 = ConvModule(
            c1=192 * 2,
            c2=192,
            k=1,
        )

        self.linear_fuse_3 = ConvModule(
            c1=384 * 3,
            c2=384,
            k=1,
        )


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)
        # for i in range(4):
        #     print(x[i].shape)
        # torch.Size([2, 48, 128, 256])
        # torch.Size([2, 96, 64, 128])
        # torch.Size([2, 192, 32, 64])
        # torch.Size([2, 384, 16, 32])
        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]
        x3 = x[3]

        x2_aux = self.msca_2.forward(x2)
        x2_aux_0 = x2_aux[0]
        x2_aux_1 = x2_aux[1]
        x3_aux = self.msca_3.forward(x3)[0]

        # print("x2_aux_0.shape")
        # print(x2_aux_0.shape)
        # print("x2_aux_1.shape")
        # print(x2_aux_1.shape)
        # print("x3_aux.shape")
        # print(x3_aux.shape)

        x2 = torch.cat([x2, x2_aux_0], 1)
        x3 = torch.cat([x3, x3_aux, x2_aux_1], 1)

        x2 = self.linear_fuse_2(x2)
        x3 = self.linear_fuse_3(x3)

        x1 = F.interpolate(x1, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x2, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x0, x1, x2, x3], 1)
        # print(x0.shape)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)

        x = self.last_layer(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
