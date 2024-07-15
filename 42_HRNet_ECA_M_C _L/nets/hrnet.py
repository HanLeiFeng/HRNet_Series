import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import BN_MOMENTUM, hrnet_classification
from .attention import se_block, cbam_block, eca_block, CA_Block

attetion_block = [se_block, cbam_block, eca_block, CA_Block]

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
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class HRnet(nn.Module):
    def __init__(self, num_classes = 21, backbone = 'hrnetv2_w18', pretrained = False, phi=1):
        super(HRnet, self).__init__()
        self.backbone       = HRnet_Backbone(backbone = backbone, pretrained = pretrained)

        last_inp_channels   = int(np.sum(self.backbone.model.pre_stage_channels))

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )
        if phi >=1 and phi <=4:
            self.feat0_attetion = attetion_block[phi - 1](48)
            self.feat1_attetion = attetion_block[phi - 1](96)
            self.feat2_attetion = attetion_block[phi - 1](192)
            self.feat3_attetion = attetion_block[phi - 1](384)
            # self.feats_attetion = attetion_block[phi - 1](720)

            self.feat_attetion_fuse3210 = attetion_block[phi - 1](384+192+96+48)
            self.feat_attetion_fuse321  = attetion_block[phi - 1](384+192+96)
            self.feat_attetion_fuse32   = attetion_block[phi - 1](384+192)



            # self.linear_fuse_2 = ConvModule(
            #     c1=576,
            #     c2=576,
            #     k=1,
            # )
            # self.linear_fuse_1 = ConvModule(
            #     c1=672,
            #     c2=672,
            #     k=1,
            # )


    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)
        # for i in range(4):
        #     print(x[i].shape)
        # Upsampling[48, 96, 192, 384]

        x0 = self.feat0_attetion(x[0])
        x1 = self.feat1_attetion(x[1])
        x2 = self.feat2_attetion(x[2])
        x3 = self.feat3_attetion(x[3])

        x0_h, x0_w = x0.size(2), x0.size(3)
        x1_h, x1_w = x1.size(2), x1.size(3)
        x2_h, x2_w = x2.size(2), x2.size(3)

        y3 = F.interpolate(x3, size=(x2_h, x2_w), mode='bicubic', align_corners=True)
        x2 = torch.cat([x[2], y3], 1)
        x2 = self.feat_attetion_fuse32(x2)
        y2 = F.interpolate(x2, size=(x1_h, x1_w), mode='bicubic', align_corners=True)
        x1 = torch.cat([x[1], y2], 1)
        x1 = self.feat_attetion_fuse321(x1)
        y0 = F.interpolate(x1, size=(x0_h, x0_w), mode='bicubic', align_corners=True)
        x0 = torch.cat([x[0], y0], 1)
        x0 = self.feat_attetion_fuse3210(x0)
        x = self.last_layer(x0)
        x = F.interpolate(x, size=(H, W), mode='bicubic', align_corners=True)
        return x
