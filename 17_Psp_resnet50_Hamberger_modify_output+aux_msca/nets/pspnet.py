import torch
import torch.nn.functional as F
from torch import nn

from nets.mobilenetv2 import mobilenetv2
from nets.resnet import resnet50

from Hamburger_Decoder.decoder import HamDecoder
import yaml
with open('Hamburger_Decoder/config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)
from MSCN_modify.MSCA import MSCANet

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
        self.bn3 = model.bn1
        self.relu3 = model.relu
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
        # print("x.shape_1")
        # print(x.shape)
        x = self.conv1(x)
        # print("x.shape_2/conv1")
        # print(x.shape)
        x = self.maxpool(x)
        # print("x.shape_3/maxpool")
        # print(x.shape)
        x = self.layer1(x)
        # print("x.shape_4/layer_1")
        # print(x.shape)
        x = self.layer2(x)
        # print("x.shape_5/layer_2")
        # print(x.shape)
        x_aux = self.layer3(x)
        # print("x.shape_6/layer_3")
        # print(x_aux.shape)
        x = self.layer4(x_aux)
        # print("x.shape_7/layer_4")
        # print(x.shape)
        return x_aux, x

class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        #--------------------------------------------------------------------------------------------#
        #   根据下采样因子修改卷积的步长与膨胀系数
        #   当downsample_factor=16的时候，我们最终获得两个特征层，shape分别是：30,30,320和30,30,96
        #--------------------------------------------------------------------------------------------#
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
        
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
        x_aux = self.features[:14](x)
        x = self.features[14:](x_aux)
        return x_aux, x
 
class _PSPModule(nn.Module):
    def __init__(self, in_channels, pool_sizes, norm_layer):
        super(_PSPModule, self).__init__()
        out_channels = in_channels // len(pool_sizes)
        #2048/4 = 512
        #-----------------------------------------------------#
        #   分区域进行平均池化
        #   30, 30, 320 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 + 30, 30, 80 = 30, 30, 640
        #-----------------------------------------------------#
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, pool_size, norm_layer) for pool_size in pool_sizes])
        
        # 30, 30, 640 -> 30, 30, 80
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(pool_sizes)), out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = norm_layer(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        # print("features.shape")
        # print(features.shape)
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages])
        # print("pyramids.shape")
        # for i in range(5):
        #     print(pyramids[i].shape)
        # print('********************************************')
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        # print("output.shape_bottleneck")
        # print(output.shape)
        # print('==============================================')
        return output


class PSPNet(nn.Module):
    def __init__(self, num_classes, downsample_factor, backbone="resnet50", pretrained=True, aux_branch=True):
        super(PSPNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        if backbone=="resnet50":
            self.backbone = Resnet(downsample_factor, pretrained)
            aux_channel = 1024
            out_channel = 2048
        elif backbone=="mobilenet":
            #----------------------------------#
            #   获得两个特征层
            #   f4为辅助分支    [30,30,96]
            #   o为主干部分     [30,30,320]
            #----------------------------------#
            self.backbone = MobileNetV2(downsample_factor, pretrained)
            aux_channel = 96
            out_channel = 320
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, resnet50.'.format(backbone))

        #--------------------------------------------------------------#
        #	PSP模块，分区域进行池化
        #   分别分割成1x1的区域，2x2的区域，3x3的区域，6x6的区域
        #   30,30,320 -> 30,30,80 -> 30,30,21
        #--------------------------------------------------------------#
        self.master_branch = nn.Sequential(
            _PSPModule(out_channel, pool_sizes=[1, 2, 3, 6], norm_layer=norm_layer),
            HamDecoder(512, config, 512),
            nn.Conv2d(out_channel//4, num_classes, kernel_size=1)
        )

        self.aux_branch = aux_branch

        if self.aux_branch:
            #---------------------------------------------------#
            #	利用特征获得预测结果
            #   30, 30, 96 -> 30, 30, 40 -> 30, 30, 21
            #---------------------------------------------------#
            self.msca = MSCANet(in_channnels=1024, embed_dims=[256, 512, 460, 256],
                                ffn_ratios=[4, 4, 4, 4], depths=[2, 4],
                                num_stages=2, ls_init_val=1e-2, drop_path=0.0)

            self.auxiliary_branch = nn.Sequential(
                #1024--->2048/8=256
                HamDecoder(512, config, 768),
                nn.Conv2d(512, out_channel//8, kernel_size=3, padding=1, bias=False),
                norm_layer(out_channel//8),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                # 256--->21
                nn.Conv2d(out_channel//8, num_classes, kernel_size=1)
            )

        self.initialize_weights(self.master_branch)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x_aux, x = self.backbone(x)
        # print("PSPNet:输入的主干特征图")
        # print(x.shape)
        # print("PSPNet:x_aux.shape：输入的辅助特征图")
        # print(x_aux.shape)
        output = self.master_branch(x)
        # print("PSPNet:output.shape_PSP：主干特征经过PSPnet后的输出，经过Hanmber解码，已经调整好输出通道数为num_class")
        # print(output.shape)
        output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=True)
        # print("PSPNet:output.shape_PSP：主干特征的最终输出")
        # print(output.shape)
        if self.aux_branch:
            x_aux = self.msca(x_aux)
            # print()
            # print("==========***********************=======")
            # print("x_aux_MSCA：辅助分类器经过多尺度变换后的输出")
            # for i in range(2):
            #     print(x_aux[i].shape)
            x_aux[1] = F.interpolate(x_aux[1], size=(x_aux[0].shape[-2], x_aux[0].shape[-1]), mode='bilinear',
                                     align_corners=True)
            # print("PSPNet:output_aux.shape_1：将网络经过MSAC后输出的特征图进行拼接输出[256+512, 64, 64]")
            # for i in range(2):
            #     print(x_aux[i].shape)
            output_aux = self.auxiliary_branch(torch.cat(x_aux, dim=1))
            # print("PSPNet:output_aux.shape：辅助分支输出，已调整通道数")
            # print(output_aux.shape)
            output_aux = F.interpolate(output_aux, size=input_size, mode='bilinear', align_corners=True)
            # print("PSPNet:output_aux.shape：辅助分支输出，已调恢复图像尺寸")
            # print(output_aux.shape)
            return output_aux, output
        else:
            return output

    def initialize_weights(self, *models):
        for model in models:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1.)
                    m.bias.data.fill_(1e-4)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, 0.0001)
                    m.bias.data.zero_()
