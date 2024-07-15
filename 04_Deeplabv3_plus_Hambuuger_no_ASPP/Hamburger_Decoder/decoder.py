#%%

from turtle import forward
import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml
from Hamburger_Decoder.hamburger import HamBurger
from Hamburger_Decoder.bricks import SeprableConv2d, ConvRelu, ConvBNRelu, resize
with open('D:/01_Current_nets/04_Deeplabv3_plus_Hambuuger_no_ASPP/Hamburger_Decoder/config.yaml') as fh:
    config = yaml.load(fh, Loader=yaml.FullLoader)

class HamDecoder(nn.Module):
    '''SegNext'''
    def __init__(self, outChannels, config, enc_embed_dims=[16,24,32,96]):
        super().__init__()

        ham_channels = config['ham_channels']

        self.squeeze = ConvRelu(sum(enc_embed_dims[1:]), ham_channels)
        self.ham_attn = HamBurger(ham_channels, config)
        self.align = ConvRelu(ham_channels, outChannels)
       
    def forward(self, features):
        
        features = features[1:] # drop stage 1 features b/c low level
        features = [resize(feature, size=features[-3].shape[2:], mode='bilinear') for feature in features]
        x = torch.cat(features, dim=1)

        x = self.squeeze(x)
        x = self.ham_attn(x)
        x = self.align(x)       

        return x


#%%
#
# import torch.nn.functional as F
#
# def resize(input,
#            size=None,
#            scale_factor=None,
#            mode='nearest',
#            align_corners=None,
#            warning=True):
#
#     return F.interpolate(input, size, scale_factor, mode, align_corners)
#
# y1 = torch.randn((6, 32, 128, 256))
# y2 = torch.randn((6, 64, 128, 256))
# y3 = torch.randn((6, 460, 128, 256))
# y4 = torch.randn((6, 256, 16, 32))
# x = []
# x.append(y1)
# x.append(y2)
# x.append(y3)
# x.append(y4)
#
# inputs = [resize(
#         level,
#         size=x[0].shape[2:],
#         mode='bilinear',
#         align_corners=False
#     ) for level in x]
#
# for i in range(4):
#     print(x[i].shape)
# for i in range(4):
#     print(inputs[i].shape)
#
#
#
# inputs = torch.cat(inputs, dim=1)
# print(inputs.shape)
#
#
# model = HamDecoder(21, config, enc_embed_dims=[32,64,460,256])
#
# x = model.forward(x)
# print(x.shape)

