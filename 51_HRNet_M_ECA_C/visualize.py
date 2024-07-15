# ----------------------------------------------------#
#   将单张图片预测、摄像头检测和FPS测试功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# ----------------------------------------------------#
import time


import numpy as np
from PIL import Image
from hrnet import HRnet_Segmentation
import torch
import torch.nn.functional as F
import cv2
import os

def feature_vis_mean(feats, img_name): # feaats形状: [b,c,h,w]
     output_shape = (256,512) # 输出形状
     channel_mean = torch.mean(feats,dim=1,keepdim=True) # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
     channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
     channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().numpy() # 四维压缩为二维
     channel_mean = (((channel_mean - np.min(channel_mean))/(np.max(channel_mean)-np.min(channel_mean)))*255).astype(np.uint8)
     savedir = 'D:/01_Current_nets/51_HRNet_M_ECA_C/visualize/'
     if not os.path.exists(savedir+'feature_vis'):
         os.makedirs(savedir+'feature_vis')
     channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
     cv2.imwrite(savedir+'feature_vis/'+ img_name,channel_mean)


def feature_vis_max(feats, img_name):  # feaats形状: [b,c,h,w]
    output_shape = (256, 512)  # 输出形状
    channel_max,_ = torch.max(feats,dim=1,keepdim=True)  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
    channel_max = F.interpolate(channel_max, size=output_shape, mode='bilinear', align_corners=False)
    channel_max = channel_max.squeeze(0).squeeze(0).cpu().numpy()  # 四维压缩为二维
    channel_max = (
                ((channel_max - np.min(channel_max)) / (np.max(channel_max) - np.min(channel_max))) * 255).astype(
        np.uint8)
    savedir = 'D:/01_Current_nets/51_HRNet_M_ECA_C/visualize/'
    if not os.path.exists(savedir + 'feature_vis'):
        os.makedirs(savedir + 'feature_vis')
    channel_max = cv2.applyColorMap(channel_max, cv2.COLORMAP_JET)
    cv2.imwrite(savedir + 'feature_vis/' + img_name, channel_max)


if __name__ == "__main__":
    # -------------------------------------------------------------------------#
    #   如果想要修改对应种类的颜色，到generate函数里修改self.colors即可
    # -------------------------------------------------------------------------#
    hrnet = HRnet_Segmentation()
    name_classes = ["road", "sidewalk", "building", "fence", "pole", "traffic light", "traffic sign",
                    "vegetation", "terrain", "sky", "diningtable", "person", "rider", "car", "truck", "bus",
                    "train", "motorcycle", "bicycle", "background"]
    img_path = 'img/street_01.png'
    try:
        image = Image.open(img_path)
    except:
        print('Open Error! Try again!')

    outs = hrnet.visualize_image(image, count=False, name_classes=name_classes)
    img_0 = outs[0]
    img_1 = outs[1]
    img_2 = outs[2]
    img_3 = outs[3]
    feature_vis_mean(img_0, "stage0_mean.jpg")
    feature_vis_mean(img_1, "stage1_mean.jpg")
    feature_vis_mean(img_2, "stage2_mean.jpg")
    feature_vis_mean(img_3, "stage3_mean.jpg")
    feature_vis_max(img_0, "stage0_max.jpg")
    feature_vis_max(img_1, "stage1_max.jpg")
    feature_vis_max(img_2, "stage2_max.jpg")
    feature_vis_max(img_3, "stage3_max.jpg")


