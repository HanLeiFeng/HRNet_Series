# -*- coding: utf-8 -*-
# @Time : 2020/4/7 16:44
# @Author : Zhao HL
# @File : gt_visualizetion.py
import os, sys, time, cv2
import numpy as np
from collections import namedtuple

# 类别信息

gts_gray_path = r'miou_out/detection-results'
gts_color_path = r'miou_out/detection-results_to_color'

Cls = namedtuple('cls', ['name', 'id', 'color'])
Clss = [
    Cls('road', 0, (128, 64,128)),
    Cls('sidewalk', 1, (244, 35,232)),
    Cls('building', 2, ( 70, 70, 70)),
    Cls('wall', 3, (102,102,156)),
    Cls('fence', 4, (190,153,153)),
    Cls('pole', 5, (153,153,153)),
    Cls('traffic light', 6, (250,170, 30)),
    Cls('traffic sign', 7, (220,220,  0)),
    Cls('vegetation', 8, (107,142, 35)),
    Cls('terrain', 9, (152,251,152)),
    Cls('sky', 10, ( 70,130,180)),
    Cls('person', 11, (220, 20, 60)),
    Cls('rider', 12, (255,  0,  0)),
    Cls('car', 13, (0,  0,142)),
    Cls('truck', 14, (0,  0, 70)),
    Cls('bus', 15, (0, 60,100)),
    Cls('train', 16, (0, 80,100)),
    Cls('motorcycle', 17, (0, 0,230)),
    Cls('bicycle', 18, (119, 11, 32)),
    Cls('background', 19 ,(0,0, 0) )
]

# 转化val_tolabelsids这个类的时候用
# Clss = [
#     #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#     Cls(  'unlabeled'            ,  0 ,       (  0,  0,  0) ),
#     Cls(  'ego vehicle'          ,  1 ,       (  0,  0,  0) ),
#     Cls(  'rectification border' ,  2 ,      (  0,  0,  0) ),
#     Cls(  'out of roi'           ,  3 ,       (  0,  0,  0) ),
#     Cls(  'static'               ,  4 ,      (  0,  0,  0) ),
#     Cls(  'dynamic'              ,  5 ,       (111, 74,  0) ),
#     Cls(  'ground'               ,  6 ,       ( 81,  0, 81) ),
#     Cls(  'road'                 ,  7 ,      (128, 64,128) ),
#     Cls(  'sidewalk'             ,  8 ,      (244, 35,232) ),
#     Cls(  'parking'              ,  9 ,      (250,170,160) ),
#     Cls(  'rail track'           , 10 ,       (230,150,140) ),
#     Cls(  'building'             , 11 ,       ( 70, 70, 70) ),
#     Cls(  'wall'                 , 12 ,       (102,102,156) ),
#     Cls(  'fence'                , 13 ,       (190,153,153) ),
#     Cls(  'guard rail'           , 14 ,        (180,165,180) ),
#     Cls(  'bridge'               , 15 ,       (150,100,100) ),
#     Cls(  'tunnel'               , 16 ,       (150,120, 90) ),
#     Cls(  'pole'                 , 17 ,       (153,153,153) ),
#     Cls(  'polegroup'            , 18 ,        (153,153,153) ),
#     Cls(  'traffic light'        , 19 ,        (250,170, 30) ),
#     Cls(  'traffic sign'         , 20 ,      (220,220,  0) ),
#     Cls(  'vegetation'           , 21 ,      (107,142, 35) ),
#     Cls(  'terrain'              , 22 ,      (152,251,152) ),
#     Cls(  'sky'                  , 23 ,      ( 70,130,180) ),
#     Cls(  'person'               , 24 ,      (220, 20, 60) ),
#     Cls(  'rider'                , 25 ,       (255,  0,  0) ),
#     Cls(  'car'                  , 26 ,       (  0,  0,142) ),
#     Cls(  'truck'                , 27 ,       (  0,  0, 70) ),
#     Cls(  'bus'                  , 28 ,        (  0, 60,100) ),
#     Cls(  'caravan'              , 29 ,       (  0,  0, 90) ),
#     Cls(  'trailer'              , 30 ,        (  0,  0,110) ),
#     Cls(  'train'                , 31 ,       (  0, 80,100) ),
#     Cls(  'motorcycle'           , 32 ,       (  0,  0,230) ),
#     Cls(  'bicycle'              , 33 ,       (119, 11, 32) ),
#     Cls(  'license plate'        , -1 ,       (  0,  0,142) ),
# ]

def gray_color(color_dict, gray_path=gts_gray_path, color_path=gts_color_path):
    '''
    swift gray image to color, by color mapping relationship
    :param color_dict:color mapping relationship, dict format
    :param gray_path:gray imgs path
    :param color_path:color imgs path
    :return:
    '''
    pass
    t1 = time.time()
    gt_list = os.listdir(gray_path)
    for index, gt_name in enumerate(gt_list):
        gt_gray_path = os.path.join(gray_path, gt_name)
        gt_color_path = os.path.join(color_path, gt_name)
        gt_gray = cv2.imread(gt_gray_path, cv2.IMREAD_GRAYSCALE)
        # print(gt_gray)
        assert len(gt_gray.shape) == 2  # make sure gt_gray is 1band

        # # region method 1: swift by pix, slow
        # gt_color = np.zeros((gt_gray.shape[0],gt_gray.shape[1],3),np.uint8)
        # for i in range(gt_gray.shape[0]):
        #     for j in range(gt_gray.shape[1]):
        #         gt_color[i][j] = color_dict[gt_gray[i][j]]      # gray to color
        # # endregion

        # region method 2: swift by array
        # gt_color = np.array(np.vectorize(color_dict.get)(gt_gray),np.uint8).transpose(1,2,0)
        # endregion

        # region method 3: swift by matrix, fast
        gt_color = matrix_mapping(color_dict, gt_gray)
        # endregion

        gt_color = cv2.cvtColor(gt_color, cv2.COLOR_RGB2BGR)
        cv2.imwrite(gt_color_path, gt_color)
        process_show(index + 1, len(gt_list))
    print(time.time() - t1)


def color_gray(color_dict, color_path=gts_color_path, gray_path=gts_gray_path, ):
    '''
    swift color image to gray, by color mapping relationship
    :param color_dict:color mapping relationship, dict format
    :param gray_path:gray imgs path
    :param color_path:color imgs path
    :return:
    '''
    gray_dict = {}
    for k, v in color_dict.items():
        gray_dict[v] = k
    t1 = time.time()
    gt_list = os.listdir(color_path)
    print(gt_list)
    for index, gt_name in enumerate(gt_list):
        gt_gray_path = os.path.join(gray_path, gt_name)
        gt_color_path = os.path.join(color_path, gt_name)
        color_array = cv2.imread(gt_color_path, cv2.IMREAD_COLOR)
        assert len(color_array.shape) == 3

        gt_gray = np.zeros((color_array.shape[0], color_array.shape[1]), np.uint8)
        b, g, r = cv2.split(color_array)
        color_array = np.array([r, g, b])
        for cls_color, cls_index in gray_dict.items():
            cls_pos = arrays_jd(color_array, cls_color)
            gt_gray[cls_pos] = cls_index

        cv2.imwrite(gt_gray_path, gt_gray)
        process_show(index + 1, len(gt_list))
    print(time.time() - t1)


def arrays_jd(arrays, cond_nums):
    r = arrays[0] == cond_nums[0]
    g = arrays[1] == cond_nums[1]
    b = arrays[2] == cond_nums[2]
    return r & g & b


def matrix_mapping(color_dict, gt):
    colorize = np.zeros([len(color_dict), 3], 'uint8')
    for cls, color in color_dict.items():
        colorize[cls, :] = list(color)
    ims = colorize[gt, :]
    ims = ims.reshape([gt.shape[0], gt.shape[1], 3])
    return ims


def nt_dic(nt=Clss):
    '''
    swift nametuple to color dict
    :param nt: nametuple
    :return:
    '''
    pass
    color_dict = {}
    for cls in nt:
        color_dict[cls.id] = cls.color
    return color_dict


def process_show(num, nums, pre_fix='', suf_fix=''):
    '''
    auxiliary function, print work progress
    :param num:
    :param nums:
    :param pre_fix:
    :param suf_fix:
    :return:
    '''
    rate = num / nums
    ratenum = round(rate, 3) * 100
    bar = '\r%s %g/%g [%s%s]%.1f%% %s' % \
          (pre_fix, num, nums, '#' * (int(ratenum) // 5), '_' * (20 - (int(ratenum) // 5)), ratenum, suf_fix)
    sys.stdout.write(bar)
    sys.stdout.flush()


if __name__ == '__main__':
    pass
    color_dict = nt_dic()
    gray_color(color_dict)
    # color_gray(color_dict)
