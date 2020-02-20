#This file is created Sep 29.
#Author: Zhenhao Zhou
#It is config of project
# all my project image map is (H,W,C)
from easydict import EasyDict as edict

__CONFIG=edict()
cfg=__CONFIG
__CONFIG.TRAIN=edict()
__CONFIG.TEST=edict()
# the number of class
__CONFIG.NUM_CLASSES=9
# the data path of VOC dataset
__CONFIG.DATA_DIR="/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/capstone/VOC2007/"
# the data set path of images
__CONFIG.IMAGESET="/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/capstone/VOC2007/JPEGImages/"
# the data set of xml annotations.
__CONFIG.ANOTATIONS="/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/capstone/VOC2007/Annotations/"
# the path of testing data
__CONFIG.DATA_TEST="/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/capstone/VOC2007/ImageSets/Main/test.txt"
# whether using train blance class
__CONFIG.TRAIN.BALANCE_CLASSES=True
# the batch size of training
__CONFIG.TRAIN.BATCH_SIZE=256
# the batch size in test time
__CONFIG.TEST.BATCH_SIZE=64
# what the size of image input network
__CONFIG.TRAIN.RESIZE_ROIDB=224
# How many anchor it will product.
__CONFIG.TRAIN.ANCHOR_NUM=12
# the ratios of Anchor
__CONFIG.TRAIN.RATIOS=[0.5,1,2]
# Anchor scale for rpn_network
__CONFIG.TRAIN.SCALES=[32,64,16,128]
# rpn_network network stride
__CONFIG.TRAIN.RPN_STRIDE=16
# the image size of training data
__CONFIG.TRAIN.IMAGE_SIZE=1024
# IOU >= thresh: positive example
__CONFIG.TRAIN.IOUPOS=0.5
# IOU < thresh: negative example
__CONFIG.TRAIN.IOUNEG=0.3
# Using scale to resize image size
# the max side pixel size
__CONFIG.TRAIN.MAX_SIZE=1000
# the min side of pixel size
__CONFIG.TRAIN.MIN_SIZE=600
# the epoch of training
__CONFIG.TRAIN.EPOCH=20
# the downscale of training
__CONFIG.TRAIN.DOWNSCSLE=16
# the weight of rpn_network reg loss
__CONFIG.TRAIN.REG_WEIGHT=10
# the max number of rois that generator by rpn_network:
__CONFIG.MAX_ROIS=64
# the threshold of roi overlap iou
__CONFIG.ROI_OVER_IOU=0.9
# 基于样本估算标准偏差。标准偏差反映数值相对于平均值(mean) 的离散程度。
__CONFIG.TRAIN.std_scaling = 4.0
__CONFIG.TRAIN.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]
# the threshold of nms algorithm
__CONFIG.TRAIN.NMS_THRESHOLD=0.9
# the max box number that nms algorithm pick.
__CONFIG.TRAIN.NMS_MAX=300
# the class name of flaw
__CONFIG.CLASSES_NAME=["PI", "PN", "XO", "NP", "HD", "FB", "FO", "FP"]
# the pooling size of roi pooling
__CONFIG.ROI_POOLING_SIZE=14
# the path of resnet weight.
__CONFIG.RESNET_PATH='/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/capstone/resnet_50_weughts.h5'
