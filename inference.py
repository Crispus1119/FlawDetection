import tensorflow as tf
from data_handler.imdb import imdb
from network.resnet import resnet
from PIL import Image as Image
import os
import cv2
from rpn_network.roi_helper import rpn_to_roi,handle_roi_fm,nms,cal_transform,match_gt_box
from tools.config import cfg
from data_handler.data_generator_rpn import data_generator_rpn
from rpn_network.rpn import rpn_net
import numpy as np
from tools import debug
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def get_result(roi_record,reg_record,cls_record):
    boxes=[]
    label_record=np.argmax(cls_record,axis=1)
    probs=np.max(cls_record,axis=1)
    # boxes :[xmin,ymin,xmax,ymax]
    for i in range(len(roi_record)):
        w=roi_record[i][2]
        h=roi_record[i][3]
        box=[roi_record[i][0]+w/2,roi_record[i][1]+h/2,w,h]
        box=cal_transform(box,reg_record[i])
        box=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
        boxes.append(box)
    cls_labels, indices = np.unique(label_record, return_inverse=True)
    boxes=np.array(boxes)
    dic={}
    for i in range(len(cls_labels)):
        index = np.where(np.equal(i, indices))[0]
        dic[str(i)]=nms(boxes[index],probs)
    return dic

def nms(box):
    xmin=box[:,0]
    ymin=box[:,1]
    xmax=box[:,2]+box[:,0]
    ymax=box[:,3]+box[:,1]
    score=box[:,4]
    areas=(xmax-xmin)*(ymax-ymin)
    # get the index of nms score.
    index=np.argsort(score)[::-1]
    store=[]
    while index.size>0:
        i=index[0]
        store.append(i)
        # in numpy array list means the index of array.
        # so now the order of  xx1,yy1,xx2,yy2 is the order like index.
        xx1=np.maximum(xmin[i],xmin[index[1:]])
        yy1=np.maximum(ymin[i],ymin[index[1:]])
        xx2=np.minimum(xmax[i],ymax[index[1:]])
        yy2=np.minimum(ymax[i],ymax[index[1:]])
        # if the two box are not intersecting,w and h will be 0
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection=w*h
        iou=intersection/(areas[i]+areas[index[1:]]-intersection)
        loc=np.where(iou==0.0)[0]
        index=index[loc+1]
    return box[store]

def iou( box, gt_box):
        # if there has no intersection there have four possible
        if (box[0] >= gt_box[2] or box[1] >= gt_box[3] or gt_box[0] >= box[2] or gt_box[1] > box[3]):
            return 0
        x_min = max(box[0], gt_box[0])
        x_max = min(box[2], gt_box[2])
        y_min = max(box[1], gt_box[1])
        y_max = min(box[3], gt_box[3])
        area = (x_max - x_min) * (y_max - y_min)
        square = (box[2] - box[0]) * (box[3] - box[1])
        square += (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        iou = area / (square - area + 1e-10)
        return iou
def test():
    data = imdb('test')
    data_gen = data_generator_rpn()
    img_input = tf.keras.layers.Input((None, None, 3))
    network =resnet()
    fpn_net=network.fpn_net(img_input)
    map_large, map_mid, map_small = network.resnet_50(is_share=True, inpt=img_input, is_FPN=True)
    roi_input = tf.keras.layers.Input(shape=(None, 4))
    classfier_large = network.fast_rcnn(map_large, roi_input, 8)
    classfier_mid = network.fast_rcnn(map_mid, roi_input, 16)
    classfier_small = network.fast_rcnn(map_small, roi_input, 32)
    model_classfier = tf.keras.models.Model(inputs=[img_input, roi_input],
                                         outputs=[classfier_large, classfier_mid, classfier_small])
    for layer in model_classfier.layers:
        layer._name = layer.name + "_base"
    fpn_net.load_weights('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/fpn_model8.h5', by_name=True)
    model_classfier.load_weights('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/final.h5',by_name=True)
    record=0
    right=0
    for img_info in data.roidb:
        record+=1
        path = os.path.join(cfg.IMAGESET, img_info.get("index") + ".jpg")
        print(img_info.get("index"))
        img = Image.open(path)
        img, box = data_gen.handle_origin_image(img, img_info.get('box'),img_info)
        img = np.array(img)[None]
        xclass_small, xloc_small, xclass_mid, xloc_mid, xclass_large, xloc_large = fpn_net.predict(img)
        rois_large = rpn_to_roi(xclass_large[0, :, :, :], xloc_large[0, :, :, :], 8)
        print("the number of rois is :" + str(len(rois_large)))
        rois_mid =rpn_to_roi(xclass_mid[0, :, :, :],
                                         xloc_mid[0, :, :, :], 16)
        print("the number of rois is :" + str(len(rois_mid)))
        rois_small = rpn_to_roi(xclass_small[0, :, :, :],
                                           xloc_small[0, :, :, :], 32)
        print("the number of rois is :" + str(len(rois_small)))
        rois = np.concatenate([rois_small, rois_large, rois_mid], axis=0)[:, :4]
        print("the number of rois is :" + str(len(rois)))
        rois_expend=np.expand_dims(rois,axis=0)
        label_record = []
        reg_record = []
        cls_confidence=[]
        roi_record=[]
        index_record=[]
        class_large,loc_large, class_mid,loc_mid, class_small,loc_small= model_classfier.predict([img,rois_expend])
        cls_score_large=np.squeeze(class_large)
        loc_reg_large=np.squeeze( loc_large)
        cls_score_mid = np.squeeze(class_mid)
        loc_reg_mid = np.squeeze(loc_mid)
        cls_score_small = np.squeeze(class_small)
        loc_reg_small = np.squeeze(loc_small)
        loc_reg_large=loc_reg_large/cfg.TRAIN.std_scaling
        loc_reg_mid=loc_reg_mid/cfg.TRAIN.std_scaling
        loc_reg_small=loc_reg_small/cfg.TRAIN.std_scaling
        for i in range(cls_score_large.shape[0]):
            w = rois[i][2] - rois[i][0]
            h = rois[i][3] - rois[i][1]
            roi = [rois[i][0] + w / 2, rois[i][1] + h / 2, w, h]
            if np.argmax(cls_score_large[i, :]) != cfg.NUM_CLASSES - 1 and max(cls_score_large[i, :]) > 0.8:
                label_record.append(cls_score_large[i, :])
                index=np.argmax(cls_score_large[i, :])
                index_record.append(index)
                roi_record.append(roi)
                roi_reg=cal_transform(roi,loc_reg_large[i][index*4:index*4+4])
                reg_record.append(roi_reg)
                cls_confidence.append(cls_score_large[i, index])
            if np.argmax(cls_score_mid[i, :]) != cfg.NUM_CLASSES - 1 and max(cls_score_mid[i, :]) > 0.8:
                label_record.append(cls_score_mid[i, :])
                index = np.argmax(cls_score_mid[i, :])
                index_record.append(index)
                roi_record.append(roi)
                roi_reg=cal_transform(roi,loc_reg_mid[i][index*4:index*4+4])
                reg_record.append(roi_reg)
                cls_confidence.append(cls_score_large[i, index])
            if np.argmax(cls_score_small[i, :]) != cfg.NUM_CLASSES - 1 and max(cls_score_small[i, :]) > 0.8:
                label_record.append(cls_score_small[i, :])
                index = np.argmax(cls_score_small[i, :])
                index_record.append(index)
                roi_record.append(roi)
                roi_reg=cal_transform(roi,loc_reg_small[i][index*4:index*4+4])
                reg_record.append(roi_reg)
                cls_confidence.append(cls_score_large[i, index])
        if len(reg_record)==0:
            continue
        reg_record=np.array(reg_record)
        cls_confidence=np.expand_dims(np.array(cls_confidence),axis=1)
        index_record=np.expand_dims(np.array(index_record),axis=1)
        np_im = np.squeeze(img)
        reg_record=np.concatenate((reg_record,cls_confidence,index_record),axis=1)
        gt_box=img_info.get('box')
        for cls_num in range(8):
            index=np.where(np.equal(reg_record[:,5],cls_num))[0]
            name=cfg.CLASSES_NAME[cls_num]
            if len(index)==0:
                continue
            boxes=nms(reg_record[index])
            for roi in boxes:
                np_im = debug.drawbox(np_im, (0, 0, 255), roi, 1,name)
        cv2.imwrite('tst' + img_info.get('index')+ '.png', np_im)
    print(str(right))
    print(str(record))

if __name__ == '__main__':
    if tf.test.is_gpu_available(
            cuda_only=False,
            min_cuda_compute_capability=None
    ):
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
    test()