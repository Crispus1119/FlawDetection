import tensorflow as tf
from data_handler.imdb import imdb
from network.resnet import resnet
from PIL import Image as Image
import os
from rpn_network.roi_helper import rpn_to_roi,handle_roi_fm,nms,cal_transform
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

def test():
    data = imdb('test')
    data_gen = data_generator_rpn()
    network = resnet()
    batch_size = cfg.TEST.BATCH_SIZE
    img_input = tf.keras.layers.Input((None, None, 3))
    feature_map = network.resnet_50(is_share=True, inpt=img_input)
    roi_input = tf.keras.layers.Input(shape=(None, 4))
    classfier = network.fast_rcnn(feature_map, roi_input)
    model_classfier = tf.keras.models.Model(inputs=[img_input, roi_input], outputs=classfier)
    rpn_model = rpn_net().get_model()
    for layer in model_classfier.layers:
        layer._name = layer.name + "_base"
    rpn_model.load_weights('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/rpn_model.h5', by_name=True)
    model_classfier.load_weights('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/final.h5',by_name=True)
    right = 0
    for img_info in data.roidb:
        if img_info.get('index')[:2]!='FB':
            continue
        path = os.path.join(cfg.IMAGESET, img_info.get("index") + ".jpg")
        print(img_info.get("index"))
        img = Image.open(path)
        img, box = data_gen.handle_origin_image(img, img_info.get('box'))
        img = np.array(img)[None]
        predict_rpn = rpn_model.predict_on_batch(img)
        # rois : [xmin,ymin,xmax,ymax]
        rois,map_height,map_width = rpn_to_roi(predict_rpn[0], predict_rpn[1])
        rois=np.array(handle_roi_fm(rois,map_height,map_width))
        # rois : [xmin, ymin, w, h]
        # debug.draw_roi(rois,data_gen,img_info)
        label_record = []
        reg_record = []
        roi_record=[]
        if len(rois) < batch_size:
            continue
        if len(rois) % batch_size != 0:
            index = np.random.choice(len(rois), len(rois) % batch_size, replace=True)
            rois = np.append(rois, rois[index, :],axis=0)
        for batch in range(int(len(rois) / batch_size)):
            test_data = np.expand_dims(rois[batch * batch_size:batch_size * (batch + 1), :], axis=0)
            cls_score, loc_reg = model_classfier.predict_on_batch([img, test_data])
            cls_score=np.squeeze(cls_score)
            loc_reg=np.squeeze(loc_reg)
            for i in range(cls_score.shape[0]):
                if np.argmax(cls_score[i, :]) == cfg.NUM_CLASSES - 1 or max(cls_score[i, :]) < 0.5:
                    continue
                label_record.append(cls_score[i, :])
                roi_record.append(test_data[0,i,:]*cfg.TRAIN.RPN_STRIDE)
                reg_record.append(loc_reg[i, :]/cfg.TRAIN.classifier_regr_std)

        result=get_result(roi_record,reg_record,label_record)


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