import data_handler.imdb as imdb
import tensorflow.keras as keras
import tensorflow as tf
from tools.config import cfg
from tools import debug
import numpy as np
import tensorflow.keras.backend as backend
from rpn_network.rpn import rpn_loss_cls,rpn_reg_loss,rpn_net
from network.resnet import resnet
from data_handler.data_generator_fpn import data_generator
from rpn_network import roi_helper
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def cls_loss(y_true,y_pred):
   """
   The method to calculate the classification loss function for classifier
    y_true is [1,rois_num,8*class_num]
    y_pred is [1,rois_num,4*class_num]
   """
   return keras.backend.sum(keras.losses.categorical_crossentropy(y_true,y_pred))

def reg_loss(y_true,y_pred):
    """
    The method to calculate the regression loss for classifier
    y_true is [1,rois_num,class_num]
    y_pred is [1,rois_num,class_num]
    """
    num_classes=cfg.NUM_CLASSES-1
    x = y_true[:, :, 4 *num_classes:] - y_pred
    x_abs = backend.abs(x)
    x_bool = backend.cast(backend.less_equal(x_abs, 1.0), 'float32')
    return backend.sum(y_true[:, :, :4 * num_classes] * ((x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))))/(backend.sum( 1e-4+y_true[:, :, :4 * num_classes])*0.25)


def select_sample(cls_label):
    postive_label = np.where(cls_label[:, -1] == 0)[0]
    negtive_label = np.where(cls_label[:, -1] == 1)[0]
    if len(postive_label) > cfg.MAX_ROIS/2:
        selected_pos_label = np.random.choice(postive_label, int(cfg.MAX_ROIS/ 2), replace=False).tolist()
    else:
        selected_pos_label = postive_label.tolist()
    try:
        selected_neg_label = np.random.choice(negtive_label, int(cfg.MAX_ROIS - len(selected_pos_label)),
                                                   replace=False).tolist()
    except:
        if len(negtive_label)>0:
              selected_neg_label = np.random.choice(negtive_label, int(cfg.MAX_ROIS - len(selected_pos_label)),
                                               replace=True).tolist()
        else:
            selected_neg_label = np.random.choice(postive_label, int(cfg.MAX_ROIS - len(selected_pos_label)), replace=True).tolist()
    train_label = selected_pos_label + selected_neg_label
    return train_label,selected_pos_label

def train_frcnn():
    network=resnet()
    img_input=keras.layers.Input(shape=(None,None,3))
    roi_input=keras.layers.Input(shape=(None,4))
    fpn_model=network.fpn_net(img_input)
    map_large, map_mid, map_small=network.resnet_50(is_share=True, inpt=img_input,is_FPN=True)
    xclass_large, xloc_large, xclass_mid, xloc_mid, xclass_small, xloc_small=network.fpn_output(img_input)
    classfier_large=network.fast_rcnn(map_large,roi_input,8)
    classfier_mid = network.fast_rcnn(map_mid, roi_input,16)
    classfier_small = network.fast_rcnn(map_small, roi_input,32)
    model_classfier=keras.models.Model(inputs=[img_input,roi_input],outputs=[classfier_large,classfier_mid,classfier_small])
    for layer in model_classfier.layers:
        layer._name = layer.name + "_base"
    fpn_model.load_weights("/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/fpn_model8.h5")
    model_classfier.load_weights('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/fpn_model8.h5',by_name=True)
    comb_model = keras.models.Model([img_input, roi_input], [xclass_large, xloc_large, xclass_mid, xloc_mid, xclass_small, xloc_small,classfier_large,classfier_mid,classfier_small])
    comb_model.compile(optimizer='sgd', loss='mae')
    optimizer = keras.optimizers.SGD(lr=0.01, decay=0.0005, momentum=0.9)
    optimizer_classifier = keras.optimizers.SGD(lr=0.001/5, decay=0.0005, momentum=0.9)
    fpn_model.compile(optimizer=optimizer, loss=[rpn_loss_cls,rpn_reg_loss,rpn_loss_cls,rpn_reg_loss,rpn_loss_cls,rpn_reg_loss])

    model_classfier.compile(optimizer=optimizer_classifier,
                             loss=[cls_loss, reg_loss,cls_loss, reg_loss,cls_loss, reg_loss],
                             metrics=['accuracy'])

    comb_model.compile(optimizer='sgd', loss='mae')
    generator=data_generator()
    num_epochs = 5
    epoch_length =generator.step_per_epoch
    data_gene=generator.g
    for epoch in range(num_epochs):
        loss_record = np.zeros((epoch_length, 6))
        batch_num = 0
        progbar = tf.keras.utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # first 3 epoch is warm up.
        if epoch == 3:
            backend.set_value(fpn_model.optimizer.lr, 0.001)
            backend.set_value(model_classfier.optimizer.lr, 0.001)
        while True:
            rpn_x,rpn_y,img_info=next(data_gene)
            # loss_rpn = fpn_model.train_on_batch(rpn_x, rpn_y)
            roi_data=np.zeros((8,cfg.MAX_ROIS,4),dtype=np.float32)
            classfiy_data = np.zeros((8, cfg.MAX_ROIS,cfg.NUM_CLASSES), dtype=np.float32)
            reg_data=np.zeros((8, cfg.MAX_ROIS, 8 *(cfg.NUM_CLASSES-1)))
            xclass_small, xloc_small, xclass_mid, xloc_mid, xclass_large, xloc_large= fpn_model.predict_on_batch(rpn_x)
            record=[]
            avaliabel=[]
            for batch_index in range(8):
                 rois_large=roi_helper.rpn_to_roi(xclass_large[batch_index,:,:,:],xloc_large[batch_index,:,:,:],8)
                 rois_mid = roi_helper.rpn_to_roi(xclass_mid[batch_index, :, :, :],
                                                    xloc_mid[batch_index, :, :, :], 16)
                 rois_small= roi_helper.rpn_to_roi(xclass_small[batch_index, :, :, :],
                                                  xloc_small[batch_index, :, :, :], 32)
                 rois=np.concatenate([rois_small,rois_large,rois_mid],axis=0)[:,:4]
                 rois=rois.astype(np.float32)
                 frcnn_label=roi_helper.match_gt_box(rois, img_info[batch_index])
                 if frcnn_label is None:
                     record.append(batch_index)
                     continue
                 rois,cls_label,reg_label=frcnn_label
                 avaliabel.append(batch_index)
                 train_label,selected_pos_label=select_sample(cls_label)
                 # debug.draw_roi(rois[selected_pos_label], img_info[batch_index],rpn_x[batch_index])
                 roi_data[batch_index]=rois[train_label]
                 classfiy_data[batch_index]=cls_label[train_label]
                 reg_data[batch_index]=reg_label[train_label]
            for batch_index in record:
                index=avaliabel[0]
                roi_data[batch_index] = roi_data[index]
                classfiy_data[batch_index] =classfiy_data[index]
                reg_data[batch_index] = reg_data[index]
            loss_classfiy0=model_classfier.train_on_batch([rpn_x[:2], roi_data[:2]],
                                                           [classfiy_data[:2], reg_data[:2], classfiy_data[:2],
                                                            reg_data[:2], classfiy_data[:2], reg_data[:2]])
            loss_classfiy1 = model_classfier.train_on_batch([rpn_x[2:4], roi_data[2:4]],
                                                           [classfiy_data[2:4], reg_data[2:4], classfiy_data[2:4],
                                                            reg_data[2:4], classfiy_data[2:4], reg_data[2:4]])
            loss_classfiy2 = model_classfier.train_on_batch([rpn_x[4:6], roi_data[4:6]],
                                                           [classfiy_data[4:6], reg_data[4:6], classfiy_data[4:6],
                                                            reg_data[4:6], classfiy_data[4:6], reg_data[4:6]])
            loss_classfiy3= model_classfier.train_on_batch([rpn_x[6:8], roi_data[6:8]],
                                                           [classfiy_data[6:8], reg_data[6:8], classfiy_data[6:8],
                                                            reg_data[6:8], classfiy_data[6:8], reg_data[6:8]])
            loss_record[batch_num, 0] = loss_classfiy0[0]
            loss_record[batch_num, 1] = loss_classfiy0[1]
            progbar.update(batch_num,
                          [('rpn', np.mean(loss_record[:batch_num+1, 0])), ('fast rcnn : ', np.mean(loss_record[:batch_num+1, 1])),
                           ("average number of objects", len(selected_pos_label))])
            batch_num += 1
            if batch_num >=epoch_length:
               break
    model_classfier.save_weights("final.h5")
    comb_model.save_weights("finalcomb.h5")
def test():
    train_frcnn()
    # generator = data_generator_rpn()
    # for epoch in range(1):
    #     rpn_x, rpn_y, img_info = next(generator.generator)
    #     debug.check_data_generator(generator, rpn_y, img_info)


if __name__ == '__main__':
    if tf.test.is_gpu_available(
            cuda_only=False,
            min_cuda_compute_capability=None
    ):
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.7
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        train_frcnn()
