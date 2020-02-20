import data_handler.imdb as imdb
import tensorflow.keras as keras
import tensorflow as tf
from tools.config import cfg
from tools import debug
import numpy as np
import tensorflow.keras.backend as backend
from rpn_network.rpn import rpn_loss_cls,rpn_reg_loss,rpn_net
from network.resnet import resnet
from data_handler.data_generator_rpn import data_generator_rpn
from rpn_network import roi_helper
import time
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

def cls_loss(y_true,y_pred):
   """
   The method to calculate the classification loss function for classifier
   """
   return keras.backend.sum(keras.losses.categorical_crossentropy(y_true,y_pred))

def reg_loss(y_true,y_pred):
    """
    THe method to calculate the regression loss for classifier
    """
    num_classes=cfg.NUM_CLASSES-1
    x = y_true[:, :, 4 *num_classes:] - y_pred
    x_abs = backend.abs(x)
    x_bool = backend.cast(backend.less_equal(x_abs, 1.0), 'float32')
    return backend.sum(y_true[:, :, :4 * num_classes] * ((x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))))/(backend.sum( 1e-4+y_true[:, :, :4 * num_classes])*0.25)

def train_frcnn():
    network=resnet()
    img_input=keras.layers.Input(shape=(None,None,3))
    roi_input=keras.layers.Input(shape=(None,4))
    feature_map=network.resnet_50(is_share=True,inpt=img_input)
    rpn_output = network.rpn_net(feature_map)
    rpn_model= rpn_net().get_model()
    classfier=network.fast_rcnn(feature_map,roi_input)
    model_classfier=keras.models.Model(inputs=[img_input,roi_input],outputs=classfier)
    for layer in model_classfier.layers:
        layer._name = layer.name + "_base"
    rpn_model.load_weights("/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/rpn_model.h5")
    model_classfier.load_weights('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/rpn_model.h5',by_name=True)
    comb_model = keras.models.Model([img_input, roi_input], [rpn_output,classfier])

    optimizer = keras.optimizers.SGD(lr=0.001/100, decay=0.0005, momentum=0.9)
    optimizer_classifier = keras.optimizers.SGD(lr=0.001/5, decay=0.0005, momentum=0.9)

    rpn_model.compile(optimizer=optimizer, loss=[rpn_loss_cls,rpn_reg_loss])
    model_classfier.compile(optimizer=optimizer_classifier,
                             loss=[cls_loss, reg_loss],
                             metrics=['accuracy'])
    comb_model.compile(optimizer='sgd', loss='mae')

    generator=data_generator_rpn()
    num_epochs = cfg.TRAIN.EPOCH
    epoch_length =generator.step_per_epoch
    loss_record=np.zeros((epoch_length,5))
    iter_num=0
    start_time = time.time()
    last_loss=0
    patient=0
    for epoch in range(num_epochs):
        progbar = tf.keras.utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # first 3 epoch is warm up.
        if epoch == 3:
            backend.set_value(rpn_model.optimizer.lr, 0.001)
            backend.set_value(model_classfier.optimizer.lr, 0.001)
        while True:
            rpn_x,rpn_y,img_info=next(generator.generator)
            print(img_info.get('index'))
            # loss_rpn = rpn_model.train_on_batch(rpn_x, rpn_y)
            predict_rpn = rpn_model.predict_on_batch(rpn_x)
            # rois is (xmin,ymin,xmax,ymax)
            rois,map_height,map_width=roi_helper.rpn_to_roi(predict_rpn[0],predict_rpn[1])
            print(rois)
            print(rois[:,4])
            rois=rois[:,:4]
            # debug.draw_roi(rois, generator, img_info)
            # f_rcnn_label=roi_helper.match_gt_box(rois, img_info,map_height,map_width)
            # if f_rcnn_label is None:
            #      continue
            # rois,cls_label,reg_label=f_rcnn_label
            # debug.draw_roi(rois, generator, img_info)
            # postive_label=np.where(cls_label[0,:,-1]==0)[0]
            # negtive_label=np.where(cls_label[0,:,-1]==1)[0]
            # if len(postive_label)>cfg.MAX_ROIS/2:
            #    selected_pos_label = np.random.choice(postive_label, cfg.MAX_ROIS/2, replace=False).tolist()
            # else:
            #    selected_pos_label=postive_label.tolist()
            # selected_neg_label=[]
            # try:
            #    selected_neg_label = np.random.choice(negtive_label, cfg.MAX_ROIS - len(selected_pos_label),
            #                                            replace=False).tolist()
            # except:
            #     if len(negtive_label)>0:
            #           selected_neg_label = np.random.choice(negtive_label, cfg.MAX_ROIS - len(selected_pos_label),
            #                                            replace=True).tolist()
            #     else:
            #         selected_neg_label = np.random.choice(postive_label, cfg.MAX_ROIS - len(selected_pos_label), replace=True).tolist()
            # # print(cls_label[:,selected_neg_label,:])
            # train_label=selected_pos_label+selected_neg_label
            # # print(cls_label[:, train_label, :])
            # loss_classfiy=model_classfier.train_on_batch([rpn_x,rois[:,train_label,:]],[cls_label[:,train_label,:],reg_label[:,train_label,:]])
            # loss_record[iter_num, 0] = loss_rpn[1]
            # loss_record[iter_num, 1] = loss_rpn[2]
            # loss_record[iter_num, 2] = loss_classfiy[1]
            # loss_record[iter_num, 3] = loss_classfiy[2]
            # loss_record[iter_num, 4] = loss_classfiy[3]
            # iter_num += 1
            # progbar.update(iter_num,
            #               [('rpn_cls', np.mean(loss_record[:iter_num, 0])), ('rpn_regr', np.mean(loss_record[:iter_num, 1])),
            #                ('detector_cls', np.mean(loss_record[:iter_num, 2])),
            #                ('detector_regr', np.mean(loss_record[:iter_num, 3])),
            #                ('accuracy', np.mean(loss_record[:iter_num, 4])),
            #                ("average number of objects", len(selected_pos_label))])
            # if iter_num == epoch_length-1:
            #    loss_rpn_cls = np.mean(loss_record[:, 0])
            #    loss_rpn_regr = np.mean(loss_record[:, 1])
            #    loss_class_cls = np.mean(loss_record[:, 2])
            #    loss_class_regr = np.mean(loss_record[:, 3])
            #    class_acc = np.mean(loss_record[:, 4])
            #    iter_num=0
            #    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
            #    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
            #    print('Loss RPN regression: {}'.format(loss_rpn_regr))
            #    print('Loss Detector classifier: {}'.format(loss_class_cls))
            #    print('Loss Detector regression: {}'.format(loss_class_regr))
            #    print('Elapsed time: {}'.format(time.time() - start_time))
            #    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            #    start_time = time.time()
            #    if curr_loss>last_loss:
            #         patient+=1
            #    if curr_loss<last_loss:
            #         patient=0
            #    if patient==4:
            #         comb_model.save_weights("final.h5")
            #         return
            #    break

    comb_model.save_weights("final.h5")
def test():
    generator = data_generator_rpn()
    for epoch in range(1):
        rpn_x, rpn_y, img_info = next(generator.generator)
        debug.check_data_generator(generator, rpn_y, img_info)


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
