import os
import tensorflow.keras as keras
import tensorflow as tf
from data_handler.imdb import imdb
from network.resnet import resnet
from tensorflow.keras import backend as K
from data_handler.data_generator_rpn import data_generator_rpn
from tools.config import cfg
import numpy as np
import tensorflow_addons as tfa
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# ////////////////////////////////////////////////////////////////////
class rpn_net():
    """
    The class that build rpn network and training.
    """
    def __init__(self):
        resnet_base = resnet()
        self.input_layer = keras.layers.Input(shape=(None, None, 3))
        self.feature_map = resnet_base.resnet_50(is_share=True, inpt=self.input_layer)
        self.rpn_output = resnet_base.rpn_net(self.feature_map)

    def get_model(self):
        rpn_model = keras.models.Model(inputs=self.input_layer, outputs=self.rpn_output)
        for layer in rpn_model.layers:
            layer._name = layer.name + "_base"
        rpn_model.summary()
        return rpn_model

    def train_rpn(self):
        rpn_model=self.get_model()
        if os.path.exists(cfg.RESNET_PATH):
            print("Successful loading the weight from pre training model.")
            rpn_model.load_weights(cfg.RESNET_PATH,by_name=True)
        rpn_model.compile(optimizer=keras.optimizers.Adam(lr=1e-5), loss=[rpn_loss_cls, rpn_reg_loss])
        # feed data.
        data_generator = data_generator_rpn()
        early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=3, verbose=2)
        rpn_model.fit(x=data_generator.generator, steps_per_epoch=data_generator.step_per_epoch,
                                epochs=100,
                                verbose=1, shuffle=True, initial_epoch=0, callbacks=[early_stop])
        rpn_model.save_weights("rpn_model.h5")


# def focal_loss(y_true,y_pred):
#     alpha=0.25
#     gamma=2
#     zeros=tf.zeros_like(y_pred,dtype=y_pred.dtype)
#     postive=tf.where(y_true>zeros,y_true-y_pred,zeros)
#     negtive=tf.where(y_true>zeros,zeros,y_pred)
#     f1_loss=-alpha*(postive**gamma)*tf.math.log(tf.clip_by_value(y_pred,1e-8,1-(1e-8)))-(1-alpha)*(negtive**gamma)*tf.math.log(tf.clip_by_value(1.0-y_pred,1e-8,1-(1e-8)))
#     return f1_loss

def rpn_loss_cls(y_true, y_pred):
    """
       The method to calculate classification loss.
       :param y_true: (1,width,height,anchor_number*2+anchor_number*2)
       :param y_pred: (1,width,height,anchor_number*2)
       :return: the loss of classification
       """
    a=K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :,2*cfg.TRAIN.ANCHOR_NUM:])
    # loss_function=tfa.losses.SigmoidFocalCrossEntropy()
    # a=focal_loss(y_pred[:, :, :, :], y_true[:, :, :,2*cfg.TRAIN.ANCHOR_NUM:])
    # a=tf.keras.losses.SigmoidFocalCrossEntropy(y_pred[:, :, :, :], y_true[:, :, :,2*cfg.TRAIN.ANCHOR_NUM:])
    return K.sum(y_true[:, :, :, :2*cfg.TRAIN.ANCHOR_NUM] * a) /(K.sum(
         1e-4 +y_true[:, :, :, :2*cfg.TRAIN.ANCHOR_NUM])*0.5)


def rpn_reg_loss(y_true, y_pred):
    """
    The method to calculate rpn regression loss.
    :param y_true: (1,width,height,anchor_number*4+anchor_number*4)
    :param y_pred: (1,width,height,anchor_number*4)
    :return: the loss of regression.
    """
    x = y_true[:, :, :, 4*cfg.TRAIN.ANCHOR_NUM:] - y_pred
    x_abs = K.abs(x)
    x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
    return K.sum(y_true[:, :, :, :4*cfg.TRAIN.ANCHOR_NUM] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) \
           / (K.sum( 1e-4+y_true[:, :, :, :4*cfg.TRAIN.ANCHOR_NUM])*0.25)


if __name__ == '__main__':
    if tf.test.is_gpu_available(
                cuda_only=False,
                min_cuda_compute_capability=None
        ):
            config = ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.7
            config.gpu_options.allow_growth = True
            session = InteractiveSession(config=config)
            rpn_ = rpn_net()
            rpn_.train_rpn()
