from network.resnet import resnet
import os
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from data_handler.data_generator_fpn import data_generator
from tools.config import cfg
from rpn_network.rpn import rpn_reg_loss,rpn_loss_cls
def train_fpn():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        net=resnet()
        rpn_model=net.fpn_net()
        if os.path.exists(cfg.RESNET_PATH):
            print("Successful loading the weight from pre training model.")
            rpn_model.load_weights(cfg.RESNET_PATH, by_name=True)
        optimizer=tf.keras.optimizers.SGD(lr=0.02,momentum=0.9)
        rpn_model.compile(optimizer=optimizer,loss=[rpn_loss_cls,rpn_reg_loss,rpn_loss_cls,rpn_reg_loss,rpn_loss_cls,rpn_reg_loss])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, verbose=2)
        gen=data_generator()
    rpn_model.fit(x=gen.g,steps_per_epoch=gen.step_per_epoch,epochs=10,
                                verbose=1, shuffle=True, initial_epoch=0, callbacks=[early_stop],max_queue_size=32,workers=8,use_multiprocessing=True)
    rpn_model.save_weights("fpn_model8.h5")
if __name__ == '__main__':
    if tf.test.is_gpu_available(
            cuda_only=False,
            min_cuda_compute_capability=None
    ):
        config = ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        train_fpn()