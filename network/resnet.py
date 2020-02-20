import tensorflow.keras as keras
import tensorflow as tf
import os,sys,math
sys.path.append('../')
import data_handler.imdb as imdb
from tools.config import cfg
from data_handler.data_generator import data_generator
from rpn_network.roi_pooling import RoiPooling
from tensorflow.keras import backend
from tensorflow.keras.layers import TimeDistributed,Conv2D

################### restrict memory
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
# backend.set_session(tf.Session(config=config))

class resnet():
    def __init__(self):
        self.channal=3
        self.width = cfg.TRAIN.RESIZE_ROIDB
        self.height = cfg.TRAIN.RESIZE_ROIDB
        self.cls = cfg.NUM_CLASSES

    def resnet34(self):
        Input=keras.layers.Input(shape=(self.width,self.height,self.channal))
        x= keras.layers.ZeroPadding2D(padding=(3,3))(Input)
        # 230*230*3
        #conv1
        x = self.conv2d_bn(x,64,(7,7),(2,2),'valid')
        #112*112
        #conv2
        x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        #56*56
        x = self.identify_block(x,64,(3,3),(1,1))(x)
        x = self.identify_block(x, 64, (3, 3), (1, 1))(x)
        x = self.identify_block(x, 64, (3, 3), (1, 1))(x)
        #56*56
        #conv3
        x= self.identify_block(x,128,(3,3),(2,2),True)
        x = self.identify_block(x, 128, (3, 3))(x)
        x = self.identify_block(x, 128, (3, 3))(x)
        x = self.identify_block(x, 128, (3, 3))(x)
        #28*28
        #conv4
        x= self.identify_block(x,256,(3,3),(2,2),True)
        x = self.identify_block(x, 256, (3, 3))(x)
        x = self.identify_block(x, 256, (3, 3))(x)
        x = self.identify_block(x, 256, (3, 3))(x)
        x = self.identify_block(x, 256, (3, 3))(x)
        x = self.identify_block(x, 256, (3, 3))(x)
        #14*14
        #conv5
        x = self.identify_block(x, 512, (3, 3), (2, 2), True)
        x = self.identify_block(x, 512, (3, 3))(x)
        x = self.identify_block(x, 512, (3, 3))(x)
        #7*7
        x=keras.layers.AveragePooling2D(pool_size=(7,7))(x)
        x=keras.layers.Flatten()(x)
        x=keras.layers.Dense(self.cls,activation='softmax')(x)
        model=keras.models.Model(inputs=Input, outputs=x)
        return model

    def resnet_50(self,is_share=False,inpt=None):
        # 224*224*3
        if not is_share:
           inpt = keras.layers.Input(shape=(self.width, self.height, self.channal))
        # 230*230*3
        x = keras.layers.ZeroPadding2D((3, 3))(inpt)
        # 230-7=223/2+1=(112,112,3) If It's not divisible, rounded down
        x = self.conv2d_bn(x,64,(7, 7), (2, 2), padding='valid')
        # (56,56,64)
        x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
        # conv2_x(56,56,256)
        x = self.bottleneck_Block(x, [64, 64, 256],strides=(1, 1), with_shortcut=True)
        x = self.bottleneck_Block(x, [64, 64, 256])
        x = self.bottleneck_Block(x, [64, 64, 256])

        # conv3_x(28*28*512)
        x = self.bottleneck_Block(x, [128, 128, 512], strides=(2, 2), with_shortcut=True)
        x = self.bottleneck_Block(x, [128, 128, 512])
        x = self.bottleneck_Block(x, [128, 128, 512])
        x = self.bottleneck_Block(x, [128, 128, 512])

        # conv4_x(14*14*1024
        x = self.bottleneck_Block(x,[256, 256, 1024], strides=(2, 2), with_shortcut=True)
        x = self.bottleneck_Block(x, [256, 256, 1024])
        x = self.bottleneck_Block(x, [256, 256, 1024])
        x = self.bottleneck_Block(x, [256, 256, 1024])
        x = self.bottleneck_Block(x, [256, 256, 1024])
        if(is_share):
            return x
        # else:
        #     # conv5_x(7*7*2048)
        #     x = self.bottleneck_Block(x,[512, 512, 2048], strides=(2, 2), with_shortcut=True)
        #     x = self.bottleneck_Block(x,[512, 512, 2048])
        #     x = self.bottleneck_Block(x,[512, 512, 2048])
        #
        #     x = keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
        #     x = keras.layers.Flatten()(x)
        #     x = keras.layers.Dense(self.cls, activation='softmax')(x)
        #     model = keras.models.Model(inputs=inpt, outputs=x)
        #     return model

    def rpn_net(self,x):
        rpn = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu', name="rpn_net")(x)
        x_class = keras.layers.Conv2D(2*cfg.TRAIN.ANCHOR_NUM, kernel_size=(1, 1), activation='sigmoid',
                                      name="rpn_class")(rpn)
        x_loc = keras.layers.Conv2D(4 * cfg.TRAIN.ANCHOR_NUM, kernel_size=(1, 1), activation="linear", name="rpn_reg")(
            rpn)
        return x_class,x_loc


    def cal_fm_size(self, width, height):
        """
        The method to calculate feature map size.
        :param width: the origin image width
        :param height: the origin image height
        :return: the feature map size.
        """
        def calculator(length):
            # zero padding 3
            length=length+3
            stride=2
            filter_size=[7,3,1,1]
            for i in filter_size:
                if i==3:
                    # pooling layer is round to up value
                    length =math.ceil((length - i + stride)/stride)
                else:
                    length = (length - i + stride) // stride
            return length
        return  calculator(width),calculator(height)

    def bottleneck_Block(self,input,filters,strides=(1,1),with_shortcut=False):
        k1, k2, k3 = filters
        x = self.conv2d_bn(input,k1,(1,1), strides, padding='same')
        x = self.conv2d_bn(x, k2, (3,3), padding='same')
        x = self.conv2d_bn(x, k3, (1,1), padding='same')
        if with_shortcut:
            shortcut = self.conv2d_bn(input, k3,(1,1),strides)
            x = keras.layers.add([x, shortcut])
            return x
        else:
            x = keras.layers.add([x, input])
            return x

    def identify_block(self,inpt,filters,kernel_size, stride=(1,1),with_shortcut=False):
        x=self.conv2d_bn(inpt,filters,kernel_size,stride=stride,padding='same')
        # this layer stride is 1*1
        x=self.conv2d_bn(x,filters,kernel_size,padding='same')
        if with_shortcut:
            shortcut=self.conv2d_bn(inpt,filters,kernel_size,stride)
            x= keras.layers.add([x,shortcut])
            return x

        return keras.layers.add([x,inpt])

    def conv2d_bn(self,x,filter,kennal_size,stride=(1,1) ,padding='same'):
        x = keras.layers.Conv2D(filters=filter,kernel_size=kennal_size,strides=stride,padding=padding,activation='relu')(x)
        # axis is point to the channel value
        x= keras.layers.BatchNormalization(axis=3)(x)
        return x

    def fast_rcnn(self,feature_map,input_rois,):
        class_num=cfg.NUM_CLASSES
        # roi_pooling class instance is callable because of __call__.
        # roi_output shape is [1,None,14,14,1024]
        roi_output=RoiPooling()([feature_map,input_rois])
        x=TimeDistributed(keras.layers.AveragePooling2D(7,7))(roi_output)
        x=TimeDistributed(keras.layers.Flatten())(x)
        x=TimeDistributed(keras.layers.Dense(4096, activation='relu'))(x)
        x=TimeDistributed(keras.layers.Dropout(0.5))(x)
        x = TimeDistributed(keras.layers.Dense(4096, activation='relu'))(x)
        x = TimeDistributed(keras.layers.Dropout(0.5))(x)
        out_class = TimeDistributed(keras.layers.Dense(class_num, activation='softmax', kernel_initializer='zero'))(x)
        out_reg=TimeDistributed(keras.layers.Dense(4 * (class_num-1), activation='linear', kernel_initializer='zero'))(x)
        return [out_class,out_reg]



def acc_top2(y_true, y_pred):
    # the method to define metrics.
    # if top 2 score of class contains right answer, it is truth
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)

def check_print():
    # Create a Keras Model
    print("building the model...........")
    model = resnet().resnet_50()
    model.summary()
    for layer in model.layers:
        layer._name = layer.name + "_base"
    if os.path.exists('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/capstone/resnet_50_weughts.h5'):
        print("loading weight......")
        model.load_weights('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/capstone/resnet_50_weughts.h5')
        print("successfully loading weight.")
    # Save a PNG of the Model Build
    # THERE is  the problem to install this .
    # keras.utils.plot_model(model, to_file='resnet.png')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',acc_top2])
    print ('Model Compiled')
    # be careful with  steps_per_epoch
    generator=data_generator(8)
    early_stop=keras.callbacks.EarlyStopping(monitor="val_loss",patience=15,verbose=2)
    model.fit_generator(generator.generator,generator.steps_per_epoch,epochs=200,
                    verbose=1,shuffle=True,initial_epoch=0,validation_data=generator.get_val_data,callbacks=[early_stop])

    return model


if __name__ == '__main__':
    i= imdb.imdb("train")
    model = check_print()
    # for layer in model.layers:
    #     layer._name = layer.name + "_base"
    model.save_weights("resnet_50_weughts.h5")
