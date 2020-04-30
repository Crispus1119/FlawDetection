import tensorflow.keras as keras
import tensorflow as tf
import os,sys,math
sys.path.append('../')
import data_handler.imdb as imdb
from tools.config import cfg
from data_handler.data_generator import data_generator
from rpn_network.roi_pooling import RoiPooling
from tensorflow.keras.layers import TimeDistributed,Conv2D
from tensorflow.keras import regularizers

class resnet():
    def __init__(self):
        self.channal=3
        self.width = cfg.TRAIN.RESIZE_ROIDB
        self.height = cfg.TRAIN.RESIZE_ROIDB
        self.cls = cfg.NUM_CLASSES


    def resnet_50(self,is_share=False,inpt=None,is_FPN=False):
        # 224*224*3
        if not is_share:
           inpt = keras.layers.Input(shape=(self.width, self.height, self.channal))
        # 224*224*3
        x = self.conv2d_bn(inpt,64,(7, 7), (2, 2))
        # (112,112,64)
        x = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),padding='same')(x)
        # conv2_x(56,56,64)
        x = self.bottleneck_Block(x, [64, 64, 256],strides=(1, 1), with_shortcut=True)
        x = self.bottleneck_Block(x, [64, 64, 256])
        x = self.bottleneck_Block(x, [64, 64, 256])
        # (56, 56, 256)
        x = self.bottleneck_Block(x, [128, 128, 512], strides=(2, 2), with_shortcut=True)
        x = self.bottleneck_Block(x, [128, 128, 512])
        x = self.bottleneck_Block(x, [128, 128, 512])
        x_large = self.bottleneck_Block(x, [128, 128, 512])
        # conv3_x(28*28*512)
        x = self.bottleneck_Block(x_large,[256, 256, 1024], strides=(2, 2), with_shortcut=True)
        x = self.bottleneck_Block(x, [256, 256, 1024])
        x = self.bottleneck_Block(x, [256, 256, 1024])
        x = self.bottleneck_Block(x, [256, 256, 1024])
        x_mid = self.bottleneck_Block(x, [256, 256, 1024])
        # conv4_x(14*14*1024)
        x = self.bottleneck_Block(x_mid, [512, 512, 2048], strides=(2, 2), with_shortcut=True)
        x = self.bottleneck_Block(x, [512, 512, 2048])
        x_small = self.bottleneck_Block(x, [512, 512, 2048])
        # conv4_x(7*7*2048)
        if is_FPN:
            return x_large,x_mid,x_small
        if(is_share):
            return x_small
        else:
            # conv5_x(7*7*2048)
            x = self.bottleneck_Block(x,[512, 512, 2048], strides=(2, 2), with_shortcut=True)
            x = self.bottleneck_Block(x,[512, 512, 2048])
            x = self.bottleneck_Block(x,[512, 512, 2048])

            x = keras.layers.AveragePooling2D(pool_size=(7, 7))(x)
            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(self.cls, activation='softmax',kernel_regularizer=regularizers.l2(0.0001))(x)
            model = keras.models.Model(inputs=inpt, outputs=x)
            return model

    def fpn_fm(self,x_large,x_mid,x_small):
        """
        The method to generate the feature map to predict.
        """
        xsmall_reduce=keras.layers.Conv2D(filters=256,kernel_size=(1,1),activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x_small)
        xmid_reduce = keras.layers.Conv2D(filters=256, kernel_size=(1, 1), activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x_mid)
        xlarge_reduce=keras.layers.Conv2D(filters=256,kernel_size=(1,1),activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x_large)
        xsmall_up=keras.layers.UpSampling2D((2,2),interpolation='nearest')(xsmall_reduce)
        mid_predict=keras.layers.Add()([xmid_reduce,xsmall_up])
        xmid_up=keras.layers.UpSampling2D((2,2),interpolation='nearest')(mid_predict)
        large_predict=keras.layers.Add()([xlarge_reduce,xmid_up])
        return xsmall_reduce,mid_predict,large_predict

    def fpn_output(self,input_layer):
        x_large, x_mid, x_small = self.resnet_50(is_share=True, inpt=input_layer, is_FPN=True)
        small_predict, mid_predict, large_predict = self.fpn_fm(x_large, x_mid, x_small)
        xclass_small, xloc_small = self.rpn_net(small_predict, "small")
        xclass_mid, xloc_mid = self.rpn_net(mid_predict, "mid")
        xclass_large, xloc_large = self.rpn_net(large_predict, "large")
        return xclass_large,xloc_large,xclass_mid, xloc_mid,xclass_small, xloc_small

    def fpn_net(self,inputLayor=None):
        input_layer = keras.layers.Input(shape=(None, None, 3))
        if inputLayor is not None:
            input_layer=inputLayor
        xclass_large, xloc_large, xclass_mid, xloc_mid, xclass_small, xloc_small=self.fpn_output(input_layer)
        rpn_model=keras.models.Model(inputs=input_layer,outputs=[xclass_small,xloc_small,xclass_mid,xloc_mid,xclass_large,xloc_large])
        for layer in rpn_model.layers:
            layer._name = layer.name + "_base"
        # rpn_model.summary()
        return rpn_model

    def rpn_net(self,x,extra_name):
        rpn = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name="rpn_net"+extra_name)(x)
        x_class = keras.layers.Conv2D(2*cfg.TRAIN.ANCHOR_NUM, kernel_size=(1, 1), activation='sigmoid',name="rpn_class"+extra_name)(rpn)
        x_loc = keras.layers.Conv2D(4 * cfg.TRAIN.ANCHOR_NUM, kernel_size=(1, 1), activation="linear",name="rpn_reg"+extra_name)(rpn)
        return x_class,x_loc


    def cal_fm_size(self, width, height,isFPN=False):
        """
        The method to calculate feature map size.
        :param width: the origin image width
        :param height: the origin image height
        :return: the feature map size.
        """
        def calculator(length,isFPN=False):
            stride=2
            filter_size=[7,3,1,1,1]
            result=[]
            for i in filter_size:
                if i==3:
                    length = math.ceil(length//stride)
                else:
                   length = length //stride
                if i==1:
                    result.append(length)
            if isFPN:
                return result
            else:return length
        return  calculator(width,isFPN),calculator(height,isFPN)

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
        x = keras.layers.Conv2D(filters=filter,kernel_size=kennal_size,strides=stride,padding=padding,activation='relu',kernel_regularizer=regularizers.l2(0.0001))(x)
        # axis is point to the channel value
        x= keras.layers.BatchNormalization(axis=3)(x)
        return x

    def fast_rcnn(self,feature_map,input_rois,rpn_stride):
        class_num=cfg.NUM_CLASSES
        # roi_pooling class instance is callable because of __call__.
        # roi_output shape is [1,None,14,14,1024]
        input_rois=input_rois/rpn_stride
        roi_output=RoiPooling()([feature_map,input_rois])
        x=TimeDistributed(keras.layers.AveragePooling2D(7,7))(roi_output)
        x=TimeDistributed(keras.layers.Flatten())(x)
        x=TimeDistributed(keras.layers.Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))(x)
        x=TimeDistributed(keras.layers.Dropout(0.5))(x)
        x = TimeDistributed(keras.layers.Dense(4096, activation='relu',kernel_regularizer=regularizers.l2(0.0001)))(x)
        x = TimeDistributed(keras.layers.Dropout(0.5))(x)
        out_class = TimeDistributed(keras.layers.Dense(class_num, activation='softmax', kernel_initializer='zero',kernel_regularizer=regularizers.l2(0.0001)))(x)
        out_reg=TimeDistributed(keras.layers.Dense(4 * (class_num-1), activation='linear', kernel_initializer='zero',kernel_regularizer=regularizers.l2(0.0001)))(x)
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
