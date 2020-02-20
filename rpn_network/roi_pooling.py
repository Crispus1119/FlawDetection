import tensorflow.keras as keras
import tensorflow as tf
from tools.config import cfg
from tensorflow.compat.v1 import image
class RoiPooling(tf.keras.layers.Layer):
    """
    @:param pool_size: the roi pooling window size
    @:param num_rois : the number of rois that go through roi_pooling layer
    """
    def __init__(self,**kwargs):
        self.pool_size=cfg.ROI_POOLING_SIZE
        self.num_rois=cfg.MAX_ROIS
        super(RoiPooling, self).__init__(**kwargs)

    # Create a trainable weight variable for this layer,but roi pooling do not have weight.
    def build(self, input_shape):
        self.nb_channles = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return (1,self.num_rois,self.pool_size,self.pool_size,self.nb_channles)

    def call(self, inputs, **kwargs):
        feature_map=inputs[0]
        rois=inputs[1]
        roi_output=[]
        record=0
        for roi_index in range(self.num_rois):
            record+=1
            xmin=rois[0,roi_index,0]
            ymin=rois[0,roi_index,1]
            w=rois[0,roi_index,2]
            h=rois[0,roi_index,3]
            w = tf.cast(w, 'int32')
            h = tf.cast(h, 'int32')
            x = tf.cast(xmin, 'int32')
            y = tf.cast(ymin, 'int32')
            roi_map=image.resize_images(feature_map[:, y:y+h, x:x+w, :],(self.pool_size,self.pool_size))
            roi_output.append(roi_map)
        # transform list to tensor(4,H,W,channal).
        result=tf.concat(roi_output,axis=0)
        # tf.print(result.shape)
        result=tf.expand_dims(result,axis=0)
        # tf.print(result.shape)
        return result
