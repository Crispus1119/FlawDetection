import tensorflow.keras as keras
import tensorflow as tf
from tools.config import cfg
from tensorflow.compat.v1 import image
class RoiPooling(tf.keras.layers.Layer):
    """ Implements Region Of Interest Max Pooling
           for channel-first images and relative bounding box coordinates

           # Constructor parameters
               pooled_height, pooled_width (int) --
                 specify height and width of layer outputs

           Shape of inputs
               [(batch_size, pooled_height, pooled_width, n_channels),
                (batch_size, num_rois, 4)]

           Shape of output
               (batch_size, num_rois, pooled_height, pooled_width, n_channels)

       """

    def __init__(self, **kwargs):
        self.pooled_height = 14
        self.pooled_width = 14
        super(RoiPooling, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        """ Returns the shape of the ROI Layer output
        """
        feature_map_shape, rois_shape = input_shape
        assert feature_map_shape[0] == rois_shape[0]
        batch_size = feature_map_shape[0]
        n_rois = rois_shape[1]
        n_channels = feature_map_shape[3]
        return (batch_size, n_rois, self.pooled_height,
                self.pooled_width, n_channels)

    def call(self, inputs, **kwargs):
        """ Maps the input tensor of the ROI layer to its output

            # Parameters
                x[0] -- Convolutional feature map tensor,
                        shape (batch_size, pooled_height, pooled_width, n_channels)
                x[1] -- Tensor of region of interests from candidate bounding boxes,
                        shape (batch_size, num_rois, 4)
                        Each region of interest is defined by four relative
                        coordinates (x_min, y_min, x_max, y_max) between 0 and 1
            # Output
                pooled_areas -- Tensor with the pooled region of interest, shape
                    (batch_size, num_rois, pooled_height, pooled_width, n_channels)
        """

        def curried_pool_rois(x):
            return RoiPooling._pool_rois(x[0], x[1],
                                              self.pooled_height,
                                              self.pooled_width)

        pooled_areas = tf.map_fn(curried_pool_rois, inputs, dtype=tf.float32)

        return pooled_areas

    @staticmethod
    def _pool_rois(feature_map, rois, pooled_height, pooled_width):
        """ Applies ROI pooling for a single image and varios ROIs
        """

        def curried_pool_roi(roi):
            return RoiPooling._pool_roi(feature_map, roi,
                                             pooled_height, pooled_width)

        pooled_areas = tf.map_fn(curried_pool_roi, rois, dtype=tf.float32)
        return pooled_areas

    @staticmethod
    def _pool_roi(feature_map, roi, pooled_height, pooled_width):
        """ Applies ROI pooling to a single image and a single region of interest
        """
        xmin=roi[0]
        xmax=roi[2]
        ymin=roi[1]
        ymax=roi[3]
        w=tf.maximum(1.0,roi[2]-roi[0])
        h=tf.maximum(1.0,roi[3]-roi[1])
        w = tf.cast(w, 'int32')
        h = tf.cast(h, 'int32')
        x = tf.cast(xmin, 'int32')
        y = tf.cast(ymin, 'int32')
        roi_map = image.resize_images(feature_map[y:y + h, x:x + w,:], (14, 14))
        return roi_map