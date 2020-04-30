from data_handler.imdb import imdb
from data_handler.generator import generator
import tensorflow as tf
import h5py
import cv2
import base64
from tools.config import cfg
from PIL import Image as Image
import os
import numpy as np
def image_process(image_info):
    """
    The method to load one image from image info to training.
    :param image_info:
    :return: the instance of image
    """
    path = os.path.join(cfg.IMAGESET, image_info.get("index") + ".jpg")
    if not os.path.exists(path):
        raise IOError("please check your file is not exists: " + path)
    def load_image(path):
        image = Image.open(path)
        return image
    return load_image(path)


def handle_origin_image(image, gt_box):
    """
    Resize image using one scale and make sure it can divide 32.
    :param image: the image to handle
    :return: the image after rescale
    """
    x = image.width
    y = image.height
    im_max = max(x, y)
    im_min = min(x, y)
    scale = cfg.TRAIN.MIN_SIZE / im_min
    if scale * im_max > cfg.TRAIN.MAX_SIZE:
        scale = cfg.TRAIN.MAX_SIZE / im_max
    width = round(round(x * scale) / 32) * 32
    height = round(round(y * scale) / 32) * 32
    im = image.resize((width, height))
    box = [round(gt_box[0] * width / x), round(gt_box[1] * height / y), round(gt_box[2] * width / x),
           round(gt_box[3] * height / y)]
    # make sure there really tiny flaw still have box to predict
    if (box[3] - box[1]) * (box[2] - box[0]) < 100:
        box = [box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3]
    return np.array(im), box


def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def convert_tf_example(image_buffer,box,label,i):
    label=label.tolist()
    print(i)
    example=tf.train.Example(features=tf.train.Features(feature={'label':_int64_feature(label.index(1)),
                                                                 'image':_bytes_feature(image_buffer),
                                                                 'xmin':_int64_feature(int(box[0])),
                                                                 'ymin': _int64_feature(int(box[1])),
                                                                 'xmax': _int64_feature(int(box[2])),
                                                                 'ymax': _int64_feature(int(box[3]))}))
    return example

def save_np():
    # the reason is numpy.save the dataset is to large to store.
    # Then using hdf5 to store image it can successfully store in file,but it will occure memory erro.
    # so finally i change to the tf.record
    filename = 'flaw.tfrecord'
    with tf.io.TFRecordWriter(filename) as writer:
        db=imdb("train")
        img_infos=db.roidb
        width_max=960
        height_max=960
        # for image_info in img_infos:
        #     gt_box=image_info.get('box')
        #     image=image_process(image_info)
        #     im,box=handle_origin_image(image,gt_box)
        #     height_max=max(im.shape[0],height_max)
        #     width_max=max(im.shape[1],width_max)
        # print(height_max)
        # print(width_max)
        i=0
        for image_info in img_infos:
            i+=1
            gt_box=image_info.get('box')
            image=image_process(image_info)
            im,box=handle_origin_image(image,gt_box)
            if image_info.get("flipped"):
                im=im[:,::-1,:]
                box=[im.shape[1]-box[2],box[1],im.shape[1]-box[0],box[3]]
            if im.shape[0] < height_max:
                pad_num = height_max - im.shape[0]
                im = np.pad(im, ((pad_num, 0), (0, 0), (0, 0)), 'constant')
                box = [box[0], box[1] + pad_num, box[2], box[3] + pad_num]
            if im.shape[1] < width_max:
                pad_num =  width_max - im.shape[1]
                im = np.pad(im, ((0, 0), (pad_num, 0), (0, 0)), 'constant')
                box = [box[0] + pad_num, box[1], box[2] + pad_num, box[3]]
            label=image_info.get('class')
            # with tf.python_io.TFRecordWriter('flaw.tfrecords') as writer:
            im=im.tobytes()
            example =convert_tf_example(im,box,label,i)
            writer.write(example.SerializeToString())



if __name__ == '__main__':
    save_np()