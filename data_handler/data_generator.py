from tools.config import cfg
import data_handler.imdb as imdb
import os,random
import numpy as np
from PIL import Image
import concurrent.futures

"""
the data generator to generate data to feed into network for training basic network for recognition.
Training process: crop the flaw part and resize to 224*224 and using data to training classification model.
"""

class data_generator():

    def __init__(self,batch_size):
        # self.image_data_set = imdb.imdb("trainval")
        # self.valid_data_set= imdb.imdb("test")
        self.batch_size=batch_size
        self.roidb=self.prepare_roidb()
        self.batch_num=len(self.image_data_set.roidb)//self.batch_size


    @property
    def steps_per_epoch(self):
        return self.batch_num

    @property
    def generator(self):
        """
        The generator to privide the data for training
        :return: the generator of training data
        """
        batch_num=len(self.image_data_set.roidb)//self.batch_size
        data_num=batch_num*self.batch_size
        while True:
            batches=self.shuffle_batch(data_num,batch_num)
            for i in range(len(batches)):
                x_data=self.readImage(batches[i])
                y_data =[]
                for i in batches[i]:
                    y_data.append(i.get('class'))
                yield np.array(x_data),np.array( y_data)


    def shuffle_batch(self,data_num,batch_num):
        random.shuffle(self.image_data_set.roidb)
        samples = np.array(self.image_data_set.roidb[:data_num])
        batches = np.split(samples, batch_num)
        return batches


    # feel like if read image's size is small, using multi-process will slow down process.
    def readImage(self,batch):
        image_size = cfg.TRAIN.RESIZE_ROIDB
        roi_data=[]
        for i in batch:
            path=os.path.join(cfg.IMAGESET,i.get('index')+".jpg")
            im = Image.open(path)
            roi = im.crop(i.get('box'))
            data = roi.resize((image_size, image_size))
            arra=np.array(data)
            roi_data.append(arra)
        return roi_data


    def get_image_roi(self,path,box,lable):
        """
        The method to get roi info from image_path
        :param image_path: the image path
        :return:
        """
        image_size = cfg.TRAIN.RESIZE_ROIDB
        im = Image.open(path)
        roi = im.crop(box)
        data = roi.resize((image_size, image_size))
        arra = np.array(data)
        return arra,lable


    def prepare_roidb(self):
        """
        Prepare roi data argument
        :return: new roidb to prepare training.
        """
        if cfg.TRAIN.BALANCE_CLASSES:
            self.image_data_set.make_class_blance()

        self.image_data_set.show_database()


    def read_image(self):
        """
        the method  to read image and crop ,resize it, and give the label to it.
        :return: roi data images(None,224,224,3).
        """
        roidb = self.image_data_set.roidb
        data_size = len(roidb)
        image_size = cfg.TRAIN.RESIZE_ROIDB
        # the image of box resize to 224.
        roi_data = np.zeros((data_size, image_size, image_size,3), dtype=np.float32)
        image_label = np.zeros((data_size, cfg.NUM_CLASSES),dtype=np.float32)
        for i in range(data_size):
            path = os.path.join(cfg.IMAGESET, roidb[i].get("index")+".jpg")
            image_label[i] = roidb[i].get("class")
            im = Image.open(path)
            box = roidb[i].get("box")
            roi = im.crop(box)
            roi_data[i] = roi.resize((image_size, image_size))
        return roi_data,image_label



    def shuffle_data(self,image_data,image_label):
        """
        The method to shuffle all data in training set.
        :param image_data:
        :param image_label:
        :return:
        """
        mid=zip(image_data,image_label)
        random.shuffle(mid)
        image_data,image_label=zip(* mid)
        return image_data,image_label

    @property
    def get_val_data(self):
        roidb = self.valid_data_set.roidb
        data_size = len(roidb)
        roi_data = []
        image_label = []
        image_files=[]
        lables=[]
        box=[]
        for i in range(data_size):
            box.append(roidb[i].get("box"))
            lables.append(roidb[i].get("class"))
            path=os.path.join(cfg.IMAGESET, roidb[i].get("index")+".jpg")
            image_files.append(path)
        with concurrent.futures.ProcessPoolExecutor() as executor:
             image_data=executor.map(self.get_image_roi, image_files,box,lables)
        for i,j in image_data:
            roi_data.append(i)
            image_label.append(j)
        return np.array(roi_data,dtype=np.float32), np.array(image_label,dtype=np.float32)

