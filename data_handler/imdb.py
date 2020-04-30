
"""
This is the data sets for storing the image information and Region of Interest area
 and label which image will process image argument.
"""
import os,pickle,random
from tools.config import cfg
import xml.etree.ElementTree as ET
import numpy as np
from tools.tracer import TraceStack

class imdb(object):
    def __init__(self, name):
        # TraceStack()
        self._name = name
        self._num_classes = cfg.NUM_CLASSES
        self._class = cfg.CLASSES_NAME
        self._image_amount = np.zeros((self._num_classes), dtype=np.int32)
        self._image_index = self.load_image_index()
        self._roidb = self.default_roidb()

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return self.num_classes

    @property
    def image_index(self):
        return self._image_index

    @property
    def num_images(self):
        return len(self.image_index)

    @property
    def cache_path(self):
        cache_path = os.path.abspath(os.path.join(cfg.DATA_DIR, 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

    @property
    def roidb(self):
        """
        this is ground truth of image.
        #the flaw part of image.
        """
        return self._roidb

    def default_roidb(self):
        """"
        The method to load ground truth area(flaw area).
        prompt: this roidb is not self.roidb and not self._roidb just variable
        """
        print("########################################################################")
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            print("loading file from :   " + cache_file)
            with open(cache_file, 'rb') as cache:
                roidb = pickle.load(cache, encoding='iso-8859-1')
            print("finish loading data")
            print("########################################################################")
            return roidb
        print("writing file to :   " + cache_file)
        # roidb is from annotation information.
        roidb = [self.load_pascal_annotation(index) for index in self.image_index]
        self.make_class_blance(roidb)
        with open(cache_file, 'wb') as cache:
            # record the roidb information into pkl file to reduce time.
            pickle.dump(roidb, cache, pickle.HIGHEST_PROTOCOL)
        print("########################################################################")
        return roidb

    def load_pascal_annotation(self, index):
        """
           Load image and bounding boxes info from XML file in the PASCAL VOC
           format.
           """
        filename = os.path.join(cfg.ANOTATIONS, index + ".xml")
        tree = ET.parse(filename)
        flaw = tree.find("object")
        box = np.zeros([4], dtype=np.uint16)
        # the array to present class
        gt_class = np.zeros([self._num_classes], dtype=np.int32)
        bbox = flaw.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        # get the number represent flaw
        cls_num = self._class.index(flaw.find('name').text.strip())
        area = (x2 - x1) * (y2 - y1)
        gt_class[cls_num] = 1
        box[:] = [x1, y1, x2, y2]
        return {'box': box, 'index': index, 'class': gt_class, 'area': area,"flipped":False}

    def load_image_index(self):
        """
        Load the indexes listed in this data set's image set file.
        """
        image_index = []
        image_set_file = os.path.join(cfg.DATA_DIR, 'ImageSets', 'Main',
                                      self.name + '.txt')
        with open(image_set_file) as f:
            for x in f.readlines():
                image_index.append(x.strip())
                index = self._class.index(x[:2])
                self._image_amount[index] += 1
        return image_index

    def make_class_blance(self,roidb):
        """""
        The method to make each class have same number of image'
        It is simmilar setter of roidb
        """
        max = self._image_amount.max()
        for i in range(len(self._class)):
            if self._image_amount[i] < max:
                adding_num = max - self._image_amount[i]
                self.random_add_roidb(self._class[i], adding_num,roidb)

    def random_add_roidb(self, enhance_class, adding_num,roidb):
        """""
           The method to random adding image to make each class's image amount balance.
           """
        # record i same class data .
        record = []
        for i in range(len(self._image_index)):
            if self.image_index[i][:2] == enhance_class:
                record.append(i)
        # sometimes is larger than 2 times
        # record the index of image want to add
        image_choice = []
        while len(image_choice) < adding_num:
            image_choice.extend(random.sample(record, 30))
        for a in image_choice:
            dic = {'box': roidb[a].get('box'), 'index': roidb[a].get('index'),
                   'class':roidb[a].get('class'), 'area': roidb[a].get('area'),"flipped":True}
            roidb.append(dic)
            self._image_amount[self._class.index(enhance_class)] += 1
            self.image_index.append(roidb[a].get('index'))
        assert len(self.image_index)==np.sum(self._image_amount),"the image index amount and image amount is not similar "


    def show_database(self):
        """
        The method to print each class' image number
        """
        print("################################################################")
        print("this is the information of your "+self.name+" database info:")
        print("Image Number : "+str(self._image_amount))
        print("------------------------------------------------------------------")
        for i in range(len(self._class)):
            print(self._class[i]+"  :  "+str(self._image_amount[i]))
        print("------------------------------------------------------------------")


    def image_path_at(self, i):
        path = os.path.join(cfg.IMAGESET, self.image_index[i], ".jpg")
        return path
