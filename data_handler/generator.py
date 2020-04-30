from data_handler import imdb
import abc
import math
import numpy as np
from PIL import Image as Image
from tools.config import cfg
import os
import random
class generator(metaclass=abc.ABCMeta):
    def __init__(self,name,net):
        self._training_data = imdb.imdb(name)
        self._iteration = self._training_data.num_images
        self._step_epoch = len(self._training_data.roidb)//cfg.TRAIN.BATCH
        self.network = net

    @property
    def step_per_epoch(self):
        return self._step_epoch

    # @abc.abstractmethod
    # def generator(self):
    #     pass

    def cal_offset(self, box, gt_box):
        """
        The method to calculate the offset of anchor boxes and gt_box(bounding box regression).
        :param box: the anchor box[cx,cy,w,h]
        :param gt_box: the ground truth box[xmin,ymin,xmax,ymax]
        :return:
        """
        std = cfg.TRAIN.std_scaling
        gt_box_width = gt_box[2] - gt_box[0]
        gt_box_height = gt_box[3] - gt_box[1]
        gt_box = [gt_box[0] + gt_box_width / 2, gt_box[1] + gt_box_height / 2, gt_box_width, gt_box_height]
        x_off = ((gt_box[0] - box[0]) / box[2]) * std
        y_off = ((gt_box[1] - box[1]) / box[3]) * std
        w_off = math.log(gt_box[2] / box[2]) * std
        h_off = math.log(gt_box[3] / box[3]) * std
        return [x_off, y_off, w_off, h_off]

    def iou(self, box, gt_box):
        """
        THe method to calculate the iou value,and using iou value to define this anchor.
        :param box: the anchor box location
        :param gt_box: the gt_box location
        :return: the iou value
        """
        # if there has no intersection there have four possible
        if (box[0] >= gt_box[2] or box[1] >= gt_box[3] or gt_box[0] >= box[2] or gt_box[1] > box[3]):
            return 0
        x_min = max(box[0], gt_box[0])
        x_max = min(box[2], gt_box[2])
        y_min = max(box[1], gt_box[1])
        y_max = min(box[3], gt_box[3])
        area = (x_max - x_min) * (y_max - y_min)
        square = (box[2] - box[0]) * (box[3] - box[1])
        square += (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        iou = area / (square - area + 1e-10)
        iou = np.round(iou,5)
        return iou

    def image_process(self, image_info):
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

    def generate_anchor(self, map_height, map_whidth,rpn_stride):
        """
        The method to generate anchor for supervised learning
        :param map_height: the feature map height
        :param map_width: the feature map width
        :return: anchor boxes
        """
        anchors = np.zeros((map_height, map_whidth, 4*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
        ratios = cfg.TRAIN.RATIOS
        if rpn_stride==32:
           scales = cfg.TRAIN.LSCALES
        if rpn_stride==16:
           scales = cfg.TRAIN.MSCALES
        if rpn_stride==8:
           scales = cfg.TRAIN.SSCALES
        scales, ratios = np.meshgrid(scales, ratios)
        scaleX = scales * np.sqrt(ratios)/2
        scaleY = scales / np.sqrt(ratios)/2
        centerX = np.arange(0, map_whidth)
        np.multiply(centerX,rpn_stride,out=centerX,casting='unsafe')
        np.add(centerX,rpn_stride/2,out=centerX,casting='unsafe')
        centerY = np.arange(0, map_height)
        np.multiply(centerY,rpn_stride,out=centerY,casting='unsafe')
        np.add(centerY, rpn_stride / 2, out=centerY, casting='unsafe')
        # larges = np.stack([scaleX.ravel(), scaleY.ravel()], axis=1)
        larges=np.stack([scaleX.ravel(), scaleY.ravel()], axis=1)
        larges_2 = -larges.copy()
        large=np.concatenate([larges_2,larges],axis=1).ravel()
        for i in range(len(centerY)):
            for j in range(len(centerX)):
                anchors[i, j] =[centerX[j],centerY[i]]*2*cfg.TRAIN.ANCHOR_NUM+large
        return anchors

    def handle_origin_image(self, image, gt_box,image_info):
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
        width=round(round(x * scale)/32)*32
        height = round(round(y * scale) / 32) * 32
        im = image.resize((width,height))
        box = [round(gt_box[0]*width/x),round(gt_box[1]*height/y),round(gt_box[2]*width/x),round(gt_box[3]*height/y)]
        # make sure there really tiny flaw still have box to predict
        if (box[3]-box[1])*(box[2]-box[0])<1000:
            box=[box[0]-5,box[1]-5,box[2]+5,box[3]+5]
        im=np.array(im)
        if image_info.get("flipped"):
            im=im[:,::-1,:]
            box=[im.shape[1]-box[2],box[1],im.shape[1]-box[0],box[3]]
        return im, box

    # def box_judger(self, boxes, f_height, f_width, whidth, height, gt_box):
    #     """
    #     The method to generate cls.-1 means negative,0 means useless anchor.
    #     and make sure the anchor box is meaningful
    #     :param boxes: the anchor boxes(cx,cy,w,h).
    #     :return: Classification of boxes.
    #     """
    #     box_infos = np.zeros((f_height, f_width, 4*cfg.TRAIN.ANCHOR_NUM),dtype=np.float32)
    #     box_locs = np.zeros((f_height, f_width, 8*cfg.TRAIN.ANCHOR_NUM),dtype=np.float32)
    #     iou_record=np.zeros((f_height,f_width,cfg.TRAIN.ANCHOR_NUM))
    #     # best_iou=0
    #     # traverse each vector in feature map and estimate whether it is valid and the label of anchor boxes.
    #     for i in range(f_height):
    #         for j in range(f_width):
    #             valid = []
    #             boxes_type = []
    #             for x in range(0, len(boxes[i, j]), 4):
    #                 box = [boxes[i, j, x], boxes[i, j, x + 1], boxes[i, j, x + 2], boxes[i, j, x + 3]]
    #                 # change box to [xmin,ymin,xmax,ymax]
    #                 box = [box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2]
    #                 box_is_valid = self.valid_box_check(box, whidth, height)
    #                 iou_value = self.iou(box, gt_box)
    #                 if box_is_valid and iou_value>0.7:
    #                     iou_record[i,j,int(x/4)]=iou_value
    #                     # get the max IoU of anchor,and using this anchor to predict this object.
    #                     # best_iou=max(best_iou,iou_value)
    #                 box_type = -1
    #                 if iou_value > cfg.TRAIN.IOUPOS and box_is_valid:
    #                     box_type = 1
    #                 # the anchors that we do not use to training.
    #                 if iou_value < cfg.TRAIN.IOUPOS and iou_value > cfg.TRAIN.IOUNEG:
    #                     box_type=0
    #                 valid.append(box_is_valid)
    #                 boxes_type.extend([box_type, 1 - box_type])
    #             box_info = []
    #             box_loc = []
    #             for x in range(len(valid)):
    #                 # handle the anchors which are not positive.
    #                 if valid[x] + boxes_type[2 * x] != 2:
    #                     box_loc = box_loc + 4 * [0]
    #                 else:
    #                     box_loc = box_loc + 4 * [1]
    #                     boxes[i, j, 4 * x:4 * x + 4]=self.cal_offset(boxes[i, j, 4 * x:4 * x + 4], gt_box)
    #                 #     box info is whether box is valid
    #                 box_info = box_info + 2 * [valid[x]]
    #             box_locs[i, j] = box_loc + list(boxes[i, j])
    #             box_infos[i, j] = box_info + list(boxes_type)
    #     # self.best_iou_label(best_iou,box_locs,box_infos,iou_record,gt_box)
    #     self.clip_data(box_infos)
    #     return box_infos, box_locs

    def valid_box_check(self, box, whidth, height):
        """
        the method to check whether box is out of bounder
        box is [xmin,ymin,xmax,ymax]
        """
        if box[0]< 0 or box[1] < 0 or box[2]> whidth or box[3] > height:
            return 0
        return 1

    # def clip_data(self,box_infos):
    #     '''
    #     The method to keep negative data and positive data balance
    #     :return:the balance data sample to training
    #     '''
    #     num=cfg.TRAIN.BATCH_SIZE/2
    #     box_valid=box_infos[:,:,:2*cfg.TRAIN.ANCHOR_NUM]
    #     box_type=box_infos[:,:,2*cfg.TRAIN.ANCHOR_NUM:]
    #     box_type=box_type[:, :, ::2]
    #     box_valid=box_valid[:, :, ::2]
    #     hard_neg_index = np.where(np.logical_and(np.equal(box_type, 0), np.equal(box_valid, 1)))
    #     neg_index = np.where(np.logical_and(np.equal(box_type, -1), np.equal(box_valid, 1)))
    #     if len(hard_neg_index[0])>num:
    #         delet_num = int(len(hard_neg_index[0]) - num)
    #         delete_index = random.sample(range(len(hard_neg_index[0])), delet_num)
    #         cancel_index = np.squeeze([i[delete_index] for i in hard_neg_index])
    #         box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
    #         box_valid = box_infos[:, :, :2 * cfg.TRAIN.ANCHOR_NUM]
    #         box_valid = box_valid[:, :, 1::2]
    #         box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
    #         cancel_index = np.squeeze([i for i in neg_index])
    #         box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
    #         box_valid = box_infos[:, :, :2 * cfg.TRAIN.ANCHOR_NUM]
    #         box_valid = box_valid[:, :, 1::2]
    #         box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
    #     if len(hard_neg_index[0])<num:
    #         delete_num=int(len(neg_index)-(num-len(hard_neg_index)))
    #         delete_index=random.sample(range(len(neg_index[0])),delete_num)
    #         cancel_index=np.squeeze([i[delete_index] for i in neg_index])
    #         box_valid[cancel_index[0],cancel_index[1],cancel_index[2]]=0
    #         box_valid = box_infos[:, :, :2*cfg.TRAIN.ANCHOR_NUM]
    #         box_valid = box_valid[:, :, 1::2]
    #         box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
        # negtive_index=np.where(np.less(box_type,0))
        # box_valid[hard_negtive_index[0], hard_negtive_index[1], hard_negtive_index[2]] = 0
        # box_valid = box_infos[:, :, :2*cfg.TRAIN.ANCHOR_NUM]
        # box_valid = box_valid[:, :, ::2]
        # box_valid[hard_negtive_index[0], hard_negtive_index[1], hard_negtive_index[2]] = 0

    def best_iou_label(self, best_iou, box_locs, box_infos, iou_record, gt_box):
        """
        The method to only choose the highest iou to predict this object
        """
        for i in range(iou_record.shape[0]):
            for j in range(iou_record.shape[1]):
                for x in range(iou_record.shape[2]):
                    if iou_record[i, j, x] == best_iou and box_infos[i, j, 2 * x] == 1:
                        box_infos[i, j, 2 * cfg.TRAIN.ANCHOR_NUM + 2 * x] = 1
                        box_infos[i, j, 2 * cfg.TRAIN.ANCHOR_NUM + 2 * x + 1] = 0
                        box_locs[i, j, 4 * x:4 * x + 4] = 1
                        box_locs[i, j,
                        4 * cfg.TRAIN.ANCHOR_NUM + 4 * x:4 * cfg.TRAIN.ANCHOR_NUM + 4 * x + 4] = self.cal_offset(
                            box_locs[i, j, 4 * cfg.TRAIN.ANCHOR_NUM + 4 * x:4 * cfg.TRAIN.ANCHOR_NUM + 4 * x + 4],
                            gt_box)
                        return


