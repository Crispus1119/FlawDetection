import data_handler.imdb as imdb
from tools.config import cfg
from network.resnet import resnet
import tensorflow as tf
from PIL import Image as Image
import random, os, math
import numpy as np
from tools import debug
"""
the module to generator data for training rpn_network.
"""


class data_generator_rpn():
    def __init__(self):
        self._training_data = imdb.imdb("train")
        self._iteration = self._training_data.num_images
        self._step_epoch = len(self._training_data.roidb)
        self.network = resnet()

    @property
    def step_per_epoch(self):
        return self._step_epoch

    @property
    def generator(self):
        while True:
            random.shuffle(self._training_data.roidb)
            for image_info in self._training_data.roidb:
                if image_info.get('index')[:2]!='XO':
                    continue
                gt_box = image_info.get("box")
                # box is(xmin,xmax,ymin,ymax) in the origin image
                im, box = self.handle_origin_image(self.image_process(image_info), gt_box)
                map_whidth, map_height = self.network.cal_fm_size(im.width, im.height)
                # boxes(centerx,centery,w,h)
                boxes = self.generate_anchor(map_height, map_whidth)
                boxes_cls, boxes_loc = self.box_judger(boxes, map_height, map_whidth, im.width, im.height, box)
                # adding one dimension for batch.
                im = np.array(im)[None]
                im=np.array(im)
                im=tf.image.per_image_standardization(im)
                boxes_cls = boxes_cls[None]
                boxes_loc = boxes_loc[None]
                image_info['change_box'] = box
                debug.check_data_generator(data_generator_rpn(),[boxes_cls, boxes_loc],image_info)
                # yield im, [boxes_cls, boxes_loc]
                yield im,[boxes_cls,boxes_loc],image_info

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

        #  i can implement data augument here.
        return load_image(path)

    def box_judger(self, boxes, f_height, f_width, whidth, height, gt_box):
        """
        The method to generate cls.-1 means negative,0 means useless anchor.
        change : the fixed number of negitive and positive anchors
        :param boxes: the anchor boxes(cx,cy,w,h).
        :return: Classification of boxes.
        """
        box_infos = np.zeros((f_height, f_width, 4*cfg.TRAIN.ANCHOR_NUM),dtype=np.float32)
        box_locs = np.zeros((f_height, f_width, 8*cfg.TRAIN.ANCHOR_NUM),dtype=np.float32)
        iou_record=np.zeros((f_height,f_width,cfg.TRAIN.ANCHOR_NUM))
        best_iou=0
        for i in range(f_height):
            for j in range(f_width):
                valid = []
                boxes_type = []
                for x in range(0, len(boxes[i, j]), 4):
                    box = [boxes[i, j, x], boxes[i, j, x + 1], boxes[i, j, x + 2], boxes[i, j, x + 3]]
                    # box is [xmin,ymin,xmax,ymax]
                    box = [box[0] - box[2] / 2, box[1] - box[3] / 2, box[0] + box[2] / 2, box[1] + box[3] / 2]
                    box_is_valid = self.valid_box_check(box, whidth, height)
                    iou_value = self.iou(box, gt_box)
                    if box_is_valid and iou_value>0:
                        iou_record[i,j,int(x/4)]=iou_value
                        best_iou=max(best_iou,iou_value)
                    box_type = 0
                    if iou_value > cfg.TRAIN.IOUPOS and box_is_valid:
                        box_type = 1
                    # the anchors that we do not use to training.
                    if iou_value < cfg.TRAIN.IOUPOS and iou_value > cfg.TRAIN.IOUNEG:
                        box_type=-1
                    valid.append(box_is_valid)
                    boxes_type.extend([box_type, 1 - box_type])
                box_info = []
                box_loc = []
                for x in range(len(valid)):
                    if valid[x] + boxes_type[2 * x] != 2:
                        box_loc = box_loc + 4 * [0]
                    else:
                        box_loc = box_loc + 4 * [1]
                        change_box=self.cal_offset(boxes[i, j, 4 * x:4 * x + 4], gt_box)
                        boxes[i, j, 4 * x:4 * x + 4] = change_box
                    #     box info is whether box is valid
                    box_info = box_info + 2 * [valid[x]]
                box_locs[i, j] = box_loc + list(boxes[i, j])
                box_infos[i, j] = box_info + list(boxes_type)
        self.best_iou_label(best_iou,box_locs,box_infos,iou_record,gt_box)
        self.clip_data(box_infos)
        return box_infos, box_locs


    def best_iou_label(self,best_iou,box_locs,box_infos,iou_record,gt_box):
        """
        The method to only choose the highest iou to predict this object
        """
        for i in range(iou_record.shape[0]):
            for j in range(iou_record.shape[1]):
                for x in range(iou_record.shape[2]):
                    if iou_record[i,j,x]==best_iou and box_infos[i,j,2*x]==1:
                        box_infos[i,j,2*cfg.TRAIN.ANCHOR_NUM+2*x]=1
                        box_infos[i, j, 2*cfg.TRAIN.ANCHOR_NUM + 2 * x+1] = 0
                        if box_locs[i,j,4*x]!=1:
                            box_locs[i,j,4*x:4*x+4]=1
                            box_locs[i,j,4*cfg.TRAIN.ANCHOR_NUM+4*x:4*cfg.TRAIN.ANCHOR_NUM+4*x+4]=self.cal_offset(box_locs[i, j, 4*cfg.TRAIN.ANCHOR_NUM+4 * x:4*cfg.TRAIN.ANCHOR_NUM+4 * x + 4], gt_box)

    def cal_offset(self, box, gt_box):
        """
        The method to calculate the offset of anchor boxes and gt_box.
        :param box: the anchor box[cx,cy,w,h]
        :param gt_box: the ground truth box[xmin,ymin,xmax,ymax]
        :return:
        """
        std=cfg.TRAIN.std_scaling
        gt_box_width = gt_box[2] - gt_box[0]
        gt_box_height = gt_box[3] - gt_box[1]
        gt_box = [gt_box[0] + gt_box_width / 2, gt_box[1] + gt_box_height / 2, gt_box_width, gt_box_height]
        x_off = ((gt_box[0] - box[0]) / box[2])*std
        y_off = ((gt_box[1] - box[1]) / box[3])*std
        w_off = math.log(gt_box[2] / box[2])*std
        h_off = math.log(gt_box[3] / box[3])*std
        return [x_off, y_off, w_off, h_off]

    def clip_data(self,box_infos):
        '''
        The method to keep negative data and positive data balance
        :return:the balance data sample to training
        '''
        num=cfg.TRAIN.BATCH_SIZE/2
        box_valid=box_infos[:,:,:2*cfg.TRAIN.ANCHOR_NUM]
        box_type=box_infos[:,:,2*cfg.TRAIN.ANCHOR_NUM:]
        box_type=box_type[:, :, ::2]
        box_valid=box_valid[:, :, ::2]
        neg_index = np.where(np.logical_and(np.equal(box_type, 0), np.equal(box_valid, 1)))
        if len(neg_index[0])>num:
            delet_num=int(len(neg_index[0])-num)
            delete_index=random.sample(range(len(neg_index[0])),delet_num)
            cancel_index=np.squeeze([i[delete_index] for i in neg_index])
            box_valid[cancel_index[0],cancel_index[1],cancel_index[2]]=0
            box_valid = box_infos[:, :, :2*cfg.TRAIN.ANCHOR_NUM]
            box_valid = box_valid[:, :, 1::2]
            box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
        hard_negtive_index=np.where(np.less(box_type,0))
        box_valid[hard_negtive_index[0], hard_negtive_index[1], hard_negtive_index[2]] = 0
        box_valid = box_infos[:, :, :2*cfg.TRAIN.ANCHOR_NUM]
        box_valid = box_valid[:, :, ::2]
        box_valid[hard_negtive_index[0], hard_negtive_index[1], hard_negtive_index[2]] = 0




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
        squareOne = (box[2] - box[0]) * (box[3] - box[1])
        squareTwo = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
        iou = area / (squareOne + squareTwo - area + 1e-10)
        iou=np.round(iou,2)
        return iou

    def valid_box_check(self, box, whidth, height):
        """
        the method to check whether box is out of bounder
        box is [xmin,ymin,xmax,ymax]
        """
        if box[2] - box[0] <= 0 or box[3] - box[1]<= 0:
            return 0
        if box[0]< 0 or box[1] < 0:
            return 0
        if box[2]> whidth or box[3] > height:
            return 0
        return 1

    def generate_anchor(self, map_height, map_whidth):
        """
        The method to generate anchor for supervised learning
        :param map_height: the feature map height
        :param map_width: the feature map width
        :return: anchor boxes
        """
        anchors = np.zeros((map_height, map_whidth, 4*cfg.TRAIN.ANCHOR_NUM))
        ratios = cfg.TRAIN.RATIOS
        scales = cfg.TRAIN.SCALES
        rpn_stride = cfg.TRAIN.RPN_STRIDE
        scales, ratios = np.meshgrid(scales, ratios)
        scaleX = scales * np.sqrt(ratios)
        scaleY = scales / np.sqrt(ratios)
        centerX = np.arange(0, map_whidth) * rpn_stride
        centerY = np.arange(0, map_height) * rpn_stride
        # centerX, centerY = np.meshgrid(centerX, centerY)
        larges = np.stack([scaleX.flatten(), scaleY.flatten()], axis=1)
        assert len(centerY) == map_height, "the height and number of anchor center is not equal"
        assert len(centerX) == map_whidth, "the width and number of anchor center is not equal"
        for i in range(len(centerY)):
            for j in range(len(centerX)):
                record = []
                for l in range(len(larges)):
                    # [centerx,centery,w,h]
                    record = record + [centerX[j], centerY[i]] + list(larges[l])
                anchors[i, j] = record
        #  the different betweent np.stack and np.concatenate is stack will add a dimension but concatenate will not.
        return anchors

    def handle_origin_image(self, image, gt_box):
        """
        Resize image using one scale
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
        im = image.resize((round(x * scale), round(y * scale)))
        box = [round(i * scale) for i in gt_box]
        return im, box
