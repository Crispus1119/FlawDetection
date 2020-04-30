from data_handler.generator import generator
from network.resnet import resnet
import random
import tensorflow as tf
from tools.config import cfg
import threading
import h5py
import numpy as np
import time
from tools.thread_safe import threadsafe_generator

class data_generator(generator):
    def __init__(self):
        super(data_generator, self).__init__('train',resnet())
        self.map_whidth, self.map_height = self.network.cal_fm_size(960, 960, isFPN=True)
        # option_no_order = tf.data.Options()
        # option_no_order.experimental_deterministic = False
        # dataset = tf.data.TFRecordDataset('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/data/flaw.tfrecord')
        # dataset = dataset.with_options(option_no_order)
        # # dataset =dataset.interleave(lambda x:tf.data.Dataset.from_tensors(x), cycle_length=8, block_length=8, num_parallel_calls=8)
        # dataset=dataset.map(self.transform)
        # # # self.dataset=self.dataset.map(self.get_label)
        # self.dataset = dataset.repeat()
        # self.batched_dataset = self.dataset.batch(cfg.TRAIN.BATCH, drop_remainder=True)
        # self.dataset = self.dataset.repeat()
        # self.batched_dataset=self.dataset.batch(cfg.TRAIN.BATCH,drop_remainder=True)
        # with h5py.File('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/data/data.h5', 'r') as f:
        #     self.images_data=f['images'].value
        #     self.boxes=f['boxes'].value
        #     self.labels=f['labels'].value
        # self.images_data=np.load("/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/data/image.npy")
        # self.boxes = np.load(
        #     "/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/data/box.npy")
        # self.label = np.load(
        #     "/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/data/label.npy")

    # @tf.function
    # def transform(self,example):
    #     image_dictionary={'label':tf.io.FixedLenFeature([],tf.int64),
    #                       'image':tf.io.FixedLenFeature([],tf.string),
    #                       'xmin':tf.io.FixedLenFeature([],tf.int64),
    #                       'xmax': tf.io.FixedLenFeature([], tf.int64),
    #                       'ymin': tf.io.FixedLenFeature([], tf.int64),
    #                       'ymax': tf.io.FixedLenFeature([], tf.int64)}
    #     return tf.io.parse_single_example(example, image_dictionary)


    # def get_data(self):
    #     dataset=tf.data.TFRecordDataset('/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/data/flaw.tfrecord')
    #     dataset=dataset.interleave(self.transform, cycle_length=8,block_length=8, num_parallel_calls=8)
    #     dataset = dataset.repeat()
    #     dataset = dataset.batch(8, drop_remainder=True)
    #     dataset = dataset.prefetch(16)
    #     return dataset
    @property
    @threadsafe_generator
    def g(self):
            while True:
                random.shuffle(self._training_data.roidb)
                num_batch=len(self._training_data.roidb)//cfg.TRAIN.BATCH
                for batch_index in range(num_batch):
                    info=[]
                    im_record=[]
                    box_record=[]
                    max_height=0
                    max_width=0
                    for image_index in range(cfg.TRAIN.BATCH):
                            image_info=self._training_data.roidb[batch_index*cfg.TRAIN.BATCH+image_index]
                            gt_box = image_info.get("box")
                            im, box = self.handle_origin_image(self.image_process(image_info),gt_box,image_info)
                            max_height=max(max_height,im.shape[0])
                            max_width=max(max_width,im.shape[1])
                            info.append(image_info)
                            im_record.append(im)
                            box_record.append(box)
                    im_data = np.zeros((cfg.TRAIN.BATCH, max_height, max_width, 3),dtype='float32')
                    map_whidth, map_height = self.network.cal_fm_size(max_width, max_height, isFPN=True)
                    small_cls_output = np.zeros((cfg.TRAIN.BATCH, map_height[2], map_whidth[2],4*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
                    small_reg_output=np.zeros((cfg.TRAIN.BATCH, map_height[2], map_whidth[2],8*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
                    mid_cls_output=np.zeros((cfg.TRAIN.BATCH, map_height[1], map_whidth[1],4*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
                    mid_reg_output = np.zeros((cfg.TRAIN.BATCH, map_height[1], map_whidth[1],8*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
                    large_cls_output=np.zeros((cfg.TRAIN.BATCH, map_height[0], map_whidth[0],4*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
                    large_reg_output =np.zeros((cfg.TRAIN.BATCH, map_height[0], map_whidth[0],8*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
                    for i in range(8):
                            im=im_record[i]
                            box=box_record[i]
                            if im.shape[0]<max_height:
                                pad_num=max_height-im.shape[0]
                                im=np.pad(im,((pad_num,0),(0,0),(0,0)),'constant')
                                box=[box[0],box[1]+pad_num,box[2],box[3]+pad_num]
                            if im.shape[1]<max_width:
                                pad_num = max_width - im.shape[1]
                                im = np.pad(im, ((0, 0), (pad_num, 0), (0, 0)), 'constant')
                                box = [box[0]+ pad_num, box[1], box[2]+ pad_num, box[3]]
                            im_data[i] = im
                            info[i]['change_box'] = box
                            rpn_stride=32
                            for size in range(len(map_whidth)-1,-1,-1):
                                boxes = self.generate_anchor(map_height[size], map_whidth[size],rpn_stride)
                                rpn_stride/=2
                                boxes_cls, boxes_loc = self.quick_version(boxes, map_height[size], map_whidth[size], im.shape[1], im.shape[0], box)
                                if size==2:
                                    small_cls_output[i]=boxes_cls
                                    small_reg_output[i]=boxes_loc
                                if size == 1:
                                    mid_cls_output[i]=boxes_cls
                                    mid_reg_output[i]=boxes_loc
                                if size == 0:
                                    large_cls_output[i]=boxes_cls
                                    large_reg_output[i]=boxes_loc

                    yield (im_data, [small_cls_output, small_reg_output,
                                              mid_cls_output, mid_reg_output,
                                              large_cls_output,large_reg_output
                            ],info)

    # def get_label(self,example):
    #     im = example["image"]
    #     img = tf.io.decode_raw(im, out_type=tf.uint8)
    #     image = tf.reshape(img, (960, 960, 3)).numpy(dtype=np.float32)
    #     box = [example['xmin'].numpy(), example['ymin'].numpy(), example['xmax'].numpy(),
    #            example['ymax'].numpy()]
    #     rpn_stride = 32
    #     for i in range(len(self.map_whidth) - 1, -1, -1):
    #         boxes = self.generate_anchor(self.map_height[i], self.map_whidth[i], rpn_stride)
    #         rpn_stride /= 2
    #         boxes_cls, boxes_loc = self.box_judger(boxes, self.map_height[i], self.map_whidth[i], 960,
    #                                                960, box)
    #         if i == 2:
    #             small_cls_output = boxes_cls
    #             small_reg_output = boxes_loc
    #         if i == 1:
    #             mid_cls_output = boxes_cls
    #             mid_reg_output = boxes_loc
    #         if i == 0:
    #             large_cls_output = boxes_cls
    #             large_reg_output = boxes_loc
    #
    #     return image, [small_cls_output, small_reg_output,
    #                    mid_cls_output, mid_reg_output,
    #                    large_cls_output, large_reg_output
    #                    ]
    # @property
    # @threadsafe_generator
    # def g(self):
    #         while True:
    #             # self.dataset=self.dataset.shuffle(buffer_size=10000)
    #             for batch in self.batched_dataset:
    #             #         batch=next(self.batched_dataset.__iter__())
    #                     # dataset = batched_dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #                     im_record=np.zeros((cfg.TRAIN.BATCH,960,960,3),dtype=np.float32)
    #                     map_whidth, map_height = self.network.cal_fm_size(960,960, isFPN=True)
    #                     small_cls_output = np.zeros((cfg.TRAIN.BATCH, map_height[2], map_whidth[2],4*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
    #                     small_reg_output=np.zeros((cfg.TRAIN.BATCH, map_height[2], map_whidth[2],8*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
    #                     mid_cls_output=np.zeros((cfg.TRAIN.BATCH, map_height[1], map_whidth[1],4*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
    #                     mid_reg_output = np.zeros((cfg.TRAIN.BATCH, map_height[1], map_whidth[1],8*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
    #                     large_cls_output=np.zeros((cfg.TRAIN.BATCH, map_height[0], map_whidth[0],4*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
    #                     large_reg_output =np.zeros((cfg.TRAIN.BATCH, map_height[0], map_whidth[0],8*cfg.TRAIN.ANCHOR_NUM),dtype='float32')
    #                     for num in range(cfg.TRAIN.BATCH):
    #                             im=batch["image"][num]
    #                             img = tf.io.decode_raw(im,out_type=tf.uint8)
    #                             image = tf.reshape(img, (960,960,3)).numpy()
    #                             im_record[num]=image
    #                             box=[batch['xmin'][num].numpy(),batch['ymin'][num].numpy(),batch['xmax'][num].numpy(),batch['ymax'][num].numpy()]
    #                             rpn_stride=32
    #                             for i in range(len(map_whidth)-1,-1,-1):
    #                                 boxes = self.generate_anchor(map_height[i], map_whidth[i],rpn_stride)
    #                                 rpn_stride/=2
    #                                 boxes_cls, boxes_loc = self.box_judger(boxes, map_height[i], map_whidth[i], im_record.shape[2], im_record.shape[1],box)
    #                                 if i==2:
    #                                     small_cls_output[num]=boxes_cls
    #                                     small_reg_output[num]=boxes_loc
    #                                 if i == 1:
    #                                     mid_cls_output[num]=boxes_cls
    #                                     mid_reg_output[num]=boxes_loc
    #                                 if i == 0:
    #                                     large_cls_output[num]=boxes_cls
    #                                     large_reg_output[num]=boxes_loc
    #                     yield (im_record, [small_cls_output, small_reg_output,
    #                                               mid_cls_output, mid_reg_output,
    #                                               large_cls_output,large_reg_output
    #                                      ])
    def quick_iou(self,boxes,gt_box):
        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        gt_area=(gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
        xx1 = np.maximum(boxes[:,0], gt_box[0])
        yy1 = np.maximum(boxes[:,1], gt_box[1])
        xx2 = np.minimum(boxes[:,2], gt_box[2])
        yy2 = np.minimum(boxes[:,3], gt_box[3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h
        iou=intersection/(areas+gt_area-intersection)
        return iou

    def quick_offset(self, box, gt_box):
        std = cfg.TRAIN.std_scaling
        box_width=box[:,2]-box[:,0]
        box_height=box[:,3]-box[:,1]
        gt_box_width = gt_box[2] - gt_box[0]
        gt_box_height = gt_box[3] - gt_box[1]
        gt_box = [gt_box[0] + gt_box_width / 2, gt_box[1] + gt_box_height / 2, gt_box_width, gt_box_height]
        x_off = ((gt_box[0] - (box[:,0]+box_width/2)) / box_width) * std
        y_off = ((gt_box[1] - (box[:,1]+box_height/2)) / box_height) * std
        w_off = np.log(gt_box[2] / box_width) * std
        h_off = np.log(gt_box[3] / box_height) * std
        return x_off, y_off, w_off, h_off

    def quick_version(self,boxes, f_height, f_width, whidth, height, gt_box):
        boxes = boxes.reshape((-1, 4))
        box_infos=np.ones((boxes.shape[0],4 ),dtype=np.float32)
        box_locs = np.zeros((boxes.shape[0], 8), dtype=np.float32)
        invalid_index=np.where(np.logical_or(np.logical_or(np.less(boxes[:,0],0),np.less(boxes[:,1],0)),np.logical_or(np.greater(boxes[:,2],whidth),np.greater(boxes[:,3],height))))[0]
        iou_record=self.quick_iou(boxes,gt_box)
        iou_record[invalid_index]=0
        box_infos[invalid_index] = 0
        max_iou=np.max(iou_record)
        positive_index=np.where(np.greater_equal(iou_record,cfg.TRAIN.IOUPOS))[0]
        no_label=np.where(np.logical_and(np.less(iou_record,cfg.TRAIN.IOUPOS),np.greater_equal(iou_record,0.1)))[0]
        hard_negtiva_label=np.where(np.less(iou_record,0.1))[0]
        box_infos[no_label]=0
        box_infos[positive_index,2:]=[1,0]
        box_infos[hard_negtiva_label, 2:] = [0, 1]
        if max_iou<cfg.TRAIN.IOUPOS:
            new_pos_index=np.where(iou_record==max_iou)[0]
            index=np.random.choice(new_pos_index,1)
            box_infos[index,2:]=[1,0]
            box_infos[index, :2] = [1, 1]
        positive_index = np.where(np.logical_and(np.equal(box_infos[:,2],1),np.equal(box_infos[:,3],0)))[0]
        negtive_index = np.where(np.logical_and(np.equal(box_infos[:, 2], 0), np.equal(box_infos[:, 3], 1)))[0]
        neg_num=256-len(positive_index)
        delete_neg=len(negtive_index)-neg_num
        if delete_neg>=len(negtive_index):
            neg_choice = np.random.choice(negtive_index, delete_neg, replace=True)
        else:
            neg_choice=np.random.choice(negtive_index,delete_neg,replace=False)
        box_infos[neg_choice]=0
        box_locs[positive_index,:4]=1
        x_off, y_off, w_off, h_off=self.quick_offset(boxes[positive_index],gt_box)
        box_locs[positive_index, 4]=x_off
        box_locs[positive_index, 5]=y_off
        box_locs[positive_index, 6]=w_off
        box_locs[positive_index, 7]=h_off
        box_locs_valid=box_locs[:,:4].reshape(f_height, f_width, 4 * cfg.TRAIN.ANCHOR_NUM)
        box_locs_offset=box_locs[:,4:].reshape(f_height, f_width, 4 * cfg.TRAIN.ANCHOR_NUM)
        box_infos_valid=box_infos[:,:2].reshape(f_height, f_width, 2 * cfg.TRAIN.ANCHOR_NUM)
        box_infos_label = box_infos[:, 2:].reshape(f_height, f_width, 2 * cfg.TRAIN.ANCHOR_NUM)
        box_infos=np.concatenate((box_infos_valid,box_infos_label),axis=2)
        box_locs=np.concatenate((box_locs_valid,box_locs_offset),axis=2)
        return box_infos,box_locs


    def box_judger(self, boxes, f_height, f_width, whidth, height, gt_box):
        """
        The method to generate cls.-1 means negative,0 means useless anchor.
        and make sure the anchor box is meaningful
        :param boxes: the anchor boxes(cx,cy,w,h).
        :return: Classification of boxes.
        """
        box_infos = np.zeros((f_height, f_width, 4 * cfg.TRAIN.ANCHOR_NUM), dtype=np.float32)
        box_locs = np.zeros((f_height, f_width, 8 * cfg.TRAIN.ANCHOR_NUM), dtype=np.float32)
        iou_record = np.zeros((f_height, f_width, cfg.TRAIN.ANCHOR_NUM))
        best_iou=0
        exist_pos=False
        # traverse each vector in feature map and estimate whether it is valid and the label of anchor boxes.
        start=time.clock()
        part1=0
        part2=0
        for i in range(f_height):
            for j in range(f_width):
                valid =[]
                boxes_type = []
                for x in range(0, len(boxes[i, j]), 4):
                    start = time.clock()
                    box = boxes[i, j, x:x+4]
                    box_is_valid =self.valid_box_check(box, whidth, height)
                    iou_value =self.iou(box, gt_box)
                    end = time.clock()
                    part1 += end - start
                    if box_is_valid and iou_value > 0:
                        iou_record[i, j, int(x / 4)] = iou_value
                        best_iou=max(best_iou,iou_value)
                    box_type = -1
                    if iou_value > cfg.TRAIN.IOUPOS and box_is_valid:
                        box_type = 1
                    # the anchors that we do not use to training.
                    # if iou_value < cfg.TRAIN.IOUPOS and iou_value > cfg.TRAIN.IOUNEG:
                    if iou_value < 0.05 and iou_value>0:
                        box_type = 0
                    valid+=[box_is_valid]
                    boxes_type+=[box_type, 1 - box_type]
                    end2 = time.clock()
                    part2 += end2 - end
                box_info = []
                box_loc = []
                for x in range(len(valid)):
                    # handle the anchors which are not positive.
                    if valid[x] + boxes_type[2 * x] != 2:
                        box_loc = box_loc + 4 * [0]
                    else:
                        box_loc = box_loc + 4 * [1]
                        exist_pos=True
                        boxes[i, j, 4 * x:4 * x + 4] = self.cal_offset(boxes[i, j, 4 * x:4 * x + 4], gt_box)
                    box_info = box_info + 2 * [valid[x]]
                box_locs[i, j] = box_loc + list(boxes[i, j])
                box_infos[i, j] = box_info + list(boxes_type)
        end = time.clock()
        print("the step 1: " + str(end - start))
        if not exist_pos:
            self.best_iou_label(best_iou,box_locs,box_infos,iou_record,gt_box)
        end3 = time.clock()
        print("the step 2: " + str(end3 - end))
        self.clip_data(box_infos)
        end4 = time.clock()
        print("the step 3: " + str(end4 - end3))
        return box_infos, box_locs

    def clip_data(self, box_infos):
        '''
        The method to keep negative data and positive data balance
        :return:the balance data sample to training
        '''
        box_valid = box_infos[:, :, :2 * cfg.TRAIN.ANCHOR_NUM]
        box_type = box_infos[:, :, 2 * cfg.TRAIN.ANCHOR_NUM:]
        box_type = box_type[:, :, ::2]
        box_valid = box_valid[:, :, ::2]
        #let the negative and positave sample ratio is 1:5.
        num=len(np.where(np.logical_and(np.equal(box_type,1),np.equal(box_valid,1)))[0])*10
        hard_neg_index = np.where(np.logical_and(np.equal(box_type, 0), np.equal(box_valid, 1)))
        neg_index = np.where(np.logical_and(np.equal(box_type, -1), np.equal(box_valid, 1)))
        if len(hard_neg_index[0]) >= num:
            # reduce hard negetive sample
            delet_num = int(len(hard_neg_index[0]) - num)
            delete_index = random.sample(range(len(hard_neg_index[0])), delet_num)
            cancel_index = np.squeeze([i[delete_index] for i in hard_neg_index])
            box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
            box_valid = box_infos[:, :, :2 * cfg.TRAIN.ANCHOR_NUM]
            box_valid = box_valid[:, :, 1::2]
            box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
            # reduce easy negetive sample.
            cancel_index = np.squeeze([i for i in neg_index])
            box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
            box_valid = box_infos[:, :, :2 * cfg.TRAIN.ANCHOR_NUM]
            box_valid = box_valid[:, :, ::2]
            box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0

        if len(hard_neg_index[0]) < num:
            # delete the extra easy negetive samples.
            delete_num = int(len(neg_index[0]) - (num - len(hard_neg_index[0])))
            delete_index = random.sample(range(len(neg_index[0])), delete_num)
            cancel_index = np.squeeze([i[delete_index] for i in neg_index])
            box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
            box_valid = box_infos[:, :, :2 * cfg.TRAIN.ANCHOR_NUM]
            box_valid = box_valid[:, :, 1::2]
            box_valid[cancel_index[0], cancel_index[1], cancel_index[2]] = 0
            # change the label of easy negative sample
            box_valid = box_infos[:, :, :2 * cfg.TRAIN.ANCHOR_NUM]
            box_valid = box_valid[:, :, ::2]
            box_type = box_infos[:, :, 2 * cfg.TRAIN.ANCHOR_NUM:]
            box_type = box_type[:, :, ::2]
            change_index = np.where(np.logical_and(np.equal(box_type, -1), np.equal(box_valid, 1)))
            box_type[change_index]=0
            box_type = box_infos[:, :, 2 * cfg.TRAIN.ANCHOR_NUM:]
            box_type = box_type[:, :, 1::2]
            box_type[change_index] = 1




