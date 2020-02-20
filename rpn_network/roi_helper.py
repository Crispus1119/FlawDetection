from tools.config import cfg
import numpy as np
import math
from network.resnet import resnet
def cal_transform(X,T):
    """
    Calculate the transform from regression layer(t) in paper
    :param X: the anchor box in feature map(xc,yc,width,height)
    :param T:the rpn_reg layer output(xc,yc,width,height)
    :return: the box after regression(xmin,xmax,w,h)
    """
    cx,cy,w,h=X
    tx,ty,tw,th=T
    cx1 = tx * w + cx
    cy1 = ty * h + cy
    w1 = np.exp(tw.astype(np.float32)) * w
    h1 = np.exp(th.astype(np.float32)) * h
    x1 = cx1 - w1 / 2.
    y1 = cy1 - h1 / 2.
    return np.stack([x1, y1, w1, h1])

def nms(box):
    """
    Non-Maximum Suppression
    :param box: the boxes
    :param probs:the probilities
    :return: top probability box.
    """
    xmin=box[:,0]
    ymin=box[:,1]
    xmax=box[:,2]
    ymax=box[:,3]
    score=box[:,4]
    # why it should add 1?because you know it is feature map
    # each one feature node means a lot in origin image.
    areas=(xmax-xmin)*(ymax-ymin)
    # get the index of nms score.
    index=np.argsort(score)[::-1]
    return box[index[40]]
    # store=[]
    # while index.size>0:
    #     i=index[0]
    #     store.append(i)
    #     # in numpy array list means the index of array.
    #     # so now the order of  xx1,yy1,xx2,yy2 is the order like index.
    #     xx1=np.maximum(xmin[i],xmin[index[1:]])
    #     yy1=np.maximum(ymin[i],ymin[index[1:]])
    #     xx2=np.minimum(xmax[i],ymax[index[1:]])
    #     yy2=np.minimum(ymax[i],ymax[index[1:]])
    #     # if the two box are not intersecting,w and h will be 0
    #     w = np.maximum(0.0, xx2 - xx1)
    #     h = np.maximum(0.0, yy2 - yy1)
    #     intersection=w*h
    #     iou=intersection/(areas[i]+areas[index[1:]]-intersection)
    #     loc=np.where(iou<cfg.TRAIN.NMS_THRESHOLD)[0]
    #     # add one cause if the new xx1,xx2....is begin with 1.
    #     index=index[loc+1]
    #     if len(store)>=20:
    #         break
    # return box[store]

def rpn_to_roi(cls_layer,reg_layer):
    '''
    the method to translate rpn network output layer to rois which in feature map.
    # the center of anchor box in feature map is the point of(H,W)
    :param cls_layer: the output of cls layer shape is(1,None,None,18). the box[0] is score of
    :param reg_layer: the output of reg layer shaoe is (1,None,None,36)
    :return: the rois(xmin,ymin,xmax,ymax)
    '''
    map_height=cls_layer.shape[1]
    map_width=cls_layer.shape[2]
    ratios = cfg.TRAIN.RATIOS
    scales = cfg.TRAIN.SCALES
    rpn_stride = cfg.TRAIN.RPN_STRIDE
    scales, ratios = np.meshgrid(scales, ratios)
    # the scaleX and scaleY in feature map.
    scaleX = scales * np.sqrt(ratios)
    scaleY = scales / np.sqrt(ratios)
    larges = np.stack([scaleX.flatten(), scaleY.flatten()], axis=1)
    boxes=np.zeros((map_height,map_width,len(larges),5))
    for h in range(map_height):
        for w in range(map_width):
            label=0
            for large in larges:
                # anchor_box should be[centerX,centerY,w,h]
                anchor_box=[w*rpn_stride,h*rpn_stride,large[0],large[1]]
                reg=np.squeeze(reg_layer[:,h,w,label*4:label*4+4])/cfg.TRAIN.std_scaling
                # the box is [xmin,ymin,W,H]
                box=cal_transform(anchor_box,reg)
                boxes[h,w,label]=np.concatenate((box,[cls_layer[0,h,w,label*2]]),axis=0)
                label+=1
    # because when we calculate np.round() make width and height be 0.
    # boxes[:,:,:,2]=np.maximum(1,boxes[:,:,:,2])
    # boxes[:, :, :, 3] = np.maximum(1, boxes[:, :, :, 3])
    # change box to [xmin,ymin,xmax,ymax]
    boxes[:,:,:,2]+=boxes[:,:,:,0]
    boxes[:,:,:,3]+=boxes[:,:,:,1]
    # make sure the box is not out of bounding of feature map.
    boxes[:,:,:,0]=np.maximum(0,boxes[:,:,:,0])
    boxes[:, :, :, 1]=np.maximum(0,boxes[:,:,:,1])
    boxes[:, :, :, 2]=np.minimum((map_width-1)*rpn_stride,boxes[:,:,:,2])
    boxes[:, :, :, 3]=np.minimum((map_height-1)*rpn_stride,boxes[:,:,:,3])
    boxes=np.reshape(boxes,(-1,5))
    # probs=np.reshape(cls_layer,(-1,2))
    # assert boxes.shape[0]==probs.shape[0]," your boxes and probability's number is not same"
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # delete the box that is totally wrong
    ids = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    all_boxes = np.delete(boxes, ids, 0)
    # all_probs = np.delete(probs, ids, 0)
    boxes=nms(all_boxes)
    # [xmin,ymin,xmax,ymax]
    return boxes[:,:5],map_height-1,map_width-1


def handle_roi_fm(rois,map_height,map_width):
    # the method to implement for inference time to clip roi to feed into roi pooling
    roi_record=[]
    for roi in rois:
        xmin,ymin,xmax,ymax=roi
        w = np.round(np.maximum(1, xmax - xmin))
        h = np.round(np.maximum(1, ymax - ymin))
        xmin = np.round(xmin)
        ymin = np.round(ymin)
        if xmin + w > map_width:
            if xmin == map_width:
                xmin -= 1
            else:
                w = map_width - xmin
        if ymin + h > map_height:
            if ymin == map_height:
                ymin -= 1
            else:
                h = map_height - ymin
        roi_record.append([xmin, ymin, w, h])
    return roi_record


def cal_offset(box,gt_box):
        """
        The method to calculate the offset of roi boxes and gt_box on feature map.
        :param box: the roi box(xmin,ymin,xmax,ymax)
        :param gt_box: the ground truth box(xmin,ymin,xmax,ymax)
        :return:
        """
        sx,sy,sw,sh = cfg.TRAIN.classifier_regr_std
        box_width=box[2]-box[0]
        box_height=box[3]-box[1]
        box=[box[0]+box_width/2,box[1]+box_height/2,box_width,box_height]
        gt_box_width = gt_box[2] - gt_box[0]
        gt_box_height = gt_box[3] - gt_box[1]
        gt_box = [gt_box[0] + gt_box_width /2, gt_box[1] + gt_box_height / 2, gt_box_width, gt_box_height]
        x_off = (gt_box[0] - box[0]) / box[2]*sx
        y_off=(gt_box[1] - box[1]) / box[3]*sy
        w_off = math.log(gt_box[2] / box[2])*sw
        h_off=math.log(gt_box[3]/box[3])*sh
        return [x_off,y_off,w_off,h_off]

def match_gt_box(rois,img_info,map_height,map_width):
    """
    The method to set label for rois to feed in Fast RCNN.
    :param rois: the rois box(xmin,ymin,xmax,ymax).
    :param img_info: the image_info which change_box(xmin,ymin,xmax,ymax)
    :return:the data to feed in Fast RCNN
    """
    # in this method i can implement to get the rois that i can not classfiy
    img_class=img_info.get('class')
    # gt_box is (xmin,xmax,xmax,ymax)
    gt_box=img_info.get('change_box')
    for i in range(4):
        gt_box[i]=gt_box[i]/cfg.TRAIN.RPN_STRIDE
    gt_box_square=(gt_box[2]-gt_box[0])*(gt_box[3]-gt_box[1])
    roi_loc=[]
    label_record=[]
    roi_record=[]
    for roi in rois:
        xmin,ymin,xmax,ymax=roi
        gt_xmin,gt_ymin,gt_xmax,gt_ymax=gt_box
        xxmin=np.maximum(xmin,gt_xmin)
        yymin=np.maximum(ymin,gt_ymin)
        xxmax=np.minimum(xmax,gt_xmax)
        yymax=np.minimum(ymax,gt_ymax)
        if yymax-yymin<0 or xxmax-xxmin<0:
            continue
        roi_square = (roi[2] - roi[0]) * (roi[3] - roi[1])
        w=xxmax-xxmin
        h=yymax-yymin
        iou=w*h/(roi_square+gt_box_square-w*h)
        print(str(roi_square)+":::::::"+str(gt_box_square)+":::::"+str(iou))
        if iou<0.1:
            continue
        else:
            # hard negative mining
            # this can use regression to optimize
            # the last 4 loc is store the regression value
            class_num=cfg.NUM_CLASSES-1
            record_loc = [0] * 8 * class_num
            if iou<0.3:
               gt_class = np.zeros([cfg.NUM_CLASSES], dtype=np.int32)
               gt_class[-1]=1
               roi_loc.append(record_loc)
               label_record.append(gt_class)
               roi_record.append(roi)
            else:
                label_record.append(img_info.get('class'))
                # the reg layer is for each class
                index=np.where(img_info.get('class')==1)[0][0]
                record_loc[4*index:4*index+4]=[1,1,1,1]
                # roi[xmin,ymin,xmax,ymax]
                # gt_box[xmin,ymin,xmax,ymax]
                record_loc[4*(class_num+index):4*(class_num+index+1)]=cal_offset(roi,gt_box)
                roi_loc.append(record_loc)
                w=np.round(np.maximum(1,xmax-xmin))
                h=np.round(np.maximum(1,ymax-ymin))
                xmin=np.round(xmin)
                ymin=np.round(ymin)
                if xmin+w>map_width:
                    if xmin==map_width:
                        xmin-=1
                    else:
                        w=map_width-xmin
                if ymin+h>map_height:
                    if ymin==map_height:
                        ymin-=1
                    else:
                        h=map_height-ymin
                roi_record.append([xmin,ymin,w,h])
        roi_loc=np.array(roi_loc,dtype='float32')
        if len(roi_loc) == 0:
            return None
        return np.expand_dims(roi_record,axis=0),np.expand_dims(label_record,axis=0),np.expand_dims(roi_loc,axis=0)








