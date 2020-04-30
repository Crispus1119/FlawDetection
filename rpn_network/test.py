from data_handler.data_generator_fpn import data_generator
from tools.config import cfg
from network.resnet import resnet
import cv2
import numpy as np
def transform_reg(X,T):
    """
    The method to transfer anchor box by regression value.
    :param X:
    :param T:
    :return:
    """
    xmin, ymin, w, h = X
    cx=xmin+w/2
    cy=ymin+h/2
    tx, ty, tw, th = T
    cx1 = tx * w + cx
    cy1 = ty * h + cy
    w1 = np.exp(tw.astype(np.float32)) * w
    h1 = np.exp(th.astype(np.float32)) * h
    x1 = cx1 - w1 / 2.
    y1 = cy1 - h1 / 2.
    return np.stack([x1, y1, w1, h1])


def generator_anchor(rpn_stride):
    """
    The method to generator anchor box.
    :return: anchor boxes.
    """
    ratios = cfg.TRAIN.RATIOS
    if rpn_stride == 32:
        scales = cfg.TRAIN.LSCALES
    if rpn_stride == 16:
        scales = cfg.TRAIN.MSCALES
    if rpn_stride == 8:
        scales = cfg.TRAIN.SSCALES
    scales, ratios = np.meshgrid(scales, ratios)
    scaleX = scales * np.sqrt(ratios)
    scaleY = scales / np.sqrt(ratios)
    larges = np.stack([scaleX.flatten(), scaleY.flatten()], axis=1)
    return scaleX,scaleY,larges

def drawbox(im,color,loc,thik):
    # :param loc: [xmin,ymin,w,h]
    image=cv2.rectangle(im,(int(loc[0]),int(loc[1])),(int(loc[0]+loc[2]),int(loc[1]+loc[3])),color, thik)
    return image

def get_index(box_infos,box_loc):
    # the method to get the index of the box that is positive for training data
    anchor_num = cfg.TRAIN.ANCHOR_NUM
    box_infos=np.squeeze(box_infos)
    box_loc = np.squeeze(box_loc)
    box_valid = box_infos[:, :, :2*anchor_num].reshape((-1, 2))
    box_type = box_infos[:, :, 2*anchor_num:].reshape((-1, 2))
    box_loc_info=box_loc[:, :, :4*anchor_num]
    box_loc = box_loc[:, :, 4 * anchor_num:].reshape((-1,4))
    index2=np.where(np.equal(box_type[:,0],1))[0]
    index = np.where(np.logical_and(np.equal(box_type[:, 0], 1),np.equal(box_valid[:,0],1)))[0]
    neg_index=np.where(np.logical_and(np.equal(box_type[:, 0], 0),np.equal(box_valid[:,0],1)))[0]
    tol_index=np.where(np.equal(box_valid[:,0],1))[0]
    assert len(tol_index)==len(neg_index)+len(index),"your index number is not correct"
    box_check=box_loc_info.reshape((-1,4))
    index_valid = np.where(np.equal(box_check[:, 0], 1))[0]
    assert all(index_valid == index), 'you index for regression is not right'
    return index,box_loc

def check_data_generator(rpn_y,rpn_stride,name,im):
    """
    The method to draw box to check rpn network data generator
    :param rpn_x:
    :param rpn_y: the box_cls[1,h,w,36],the box_reg[1,w,h,72]
    :param img_info:
    :return:
    """
    std = cfg.TRAIN.std_scaling
    box_cls,box_loc=rpn_y
    f_width = box_loc.shape[1]
    f_height = box_loc.shape[0]
    index, box_loc=get_index(box_cls,box_loc)
    draw_boxes_reg=box_loc[index]/std
    # f_width,f_height=network.cal_fm_size(im.width,im.height)
    np_im=np.array(im)
    # w = gt_box[2] - gt_box[0]
    # h = gt_box[3] - gt_box[1]
    # np_im = drawbox(np_im, (255, 0, 0), [gt_box[0], gt_box[1], w, h], 3)
    scaleX, scaleY, larges = generator_anchor(rpn_stride)
    record=[]
    for h in range(f_height):
        for w in range(f_width):
            for large in larges:
                record.append([w*rpn_stride+rpn_stride/2-large[0]/2,h*rpn_stride+rpn_stride/2-large[1]/2,large[0],large[1]])
    record=np.array(record)
    draw_box=record[index]
    # print('valid positive number is : '+str(len(draw_boxes_reg)))
    for i in range(len(draw_boxes_reg)):
        np_im=drawbox(np_im,(0,255,0),draw_box[i],1)
        box=transform_reg(draw_box[i],draw_boxes_reg[i])
        np_im=drawbox(np_im,(0,0,255),box,1)
    cv2.imwrite('tst'+name+'.png',np_im)

def test():
    d=data_generator().g
    flag=0
    for i in range(100):
        data=next(d)
        image = data[0]
        small_cls_output, small_reg_output, mid_cls_output, mid_reg_output, large_cls_output, large_reg_output = data[1]
        for n in range(8):
            im=image[n,:,:,:]
            check_data_generator([small_cls_output[n,:,:,:], small_reg_output[n,:,:,:]],32,str(flag),im)
            flag+=1
            check_data_generator([mid_cls_output[n,:,:,:], mid_reg_output[n,:,:,:]], 16, str(flag), im)
            flag+=1
            check_data_generator([large_cls_output[n,:,:,:], large_reg_output[n,:,:,:]], 8, str(flag), im)
            flag += 1
if __name__ == '__main__':
    test()