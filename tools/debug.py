import numpy as np
import cv2
from tools.config import cfg
from network.resnet import resnet
anchor_num=cfg.TRAIN.ANCHOR_NUM


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
    x1 = np.round(x1)
    y1 = np.round(y1)
    w1 = np.round(w1)
    h1 = np.round(h1)
    return np.stack([x1, y1, w1, h1])


def loadimg(img_info,generator):
    # the method to load Image
    im = generator.image_process(img_info)
    gt_box = img_info.get('box')
    im, box = generator.handle_origin_image(im, gt_box)
    return im, box

def drawbox(im,color,loc,thik,name):
    # :param loc: [xmin,ymin,w,h]
    font = cv2.FONT_HERSHEY_SIMPLEX
    image=cv2.rectangle(im,(int(loc[0]),int(loc[1])),(int(loc[0]+loc[2]),int(loc[1]+loc[3])),color, thik)
    image=cv2.putText(image,name,(int(loc[0]),int(loc[1]-7)),font, 1, (0,0,255), 2)
    return image

def get_index(box_infos,box_loc):
    # the method to get the index of the box that is positive for training data
    box_infos=np.squeeze(box_infos)
    box_loc = np.squeeze(box_loc)
    box_valid = box_infos[:, :, :2*anchor_num].reshape((-1, 2))
    box_type = box_infos[:, :, 2*anchor_num:].reshape((-1, 2))
    box_loc_info=box_loc[:, :, :4*anchor_num]
    box_loc = box_loc[:, :, 4 * anchor_num:].reshape((-1,4))
    index = np.where(np.logical_and(np.equal(box_type[:, 0], 1),np.equal(box_valid[:,0],1)))[0]
    neg_index = np.where(np.logical_and(np.equal(box_type[:, 0], 0), np.equal(box_valid[:, 0], 1)))[0]
    print("the negative sample number is :"+str(len(neg_index)))
    print("the total positive sample is :"+str(len(index)))
    box_check=box_loc_info.reshape((-1,4))
    assert len(box_check)==len(box_type), "your shape is not right"
    reg_valid=np.where(box_check[:,0]==1)[0]
    print(reg_valid)
    print(index)
    assert all(reg_valid==index),'you index for regression is not right'
    index_valid = np.where(np.equal(box_check[:, 0], 1))[0]
    # box_loc = boxes[:, 4*anchor_num:].reshape(-1, 4)
    index = np.intersect1d(index, index_valid)
    return index,box_loc


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
    return larges


def draw_rpn_result(cy,cx,type,rpn_reg,np_im,std,img_info):
    np_im=np.squeeze(np_im)
    xmax=np_im.shape[1]
    ymax=np_im.shape[0]
    regulation = cfg.TRAIN.std_scaling
    rpn_reg = rpn_reg /regulation
    x_index=cx.copy()
    y_index=cy.copy()
    cx=cx*std+std/2
    cy=cy*std+std/2
    larges=generator_anchor(std)
    rpn_reg=np.squeeze(rpn_reg)
    print("the length of cx:: "+str(len(cx)))
    print(type)
    for i in range(len(cx)):
       large=larges[type[i]]
       origin_box=[cx[i],cy[i],large[0],large[1]]
       np_im = drawbox(np_im, (0, 255,0), [origin_box[0]-origin_box[2]/2,origin_box[1]-origin_box[3]/2,origin_box[2],origin_box[3]], 1)
       box = cal_transform(origin_box, rpn_reg[y_index[i],x_index[i],4*type[i]:4*type[i]+4])
       if box[0]<0 or box[1]<0 or box[0]+box[2]>xmax or box[1]+box[3]>ymax:
          continue
       np_im = drawbox(np_im, (0, 0, 255), box, 3)
    cv2.imwrite('tst'+img_info.get('index') +str(std)+ '.png', np_im)

def check_data_generator(generator,rpn_y,img_info,rpn_stride,name):
    """
    The method to draw box to check rpn network data generator
    :param rpn_x:
    :param rpn_y: the box_cls[1,h,w,36],the box_reg[1,w,h,72]
    :param img_info:
    :return:
    """
    std = cfg.TRAIN.std_scaling
    network=resnet()
    box_cls,box_loc=rpn_y
    f_width = box_loc.shape[2]
    f_height = box_loc.shape[1]
    index, box_loc=get_index(box_cls,box_loc)
    draw_boxes_reg=box_loc[index]/std
    im,box=loadimg(img_info,generator)
    print(img_info.get('index'))
    # f_width,f_height=network.cal_fm_size(im.width,im.height)
    np_im=np.array(im)
    gt_box=img_info.get('change_box')
    w = gt_box[2] - gt_box[0]
    h = gt_box[3] - gt_box[1]
    np_im = drawbox(np_im, (255, 0, 0), [gt_box[0], gt_box[1], w, h], 3)
    larges = generator_anchor()
    record=[]
    for h in range(f_height):
        for w in range(f_width):
            for large in larges:
                record.append([w*rpn_stride-large[0]/2,h*rpn_stride-large[1]/2,large[0],large[1]])
    record=np.array(record)
    draw_box=record[index]
    print('valid positive number is : '+str(len(draw_boxes_reg)))
    for i in range(len(draw_boxes_reg)):
        np_im=drawbox(np_im,(0,255,0),draw_box[i],1)
        box=transform_reg(draw_box[i],draw_boxes_reg[i])
        np_im=drawbox(np_im,(0,0,255),box,1)
    cv2.imwrite('tst'+img_info.get('index')+name+'.png',np_im)


def draw_roi(rois,img_info,image):
    """
    The method to draw roi that input of fast rcnn
    :param rois: [x,y,w,h]
    :return:
    """
    rois=rois.reshape((-1,4))
    ge_box=img_info.get("change_box")
    np_im = np.array(image)
    np_im=drawbox(np_im,(0,255,0),[ge_box[0],ge_box[1],ge_box[2]-ge_box[0],ge_box[3]-ge_box[1]],1)
    for roi in rois:
        np_im=drawbox(np_im,(255,0,0),[roi[0],roi[1],roi[2]-roi[0],roi[3]-roi[1]],3)
    cv2.imwrite('tst' + img_info.get('index') + '.png', np_im)