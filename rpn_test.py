from data_handler.data_generator_rpn import data_generator_rpn
from network.resnet import resnet
from tools.debug import draw_rpn_result
import numpy as np


def generate_cx(cls_layer):
    object=cls_layer[0,:,:,0::2]
    a = np.array(object).ravel()
    a = a.argsort()[::-1][:500]
    height_index, width_index, large_index = np.unravel_index(a, (
    object.shape[0], object.shape[1], object.shape[2]))
    index=np.where(np.greater(object,0.9))
    return height_index, width_index, large_index

def test_rpn():
    net=resnet()
    generator=data_generator_rpn()
    fpn_rpn=net.fpn_net()
    fpn_rpn.load_weights("/Network/Servers/lab.cs.uwlax.edu/nfs-homes/zhou2494/Desktop/expirements/fpn_model8.h5",by_name=True)
    for i in range(100):
        data=next(generator.generator)
        output=fpn_rpn.predict_on_batch(data[0])
        xclass_small, xloc_small, xclass_mid, xloc_mid, xclass_large, xloc_large=output
        cy, cx, type = generate_cx(xclass_large)
        draw_rpn_result(cy, cx, type, xloc_large, data[0], 8, data[2])
        cy,cx,type=generate_cx(xclass_small)
        draw_rpn_result(cy,cx,type,xloc_small,data[0],32,data[2])
        cy, cx, type = generate_cx(xclass_mid)
        draw_rpn_result(cy, cx, type, xloc_mid, data[0], 16,data[2])

if __name__ == '__main__':

    test_rpn()