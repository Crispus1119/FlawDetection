import sys
import numpy as np
def TraceStack():
    """
    The method to print the method using to debug
    """
    # _getframe(0) means get current framework,_getframe(1)means get one depth frame
    frame = sys._getframe(1)
    index=0
    while frame:
        print("--------------------------------")
        print('depth: '+str(index))
        print('method name is : '+frame.f_code.co_name)
        print('from: ')
        print( frame.f_code.co_filename)
        print('In line: '+str(frame.f_lineno))
        # the method to go deep
        frame = frame.f_back
        index+=1

def compare_weight(model,filepath):
    """
    The method to check whether model successfully load weight from file path
    """
    preloaded_layers = model.layers.copy()
    preloaded_weights = []
    for pre in preloaded_layers:
        preloaded_weights.append(pre.get_weights())
    # load pre-trained weights
    model.load_weights(filepath, by_name=True)

    # compare previews weights vs loaded weights
    for layer, pre in zip(model.layers, preloaded_weights):
        weights = layer.get_weights()
        if weights:
            if np.array_equal(weights, pre):
                print('not loaded', layer.name)
            else:
                print('loaded', layer.name)