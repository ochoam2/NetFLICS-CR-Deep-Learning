# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:45:02 2018

@author: mry09
"""

from keras.layers import add
from keras.layers import Conv1D, Conv2D, BatchNormalization
from keras.layers.core import Activation

# define blocks for FLICS-Net

def resblock_1D(num_filters, size_filter, x):
    Fx = Conv1D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = Activation('relu')(Fx)
    Fx = Conv1D(num_filters, size_filter, padding='same', activation=None)(Fx)
    output = add([Fx, x])
    output = Activation('relu')(output)
    return output

def resblock_1D_BN(num_filters, size_filter, x):
    Fx = Conv1D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation('relu')(Fx)
    Fx = Conv1D(num_filters, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    output = add([Fx, x])
#    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output

def resblock_1D_BN2(num_filters, size_filter, x):
    Fx = Conv1D(num_filters, size_filter, padding='same', activation='relu')(x)
    Fx = BatchNormalization()(Fx)
    Fx = Conv1D(num_filters, size_filter, padding='same', activation='relu')(Fx)
    Fx = BatchNormalization()(Fx)
    output = add([Fx, x])
    output = Activation('relu')(output)
    return output

def resblock_2D(num_filters, size_filter, x):
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = Activation('relu')(Fx)
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(Fx)
    output = add([Fx, x])
    output = Activation('relu')(output)
    return output

def resblock_2D_BN(num_filters, size_filter, x):
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation('relu')(Fx)
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    output = add([Fx, x])
#    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output

def resblock_2D_BN2(num_filters, size_filter, x):
    Fx = Conv2D(num_filters, size_filter, padding='same', activation='relu')(x)
    Fx = BatchNormalization()(Fx)
    Fx = Conv2D(num_filters, size_filter, padding='same', activation='relu')(Fx)
    Fx = BatchNormalization()(Fx)
    output = add([Fx, x])
    output = Activation('relu')(output)
    return output