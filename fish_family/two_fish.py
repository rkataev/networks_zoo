# -*- coding: utf-8 -*
#
import numpy as np
import math
import easygui
from keras.models import load_model
from keras.layers import (
    Input, Flatten,
    BatchNormalization,
    Activation, Dense, Dropout,
    Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Conv2DTranspose,
    UpSampling1D, AtrousConvolution1D, SeparableConv2D, Conv1D
)

from keras.layers import merge


from keras.models import Model
from keras.losses import (
    mean_squared_error
)
from keras.optimizers import (
    adam, sgd, adadelta
)

import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf

from caterpillar_feeder import ecg_batches_generator
from dataset_getter import prepare_data
from utils import (
    draw_reconstruction_to_png, save_history
)

def conv_block(num_kernels, kernel_size, stride, name, activation='relu'):
    def f(prev):
        conv = prev
        conv = Conv2D(filters=num_kernels, kernel_size=(kernel_size,1), padding='same', strides=(stride,1))(conv)
        conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)
        conv = MaxPooling2D(pool_size=(2,1))(conv)
        return conv

    return f

ecg_segment_len = 200
def encoder():
    in_shape = (ecg_segment_len, 1, 1)
    x = Input(shape=in_shape)
    f = conv_block(num_kernels=2, kernel_size=(5,1), stride=(3, 1), name="encoded_1", activation='relu')
    x = f(x)

