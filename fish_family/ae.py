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

def conv_block(num_kernels, kernel_size, stride):
    def f(prev):
        conv = prev
        conv = SeparableConv2D(filters=num_kernels, kernel_size=(kernel_size,1), padding='same', strides=(stride,1))(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('linear')(conv)
        conv = MaxPooling2D(pool_size=(2,1))(conv)
        return conv

    return f

def deconv_block(num_kernels, kernel_size, upsampling):
    def f(prev):
        deconv = prev
        deconv = UpSampling2D(size=(upsampling, 1))(deconv)
        #deconv = Conv2DTranspose(filters=num_kernels, kernel_size=(kernel_size,1), padding='same')(deconv)
        deconv = SeparableConv2D(filters=num_kernels, kernel_size=(kernel_size,1), padding='same')(deconv)
        deconv = BatchNormalization()(deconv)
        deconv = Activation('linear')(deconv)
        return deconv

    return f



class AE:
    def __init__(self):
        self.ecg_segment_len = 512
        self.n_channles = 12

        self.num_kernels_arr=[10, 13]
        self.kernels_sizes_arr=[5, 3]
        self.strides_arr = [2, 2]

    def encoder(self):
        def f(input):
            x = input
            for i in range(len(self.num_kernels_arr)):
                num_kernels = self.num_kernels_arr[i]
                kernel_size = self.kernels_sizes_arr[i]
                stride = self.strides_arr[i]
                x = conv_block(num_kernels, kernel_size, stride)(x)
            x = conv_block(num_kernels=1, kernel_size=3, stride=1)(x)

            return x
        return f

    def decoder(self):
        num_kernels_arr = list(reversed(self.num_kernels_arr))
        upsemblings_arr=[2*x for x in self.strides_arr]
        kernels_sizes_arr = list(reversed(self.kernels_sizes_arr)
        def f(input):
            x = input
            for i in range(len(num_kernels_arr)):
                num_kernels = num_kernels_arr[i]
                kernel_size = kernels_sizes_arr[i]
                upsampling = upsemblings_arr[i]
                x = deconv_block(num_kernels, kernel_size, upsampling)(x)
            x = deconv_block(num_kernels=self.n_channles, kernel_size=5, upsampling=2)(x)
            return x

        return f

    def make_net(self):
        input_shape = (self.ecg_segment_len, 1, self.n_channles)
        input = Input(shape=input_shape)
        code = self.encoder()(input)
        model = Model(input, self.decoder()(code))
        model.summary()
        return model

ae = AE()
ae.make_net()
