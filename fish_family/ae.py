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

def conv_block(num_kernels, kernel_size, stride, activation='relu'):
    def f(prev):
        conv = prev
        conv = Conv2D(filters=num_kernels, kernel_size=(kernel_size,1), padding='same', strides=(stride,1))(conv)
        conv = BatchNormalization()(conv)
        conv = Activation(activation)(conv)
        conv = MaxPooling2D(pool_size=(2,1))(conv)
        return conv

    return f

def deconv_block(num_kernels, kernel_size, upsampling, activation='relu'):
    def f(prev):
        deconv = prev
        deconv = UpSampling2D(size=(upsampling, 1))(deconv)
        #deconv = Conv2DTranspose(filters=num_kernels, kernel_size=(kernel_size,1), padding='same')(deconv)
        deconv = Conv2D(filters=num_kernels, kernel_size=(kernel_size,1), padding='same')(deconv)
        deconv = BatchNormalization()(deconv)
        deconv = Activation(activation=activation)(deconv)
        return deconv

    return f

def _decoder(num_kernels_arr, kernels_sizes_arr, upsemblings_arr, n_channles):
        def f(input):
            x = input
            for i in range(len(num_kernels_arr)):
                num_kernels = num_kernels_arr[i]
                kernel_size = kernels_sizes_arr[i]
                upsampling = upsemblings_arr[i]
                x = deconv_block(num_kernels, kernel_size, upsampling)(x)
            x = deconv_block(num_kernels=n_channles, kernel_size=5, upsampling=2, activation='linear')(x)
            return x

        return f

def _encoder(num_kernels_arr, kernels_sizes_arr, strides_arr):
    def f(input):
        x = input
        for i in range(len(num_kernels_arr)):
            num_kernels = num_kernels_arr[i]
            kernel_size = kernels_sizes_arr[i]
            stride = strides_arr[i]
            x = conv_block(num_kernels, kernel_size, stride)(x)
        x = conv_block(num_kernels=1, kernel_size=3, stride=1, activation='linear')(x)
        return x
    return f

class AE:
    def __init__(self):
        self.ecg_segment_len = 512
        self.n_channles = 12

        self.num_kernels_arr=[15, 20, 15]
        self.kernels_sizes_arr=[5, 3,3]
        self.strides_arr = [2, 1,1]

    def encoder(self):
        return _encoder(num_kernels_arr=self.num_kernels_arr,
                        kernels_sizes_arr=self.kernels_sizes_arr,
                        strides_arr=self.strides_arr)

    def decoder(self):
        num_kernels_arr = list(reversed(self.num_kernels_arr))
        upsemblings_arr=[2*x for x in self.strides_arr]
        kernels_sizes_arr = list(reversed(self.kernels_sizes_arr))
        return _decoder(num_kernels_arr=num_kernels_arr,
                        kernels_sizes_arr=kernels_sizes_arr,
                        upsemblings_arr=upsemblings_arr,
                        n_channles=self.n_channles)


    def make_net(self):
        input_shape = (self.ecg_segment_len, 1, self.n_channles)
        input = Input(shape=input_shape)
        code = self.encoder()(input)
        model = Model(input, self.decoder()(code))
        model.summary()
        return model

ae = AE()
ae.make_net()
