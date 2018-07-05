#
import numpy as np
from keras.layers import (
    Input,
    BatchNormalization,
    Activation, Dense, Dropout, Merge,
    Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
)
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
from dataset_getter import open_dataset

ecg_segment_len = 252
n_channles = 12

def prepare_data():
    # готовим данные
    x_train, x_test, _, _ = open_dataset()

    x_train = np.swapaxes(x_train, 1, 3)
    x_test = np.swapaxes(x_test, 1, 3)
    x_train = x_train[:, 0:ecg_segment_len, :, : ]
    x_test = x_test[:, 0:ecg_segment_len, :, :]
    print("после свопа - " + str(x_test.shape))

    return x_train, x_test

def save_history(history, canterpillar_name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(canterpillar_name+".png")

def conv_block(num_kernels, kernel_size, stride):
    def f(prev):
        conv = prev
        conv = Conv2D(filters=num_kernels, kernel_size=(kernel_size,1), padding='same', strides=(stride,1))(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = MaxPooling2D(pool_size=(2,1))(conv)
        return conv

    return f

def deconv_block(num_kernels, kernel_size, upsampling):
    def f(prev):
        deconv = prev
        deconv = UpSampling2D(size=(upsampling, 1))(deconv)
        deconv = Conv2D(filters=num_kernels, kernel_size=(kernel_size,1), padding='same')(deconv)
        deconv = BatchNormalization()(deconv)
        deconv = Activation('relu')(deconv)

        return deconv

    return f

def encoder(num_kernels_arr=[25, 30], kernels_sizes_arr=(5, 3), strides_arr=[1,1]):
    def f(input):
        x = input
        for i in range(len(num_kernels_arr)):
            num_kernels = num_kernels_arr[i]
            kernel_size = kernels_sizes_arr[i]
            stride = strides_arr[i]
            x = conv_block(num_kernels, kernel_size, stride)(x)
        return x
    return f


def decoder(num_kernels_arr=[30, 25, n_channles], kernels_sizes_arr=[3, 5, 1], upsemblings_arr=[1,2,2]):
    def f(input):
        x = input
        for i in range(len(num_kernels_arr)):
            num_kernels = num_kernels_arr[i]
            kernel_size = kernels_sizes_arr[i]
            upsampling = upsemblings_arr[i]
            x = deconv_block(num_kernels, kernel_size, upsampling)(x)

        return x
    return f

def canterpillar_net():
    input_shape = (ecg_segment_len, 1, n_channles)
    input = Input(shape=input_shape)
    code = encoder()(input)
    model = Model(input, decoder()(code))
    return model

def train():
    model = canterpillar_net()
    model.summary()
    optimiser = sgd(momentum=0.9, nesterov=True)

    model.compile(optimizer=optimiser,
                 loss=mean_squared_error)

    x_train, x_test = prepare_data()
    history = model.fit(x=x_train, y=x_train,
                       validation_data=(x_test, x_test),
                       epochs=1000)
    save_history(history, "ardold_shvartsneger_1000")


train()



