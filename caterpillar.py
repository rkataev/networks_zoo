# -*- coding: utf-8 -*
# гусеница
import numpy as np
import math
import easygui
from keras.models import load_model
from keras.layers import (
    Input,
    BatchNormalization,
    Activation, Dense, Dropout,
    Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
)

from keras.layers import merge

from keras.callbacks import TensorBoard
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

ecg_segment_len = 512
n_channles = 12

def prepare_data_for_canterpillar(segment_len=None):
    x_train, x_test, _, _ = prepare_data(segment_len)
    return x_train, x_test


def conv_block(num_kernels, kernel_size, stride):
    def f(prev):
        conv = prev
        conv = Conv2D(filters=num_kernels, kernel_size=(kernel_size,1), padding='same', strides=(stride,1))(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('linear')(conv)
        conv = MaxPooling2D(pool_size=(2,1))(conv)
        return conv

    return f

def deconv_block(num_kernels, kernel_size, upsampling, name):
    def f(prev):
        deconv = prev
        deconv = UpSampling2D(size=(upsampling, 1))(deconv)
        deconv = Conv2D(filters=num_kernels, kernel_size=(kernel_size,1), padding='same')(deconv)
        deconv = BatchNormalization()(deconv)
        if name is not None:
            deconv = Activation('linear', name=name)(deconv)
        else:
            deconv = Activation('linear')(deconv)
        return deconv

    return f

def encoder(num_kernels_arr=[10, 13], kernels_sizes_arr=(5, 3), strides_arr=[1,1]):
    def f(input):
        x = input
        for i in range(len(num_kernels_arr)):
            num_kernels = num_kernels_arr[i]
            kernel_size = kernels_sizes_arr[i]
            stride = strides_arr[i]
            x = conv_block(num_kernels, kernel_size, stride)(x)
        return x
    return f

def decoder(num_kernels_arr=[13, 10, n_channles], kernels_sizes_arr=[3, 5, 1], upsemblings_arr=[1,2,2], names = ['bottleneck',None, None]):
    def f(input):
        x = input
        for i in range(len(num_kernels_arr)):
            num_kernels = num_kernels_arr[i]
            kernel_size = kernels_sizes_arr[i]
            upsampling = upsemblings_arr[i]
            name = names[i]
            x = deconv_block(num_kernels, kernel_size, upsampling, name)(x)

        return x
    return f

def canterpillar_net():
    input_shape = (ecg_segment_len, 1, n_channles)
    input = Input(shape=input_shape)
    code = encoder()(input)
    model = Model(input, decoder()(code))
    return model


def train_canterpillar_with_generator(name):
    model = canterpillar_net()
    model.summary()
    optimiser = sgd(momentum=0.9, nesterov=True)

    model.compile(optimizer=optimiser,
                 loss=mean_squared_error)


    x_train, x_test = prepare_data_for_canterpillar(segment_len=None)
    batch_size = 20
    steps_per_epoch = 15
    print("батчей за эпоху будет:" + str(steps_per_epoch))
    print("в одном батче " + str(batch_size) + " кардиограмм.")
    train_generator = ecg_batches_generator(segment_len=ecg_segment_len,
                                            batch_size=batch_size,
                                            ecg_dataset=x_train)
    test_generator = ecg_batches_generator(segment_len=ecg_segment_len,
                                            batch_size=batch_size,
                                            ecg_dataset=x_test)
    
    tb_callback = TensorBoard(log_dir='./caterpillar_logs', histogram_freq=5, write_graph=True, write_grads=True)
    y_test = next(test_generator)

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=50,
                                  validation_data=y_test,
                                  validation_steps=2, callbacks = [tb_callback])

    save_history(history, name)
    model.save(name+'.h5')
    return model


def show_reconstruction_by_ae(ecg_sample, name):
    filepath = easygui.fileopenbox("выберите файл с обученной моделью .h5")
    trained_model = load_model(filepath)
    trained_model.summary()
    ecg_sample = np.array([ecg_sample])
    prediction = trained_model.predict(ecg_sample)

    draw_reconstruction_to_png(ecg_sample[0],prediction[0], name)

def get_ecg_test_sample(num_patient):
    _, x_test = prepare_data_for_canterpillar(segment_len=ecg_segment_len)
    sample = x_test[num_patient,:,:,:]
    print("форма тензора с экг: "+ str(sample.shape))
    return sample


name = "mimino_ae"
#model = train_canterpillar_with_generator(name)
ecg_sample = get_ecg_test_sample(num_patient=29)
show_reconstruction_by_ae(ecg_sample, name)





