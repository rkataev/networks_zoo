# -*- coding: utf-8 -*
# гусеница
import numpy as np
import easygui
from keras.models import load_model
from keras.layers import (
    Input,
    BatchNormalization,
    Activation, Dense, Dropout,
    Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D
)

from keras.layers import merge
from keras.preprocessing.sequence import TimeseriesGenerator

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

from dataset_getter import prepare_data
from utils import (
    draw_reconstruction_to_png
)

ecg_segment_len = 252
n_channles = 12

def prepare_data_for_canterpillar():
    x_train, x_test, _, _ = prepare_data(ecg_segment_len)
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

def train_canterpillar(name):
    model = canterpillar_net()
    model.summary()
    optimiser = sgd(momentum=0.9, nesterov=True)

    model.compile(optimizer=optimiser,
                 loss=mean_squared_error)


    x_train, x_test = prepare_data_for_canterpillar()
    history = model.fit(x=x_train, y=x_train,
                       validation_data=(x_test, x_test),
                        batch_size=20,
                       epochs=100)

    save_history(history, name)
    model.save(name+'.h5')
    return model

def show_reconstruction_by_ae(ecg_sample, name):
    filepath = easygui.fileopenbox("выберите файл с обученной моделью .h5")
    trained_model = load_model(filepath)

    ecg_sample = np.array([ecg_sample])
    prediction = trained_model.predict(ecg_sample)

    draw_reconstruction_to_png(ecg_sample[0],prediction[0], name)

def get_ecg_test_sample(num_patient):
    _, x_test = prepare_data_for_canterpillar()
    sample = x_test[num_patient,:,:,:]
    print("форма тензора с экг: "+ str(sample.shape))
    return sample


name = "merkel"
#model = train_canterpillar(name)
ecg_sample = get_ecg_test_sample(num_patient=0)
show_reconstruction_by_ae(ecg_sample, name)





