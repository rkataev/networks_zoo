# -*- coding: utf-8 -*
# нейроестью пытаемся восстановить ее до исходной
# при этом на вход подаем в качестве доп. входа f(код цифры) и кусок чистой энтропии
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras import Model
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from sklearn.metrics import label_ranking_average_precision_score

import matplotlib.pyplot as plt
import numpy as np

from ameba_family import amoeba_proteus

# загружаем неповрежденные картинки и метки
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

X_train = x_train.reshape(-1, 28, 28, 1)
X_test  = x_test.reshape(-1, 28, 28, 1)

# one hot
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


def train():
    amoeba = amoeba_proteus.AmoebaProteus(len_noise=9, side_corrupter=4)
    ae = amoeba.get_ae()
    ae.summary()
    ae.compile(optimizer='adadelta', loss='binary_crossentropy')
    noise_input_data_train = amoeba.get_noise_tensor(len=len(X_train))
    noise_input_data_test = amoeba.get_noise_tensor(len=len(X_test))
    MAP_OF_INPUT_TENSORS = {'code_input': y_train,
                            'raw_image_input': X_train,
                            'noise_input': noise_input_data_train}
    MAP_OF_INPUT_TENSORS_TEST = {'code_input': y_test,
                            'raw_image_input': X_test,
                            'noise_input': noise_input_data_test}
    X_train_flatten = X_train.reshape(len(X_train), 784)
    X_test_flatten = X_test.reshape(len(X_test), 784)

    callbacks = []
    boardwriter = TensorBoard(log_dir='./logs_ameba',
                              histogram_freq=1,
                              write_graph=True,
                              write_grads=True,
                              write_images=True)
    callbacks.append(boardwriter)

    ae.fit(MAP_OF_INPUT_TENSORS,
              {'decoded': X_train_flatten},
                validation_data=(MAP_OF_INPUT_TENSORS_TEST, {'decoded': X_test_flatten}),
              epochs=50, batch_size=32, callbacks=callbacks)


if __name__ == "__main__":
    train()


# затем из тестового куска данных берем картинку и сравниваем ее восстановление с доп. входом и без.
