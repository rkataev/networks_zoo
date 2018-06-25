#повреждаем картинку при помощи апмсемплинга - даунсемплинга например
# -*- coding: utf-8 -*
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
def get_corrupter7():
    inputs = Input(shape=(28, 28, 1), name='raw_image_input')
    corrupted = MaxPooling2D(padding='same',strides=4)(inputs)
    corrupted = AveragePooling2D(padding='same',strides=1)(corrupted)
    return inputs,corrupted

def get_corrupter6():
    inputs = Input(shape=(28, 28, 1), name='raw_image_input')
    corrupted = MaxPooling2D(padding='same',strides=5)(inputs)
    #corrupted = AveragePooling2D(padding='same',strides=1)(corrupted)
    return inputs,corrupted

def get_corrupter5():
    inputs = Input(shape=(28, 28, 1), name='raw_image_input')
    corrupted = AveragePooling2D(padding='same',strides=3, pool_size=(3, 3) )(inputs)
    corrupted = AveragePooling2D(padding='same',strides=2, name='corrupted_image')(corrupted)
    return inputs,corrupted

def get_corrupter4():
    inputs = Input(shape=(28, 28, 1), name='raw_image_input')
    corrupted = MaxPooling2D(padding='same',strides=3)(inputs)
    corrupted = AveragePooling2D(padding='same',strides=3)(corrupted)
    return inputs,corrupted

def get_corrupted(side_corrupter):
    if side_corrupter == 7:
        return get_corrupter7()
    if side_corrupter == 6:
        return get_corrupter6()
    if side_corrupter == 5:
        return get_corrupter5()
    if side_corrupter == 4:
        return get_corrupter4()

def draw_corrution(X_test): # 28, 28, 1
    inputs,corrupted = get_corrupter4()
    corrupter = Model(inputs,corrupted)
    corrupter.summary()
    X_corrupted = corrupter.predict(X_test)
    print ("после повреждения форма картинка имеет форму " + str(X_corrupted[0].shape))


    X = X_corrupted[0].reshape([X_corrupted[0].shape[0], X_corrupted[0].shape[1]])

    plt.gray()
    plt.imshow(X)
    plt.show()

if __name__ == "__main__":
    # загружаем неповрежденные картинки и метки
    (_, _), (x_test, _) = mnist.load_data()
    x_test = x_test.astype('float32') / 255.
    X_test  = x_test.reshape(-1, 28, 28, 1)
    draw_corrution(X_test)
