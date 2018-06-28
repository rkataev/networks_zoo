# -*- coding: utf-8 -*

import matplotlib.pyplot as plt
import easygui
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Add
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras import Model

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
import pickle as pkl


def show_losses_paths(losses_pkl):
    # plt.xkcd()
    infile = open(losses_pkl, 'rb')
    series_of_losses = pkl.load(infile)
    infile.close()
    x = [i for i in range(len(series_of_losses[0]))]

    for i in range(len(series_of_losses)):
        plt.plot(x, series_of_losses[i], label=str(i))

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=False)

    leg.get_frame().set_alpha(0.6)
    plt.show()

def get_model_overfitter():
    # генерим модельку оверфиттера
    input = Input(shape=(28, 28, 1), name='input_prediction')
    x = Flatten()(input)
    x = Dense(5, name='middle', activation='relu')(x)
    x = Dense(5, name='yt')(x)
    corrected = Dense(784, activation='sigmoid', name='decoded')(x)
    overfitter = Model(input, corrected)
    overfitter.compile(optimizer='adadelta', loss='binary_crossentropy')
    overfitter.summary()
    return overfitter

def overfit_prediction(prediction, true_picture, epoches):
    print ("оверфитим условное предсказание к реальности")
    overfitter = get_model_overfitter()
    true_picture = np.array([true_picture])
    true_picture_flatten = true_picture.reshape(len(true_picture), 784)
    history = overfitter.fit(np.array([prediction]), true_picture_flatten, epochs=epoches, batch_size=1)
    losses = history.history['loss']
    return losses

def check_different_overfits(conditioned_predictions, true_picture, epoches):
    series = []
    for i in range(len(conditioned_predictions)):
        loss_seria = overfit_prediction(conditioned_predictions[i], true_picture, epoches=epoches)
        series.append(loss_seria)
    return series

def get_true_pic(pic_id):
    # загружаем неповрежденные картинки и метки
    (_, _), (x_test, _) = mnist.load_data()
    x = x_test[pic_id]
    x = x.astype('float32') / 255.
    x = x.reshape(28, 28, 1)
    return x

def select_predictions_for_overfit():
    pkl_predictions_file = easygui.fileopenbox("ыберите файл с прегенерированными предсказаниями")
    infile = open(pkl_predictions_file, 'rb')
    predictions = pkl.load(infile)
    infile.close()
    return predictions

if __name__ == "__main__":
    true_picture = get_true_pic(pic_id=4)
    predictions = select_predictions_for_overfit()
    series = check_different_overfits(predictions, true_picture, epoches=136)
    with open('series_10bn.pkl', 'wb') as f:
        pkl.dump(series, f)
    show_losses_paths('series_10bn.pkl')
