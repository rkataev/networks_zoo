# -*- coding: utf-8 -*
# нейроестью пытаемся восстановить ее до исходной
# при этом на вход подаем в качестве доп. входа f(код цифры) и кусок чистой энтропии
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Add
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras import Model
from keras.utils import np_utils
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import label_ranking_average_precision_score
from keras.models import model_from_json

import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
import pickle as pkl

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
    amoeba = amoeba_proteus.AmoebaProteus(side_corrupter=4)
    ae = amoeba.get_ae()
    ae.summary()
    ae.compile(optimizer='adadelta', loss='binary_crossentropy')

    MAP_OF_INPUT_TENSORS = {'code_input': y_train,
                            'raw_image_input': X_train
                            }
    MAP_OF_INPUT_TENSORS_TEST = {'code_input': y_test,
                            'raw_image_input': X_test
                            }
    X_train_flatten = X_train.reshape(len(X_train), 784)
    X_test_flatten = X_test.reshape(len(X_test), 784)

    callbacks = []
    boardwriter = TensorBoard(log_dir='./logs_amoeba',
                              histogram_freq=1,
                              write_graph=True,
                              write_grads=True,
                              write_images=True)
    callbacks.append(boardwriter)
    ae.fit(MAP_OF_INPUT_TENSORS,
              {'decoded': X_train_flatten},
                validation_data=(MAP_OF_INPUT_TENSORS_TEST, {'decoded': X_test_flatten}),
              epochs=50, batch_size=30, callbacks=callbacks)
    # охранение модели
    model_json = ae.to_json()
    with open("amoeba_proteus.json", "w") as json_file:
        json_file.write(model_json)
    ae.save_weights("amoeba_proteus.hdf5", overwrite=True)

def load(filename_weights="amoeba_proteus.hdf5", model_json='amoeba_proteus.json'):
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_amoeba = model_from_json(loaded_model_json)
    print("построили пустую амебу")
    print("загружаем веса в амебу с диска")
    loaded_amoeba.load_weights(filename_weights)
    loaded_amoeba.compile(optimizer='adadelta', loss='binary_crossentropy')
    return loaded_amoeba

def get_conditioned_predictions(amoeba, true_picture):
    print("генерим сетью условные предсказания")
    ys= np.array(range(0,10,1))
    ys = np_utils.to_categorical(ys)

    xs = []
    for i in range(0,10,1):
        xs.append(true_picture)
    xs = np.array(xs)

    print ( ys.shape," ys shape")
    print ( xs.shape, " xs shape")

    #теперь генерим репрезентации

    MAP_OF_INPUT_TENSORS = {'code_input': ys,
                            'raw_image_input': xs}
    predictions = amoeba.predict(MAP_OF_INPUT_TENSORS)
    reshaped_predictions = predictions.reshape(-1, 28, 28, 1)
    return reshaped_predictions

def overfit_prediction(prediction, true_picture):
    print ("оверфитим условное предсказание к реальности")
    # генерим модельку оверфиттера
    input = Input(shape=(28, 28, 1), name='input_prediction')
    x = Flatten()(input)
    x = Dense(25, name='middle')(x)
    corrected = Dense(784, activation='sigmoid', name='decoded')(x)
    overfitter = Model(input, corrected)
    overfitter.compile(optimizer='adadelta', loss='binary_crossentropy')
    overfitter.summary()
    true_picture = np.array([true_picture])
    true_picture_flatten = true_picture.reshape(len(true_picture), 784)
    history = overfitter.fit(np.array([prediction]), true_picture_flatten, epochs=5, batch_size=1)
    losses = history.history['loss']
    return losses

def check_different_overfits(conditioned_predictions):
    series = []
    for i in range(0,10,1):
        loss_seria = overfit_prediction(conditioned_predictions[i], true_picture)
        series.append(loss_seria)
    return series

def show_results_overfits(series_of_losses):
    for i in range(len(series_of_losses)+1):
        plt.plot(series_of_losses, label=str(i))
    plt.savefig("results.png")
    plt.clf()

def show_conditioned_predictions(conditioned_predictions):
    print("рисуем условные предсказания сети ")
    for i, prediction in enumerate(conditioned_predictions):
        ax = plt.subplot(1, len(conditioned_predictions), i + 1)
        ax.set_title("by "+ str (i))
        prediction = prediction.reshape(prediction.shape[0], prediction.shape[1])
        ax.imshow(prediction, cmap='gray')

    plt.savefig("conditioned_predictions.png")
    plt.clf()


def show_true_img(true_img):
    true_img = true_img.reshape(true_img.shape[0], true_img.shape[1])
    plt.imshow(true_img, cmap='gray')
    plt.title("true image")
    plt.savefig("true_image.png")
    plt.clf()

if __name__ == "__main__":
    #train()


    true_picture = X_test[0]
    right_ansver = y_test[0]
    #show_true_img(true_picture)
    trained_amoeba = load()

    conditioned_predictions = get_conditioned_predictions(trained_amoeba, true_picture=true_picture)
    show_conditioned_predictions(conditioned_predictions)

    series_of_losses = check_different_overfits(conditioned_predictions)

    with open('series.pkl', 'wb') as f:
        pkl.dump(series_of_losses, f)

    show_results_overfits(series_of_losses)










