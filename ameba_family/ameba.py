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
    filepath = "amoeba_proteus_best.hdf5"
    best_weights_savier = ModelCheckpoint(save_best_only=True, save_weights_only=True, filepath=filepath, monitor='val_acc', verbose=1,  mode='max')
    callbacks.append(best_weights_savier)

    ae.fit(MAP_OF_INPUT_TENSORS,
              {'decoded': X_train_flatten},
                validation_data=(MAP_OF_INPUT_TENSORS_TEST, {'decoded': X_test_flatten}),
              epochs=1, batch_size=40, callbacks=callbacks)
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
    ys= np.array(range(0,9,1))
    ys = np_utils.to_categorical(ys)

    xs = []
    for i in range(9):
        xs.append(true_picture)
    xs = np.array(xs)

    print ( ys.shape," ys shape")
    print ( xs.shape, " xs shape")

    #теперь генерим репрезентации

    MAP_OF_INPUT_TENSORS = {'code_input': y_train,
                            'raw_image_input': X_train}
    predictions = amoeba.predict(MAP_OF_INPUT_TENSORS)
    reshaped_predictions = predictions.reshape(-1, 28, 28, 1)
    return reshaped_predictions

def overfit_prediction(prediction, true_picture, right_ansver):
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
    history = overfitter.fit(np.array([prediction]), true_picture_flatten, epochs=1, batch_size=1)
    losses = history.history['loss']
    print (losses)


if __name__ == "__main__":
    #train()
    trained_amoeba = load()
    true_picture = X_test[0]
    right_ansver = y_test[0]
    conditioned_predictions = get_conditioned_predictions(trained_amoeba, true_picture=true_picture)
    overfit_prediction( conditioned_predictions[0], true_picture, right_ansver)





