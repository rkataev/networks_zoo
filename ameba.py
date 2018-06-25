# -*- coding: utf-8 -*
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras import Model
from keras.utils import np_utils
from sklearn.metrics import label_ranking_average_precision_score

import matplotlib.pyplot as plt
import numpy as np

# загружаем неповрежденные картинки и метки
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

X_train = x_train.reshape(-1, 28, 28, 1)
X_test  = x_test.reshape(-1, 28, 28, 1)

# one hot
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

print ("форма меток " +str(y_test.shape))

#повреждаем картинку при помощи апмсемплинга - даунсемплинга например

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

def draw_corrution():
    inputs,corrupted = get_corrupter4()
    corrupter = Model(inputs,corrupted)
    corrupter.summary()
    X_corrupted = corrupter.predict(X_test)
    print ("после повреждения форма картинка имеет форму " + str(X_corrupted[0].shape))


    X = X_corrupted[0].reshape([X_corrupted[0].shape[0], X_corrupted[0].shape[1]])

    plt.gray()
    plt.imshow(X)
    plt.show()


# нейроестью пытаемся восстановить ее до исходной
# при этом на вход подаем в качестве доп. входа f(код цифры) и кусок чистой энтропии


def get_ae():
    code_input = Input(shape=(10,), name='code_input')
    code_embedding = Dense(20)(code_input)

    noise_input = Input(shape=(10,), name='noise_input')

    non_corrupted_images, corrupted = get_corrupter4()

    corrupted_input_embedding = Dense(20)(corrupted)
    corrupted_input_embedding = Flatten()(corrupted_input_embedding)

    x = keras.layers.concatenate([corrupted_input_embedding, code_embedding, noise_input])

    decoded = Dense(100, activation='relu')(x)
    decoded = Dropout(rate=0.25)(decoded)
    decoded = Dense(784, activation='sigmoid', name='decoded')(decoded)
    ae_model = Model(inputs=[non_corrupted_images, code_input, noise_input], outputs=decoded)
    return ae_model

def get_noise_tensor_train():
    n = len(X_train)
    return np.random.normal(size=(n, 10,))

def train():
    ae = get_ae()
    ae.summary()
    ae.compile(optimizer='adadelta', loss='binary_crossentropy')
    noise_input_data = get_noise_tensor_train()

    MAP_OF_INPUT_TENSORS = {'code_input': y_train,
                            'raw_image_input': X_train,
                            'noise_input': noise_input_data}
    X_train_flatten = X_train.reshape(len(X_train), 784)

    callbacks = []
    boardwriter = TensorBoard(log_dir='./logs_bacteria',
                              histogram_freq=1,
                              write_graph=True,
                              write_grads=True,
                              write_images=True)
    callbacks.append(boardwriter)

    ae.fit(MAP_OF_INPUT_TENSORS,
              {'decoded': X_train_flatten},
              epochs=50, batch_size=32)


draw_corrution()
train()


# затем из тестового куска данных берем картинку и сравниваем ее восстановление с доп. входом и без.
