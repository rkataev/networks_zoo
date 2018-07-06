# -*- coding: utf-8 -*
from __future__ import print_function

import tensorflow as tf
print("версия тезорфлоу: " + str(tf.__version__))
import keras
print("версия керас: " + str(keras.__version__))
from keras import backend as K
if K.image_data_format() == 'channels_first':
    print("количество каналов - первое в кортеже")
else:
    print("количество каналов - последнее в кортеже")

print ("image dim ordering: " + K.image_dim_ordering() )
import numpy as np
from time import time
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import TensorBoard, ModelCheckpoint
from dataset_getter import open_dataset_as_train_test

batch_size = 50
num_classes = 5
epochs = 10

# солько тактов времени в кардиограме
img_len = 5000

# олучаем данные, сразу трейн и тест
x_train, x_test, y_train, y_test = open_dataset_as_train_test()


# простейшая сверточная моделька
model = Sequential()
model.add(Conv1D(20,
                 kernel_size=5,
                 activation='relu',
                 input_shape=(img_len, 1)))

model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()

# коллбеки
callbacks = []
boardwriter = TensorBoard(log_dir='./logs_bacteria',
                          histogram_freq=1,
                          write_graph=True,
                          write_grads=True,
                          write_images=True)
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks.append(boardwriter)
callbacks.append(checkpoint)
# само обучение
model.fit(x=x_train, y=y_train, verbose=1,validation_data=(x_test, y_test), epochs=epochs, callbacks=callbacks)