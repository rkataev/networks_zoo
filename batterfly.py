# -*- coding: utf-8 -*

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
from utils import (
    draw_reconstruction_to_png, save_history
)

from keras.models import Model
from keras.losses import (
    mean_squared_error, binary_crossentropy
)
from keras.optimizers import (
    adam, sgd, adadelta
)

ecg_segment_len = 252
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf


from dataset_getter import prepare_data

def _get_trained_canterpillar():
    filepath = easygui.fileopenbox("выберите файл с обученной моделью .h5")
    trained_model = load_model(filepath)
    return trained_model

def get_classifier( hidden_len=15):
    def f(input):
        x = Dense(units=hidden_len, activation='relu')(input)
        x = Dense(units=1,activation='sigmoid')(x)
        return x
    return f


def add_head_to_model(trained_model, head_name, gate_name):
    # наращиваем классификатр от слоя по имени gate_name
    bottleneck = trained_model.get_layer(name=gate_name).output
    output = get_classifier()(bottleneck)
    model = Model(inputs=trained_model.input, outputs=output, name=head_name)
    optimiser = sgd(momentum=0.9, nesterov=True)

    model.compile(optimizer=optimiser,
                  loss=binary_crossentropy)
    return model

def create_batterfly():
    trained_ae = _get_trained_canterpillar()
    batterfly_model = add_head_to_model(trained_model=trained_ae,
                                        head_name='golova1',
                                        gate_name='bottleneck')
    return batterfly_model

def train_batterfly(name):
    model = create_batterfly()
    model.summary()
    x_train, x_test, y_train, y_test = prepare_data(seg_len=ecg_segment_len)
    history = model.fit(x=x_train, y=y_train,
                        validation_data=(x_test, y_test),
                        batch_size=20,
                        epochs=50)

    save_history(history, name)
    model.save(name + '.h5')
    return model

K.clear_session()
batterfly_name = "batterfly_left_atrum"
train_batterfly(batterfly_name)



