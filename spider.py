# -*- coding: utf-8 -*
# модель для учета нераномерности распределения меток
import pandas as pd

import numpy as np
import math
import easygui
from keras.models import load_model
from keras.layers import (
    Input,
    BatchNormalization,
    Activation, Dense, Dropout,Flatten,
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

from caterpillar_feeder import (
    ecg_batches_generator_for_classifier
)

ecg_segment_len = 400
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf


from dataset_getter import prepare_data

def _get_trained_canterpillar():
    filepath = easygui.fileopenbox("выберите файл с обученной моделью .h5")
    trained_model = load_model(filepath)
    return trained_model

def get_classifier(output_neurons, hidden_lens=[10, 5, 3]):
    def f(input):
        x = Flatten()(input)
        i =0
        for hidden_len in hidden_lens:
            x = Dense(units=hidden_len, activation='relu')(x)

            x = BatchNormalization(name = "bn_classifier_" + str(i) )(x)
            i+=1
        x = Dense(units=output_neurons, activation='sigmoid', name='classifier')(x)
        return x
    return f
