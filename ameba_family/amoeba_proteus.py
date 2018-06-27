# амеба протеус - полносвязный вариант базовой сети
# -*- coding: utf-8 -*
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input, Concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras import backend as K
from keras import Model
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from sklearn.metrics import label_ranking_average_precision_score

import matplotlib.pyplot as plt
import numpy as np

from ameba_family import _corrupters as corrupt

class AmoebaProteus:
    def __init__(self, side_corrupter):
        self.side_corrupter = side_corrupter


    def get_ae(self):
        code_input = Input(shape=(10,), name='code_input')
        code_embedding = Dense(10, name="code_embedding")(code_input)



        non_corrupted_images, corrupted = corrupt.get_corrupted(self.side_corrupter)

        corrupted_input_embedding = Dense(20, name="corrupted_input_embedding")(corrupted)
        corrupted_input_embedding = Flatten(name = "faltten_embedding")(corrupted_input_embedding)

        x = keras.layers.concatenate([corrupted_input_embedding, code_embedding])

        decoded = Dense(100, activation='relu')(x)
        decoded = Dropout(rate=0.25)(decoded)
        decoded = Dense(784, activation='sigmoid', name='decoded')(decoded)
        ae_model = Model(inputs=[non_corrupted_images, code_input], outputs=decoded)
        return ae_model




