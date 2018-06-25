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

from ameba_family import _corrupters as corrupt

class AmoebaProteus:
    def __init__(self, len_noise, side_corrupter):
        self.len_of_noise = len_noise
        self.side_corrupter = side_corrupter


    def get_ae(self):
        code_input = Input(shape=(10,), name='code_input')
        code_embedding = Dense(20)(code_input)

        noise_input = Input(shape=(self.len_of_noise,), name='noise_input')

        non_corrupted_images, corrupted = corrupt.get_corrupted(self.side_corrupter)

        corrupted_input_embedding = Dense(20)(corrupted)
        corrupted_input_embedding = Flatten()(corrupted_input_embedding)

        x = keras.layers.concatenate([corrupted_input_embedding, code_embedding, noise_input])

        decoded = Dense(100, activation='relu')(x)
        decoded = Dropout(rate=0.25)(decoded)
        decoded = Dense(784, activation='sigmoid', name='decoded')(decoded)
        ae_model = Model(inputs=[non_corrupted_images, code_input, noise_input], outputs=decoded)
        return ae_model

    def get_noise_tensor(self, len):
        return np.random.normal(size=(len, self.len_of_noise,))
