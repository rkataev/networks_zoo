# -*- coding: utf-8 -*
import numpy as np

from keras.callbacks import (
    TensorBoard,
    ModelCheckpoint
)
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
from keras.layers import (
    Input,
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    SeparableConv2D,
    BatchNormalization,
    Add,
    Concatenate,
    AveragePooling2D


)


from dataset_getter import open_dataset

epoches = None
batch_size = None
worm_name = None
worm_optimizer = None
worm_loss = None
ecg_len = None  # < 5000

def set_worm_hyperparams():
    global epoches
    global batch_size
    global worm_name
    global worm_optimizer
    global worm_loss
    global ecg_len

    epoches = 12
    batch_size = 40
    worm_name = "_worm_spiridon1"
    worm_optimizer = 'adadelta'
    worm_loss = 'binary_crossentropy'
    ecg_len = 500

def conv_bn(x,
              filters,
              kernel_size,
              strides=1,
              padding='same',
              activation='relu',
              use_bias=False,
              name=None):
    x = Conv2D(filters,
               kernel_size,
               strides=strides,
               padding=padding,
               use_bias=use_bias,
               activation=activation,
               name=name)(x)

    x = BatchNormalization()(x)
    return x

def block1(x, block_name):
    branch_0 = conv_bn(x, 32, 1, name="end_branch_0_" + block_name)
    branch_1 = conv_bn(x, 32, 1)
    branch_1 = conv_bn(branch_1, 32, 3, name="end_branch_1_" + block_name)
    branch_2 = conv_bn(x, 32, 1)
    branch_2 = conv_bn(branch_2, 48, 3)
    branch_2 = conv_bn(branch_2, 64, 3 , name="end_branch_2_" + block_name)
    x = Concatenate(name = block_name)([branch_0, branch_1, branch_2])
    return x


def make_worm():
    n_channles = 12
    n_classes = 5
    input_shape = (ecg_len, 1, n_channles)
    output_shape = (n_classes, 1)
    # вход в сеть
    input = Input(shape=input_shape)

    # редукция 1
    x = MaxPooling2D(pool_size=(2,1))(input)
    x = conv_bn(x, 32, kernel_size=(5,1), strides=4, padding='valid', name="enf_of_eduction_1")

    # блоки типа 1
    x = block1(x, "FIRST_BLOCK")
    x = block1(x, "SECOND_BLOCK")

    # редукция 2

    # блоки типа  2

    # классификатор
    block_shape = K.int_shape(x)
    print(block_shape)
    x = AveragePooling2D(pool_size=(62, 1),
                             strides=(1, 1))(x)
    x = Flatten()(x)
    x = Dense(units=n_classes, kernel_initializer="he_normal",
                  activation="softmax")(x)
    worm = Model(input, x)
    worm.summary()
    return worm

def get_callbacks_for_worm():
    callbacks = []
    boardwriter = TensorBoard(log_dir='./logs'+worm_name,
                              histogram_freq=1,
                              write_graph=True,
                              write_grads=True,
                              write_images=True)
    callbacks.append(boardwriter)
    return callbacks


def prepare_data():
    # готовим данные
    x_train, x_test, y_train, y_test = open_dataset()

    x_train = np.swapaxes(x_train, 1, 3)
    x_test = np.swapaxes(x_test, 1, 3)
    x_train = x_train[:, 0:ecg_len, :, : ]
    x_test = x_test[:, 0:ecg_len, :, :]
    print("после свопа - " + str(x_test.shape))

    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    set_worm_hyperparams()
    x_train, x_test, y_train, y_test = prepare_data()
    worm = make_worm()
    worm.compile(optimizer=worm_optimizer,
                       loss=worm_loss)

    history = worm.fit(x=x_train, y=y_train,
                   verbose=1,
                   validation_data=(x_test, y_test),
                   epochs=epoches,
                   callbacks=get_callbacks_for_worm())
    worm.save_weights("weights_"+worm_name+".hdf5")
    losses = history.history['loss']
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.show()



