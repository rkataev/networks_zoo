# -*- coding: utf-8 -*

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


def add_head_to_model(trained_model, head_name, gate_name, num_output_neurons):
    # наращиваем классификатр от слоя по имени gate_name
    bottleneck = trained_model.get_layer(name=gate_name).output
    output = get_classifier(output_neurons=num_output_neurons)(bottleneck)
    model = Model(inputs=trained_model.input, outputs=output, name=head_name)
    optimiser = sgd(momentum=0.9, nesterov=True)

    model.compile(optimizer=optimiser,
                  loss=binary_crossentropy)
    return model

def create_batterfly(num_labels):
    trained_ae = _get_trained_canterpillar()
    batterfly_model = add_head_to_model(trained_model=trained_ae,
                                        head_name='golova1',
                                        gate_name='bottleneck',
                                        num_output_neurons=num_labels )
    return batterfly_model

def train_batterfly(name):
    x_train, x_test, y_train, y_test = prepare_data(seg_len=None)  # вытаскиваем полный непорезанный датасет
    num_labels = y_train.shape[1]
    model = create_batterfly(num_labels=num_labels)
    model.summary()
    batch_size = 20

    train_generator = ecg_batches_generator_for_classifier(segment_len = ecg_segment_len,
                                                           batch_size=batch_size,
                                                           ecg_dataset=x_train,
                                                           diagnodses=y_train)
    test_generator = ecg_batches_generator_for_classifier(segment_len = ecg_segment_len,
                                                           batch_size=40,
                                                           ecg_dataset=x_test,
                                                           diagnodses=y_test)
    steps_per_epoch = 15
    print("батчей за эпоху будет:" + str(steps_per_epoch))
    print("в одном батче " + str(batch_size) + " кардиограмм.")
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=50,
                                  validation_data=test_generator,
                                  validation_steps=1)


    save_history(history, name)
    model.save(name + '.h5')
    return model

def eval_butterfly():
    # выбираем модель-классификатор
    filepath_model = easygui.fileopenbox("выберите файл с обученной моделью классиикатором.h5")
    trained_batterfly = load_model(filepath_model)
    trained_batterfly.summary()

    # вытаскиваем полный непорезанный датасет
    _, x_test, _, y_test = prepare_data(seg_len=None)

    # убеждаемся что датасет подходит к модели
    output_len = trained_batterfly.output_shape[1]
    num_labels = y_test.shape[1]
    assert output_len == num_labels

    # включаем его порезку на сегменты
    test_generator = ecg_batches_generator_for_classifier(segment_len=ecg_segment_len,
                                                          batch_size=70,
                                                          ecg_dataset=x_test,
                                                          diagnodses=y_test)
    xy = next(test_generator)

    # заставляем модель предсказать
    prediction = trained_batterfly.predict_on_batch(x=xy[0])
    print ("ответ модели:")
    print(prediction)
    print("правильный ответ:")
    print (xy[1])

if __name__ == "__main__":
    batterfly_name = "batterfly_top5_generator_eval40"
    #train_batterfly(batterfly_name)
    eval_butterfly()




