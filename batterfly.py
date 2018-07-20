# -*- coding: utf-8 -*
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
    adam, sgd, adadelta, adagrad
)
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, TensorBoard

from caterpillar_feeder import (
    ecg_batches_generator_for_classifier
)
from binary_dataset import (
    get_generators
)
ecg_segment_len = 512
import matplotlib.pyplot as plt
from keras import backend as K
import tensorflow as tf

import cosine_lr
from dataset_getter import prepare_data

def _get_trained_canterpillar():
    filepath = easygui.fileopenbox("выберите файл с обученной моделью .h5")
    trained_model = load_model(filepath)
    return trained_model

def conv_block(num_kernels, kernel_size, stride, name):
    def f(prev):
        conv = prev
        conv = Conv2D(filters=num_kernels, kernel_size=(kernel_size,1), padding='same', strides=(stride,1), name=name)(conv)
        conv = BatchNormalization(name=name+"_bn")(conv)
        conv = Activation('elu', name=name+"_ac")(conv)
        conv = MaxPooling2D(pool_size=(2,1), name=name+"_mpool")(conv)
        return conv

    return f

def get_classifier(output_neurons):
    def f(input):

        x = input
        f = conv_block(num_kernels=5, kernel_size=3, stride=1, name = "first")
        x = f(x)

        x = Flatten(name = "cl_flatten")(x)

        x = Dense(units=output_neurons, activation='sigmoid', name='classifier')(x)
        return x
    return f


def add_head_to_model(trained_model, head_name, gate_name, num_output_neurons):
    # наращиваем классификатр от слоя по имени gate_name
    bottleneck = trained_model.get_layer(name=gate_name).output
    output = get_classifier(output_neurons=num_output_neurons)(bottleneck)
    model = Model(inputs=trained_model.input, outputs=output, name=head_name)
    optimiser = adagrad(lr=0.02)#sgd(momentum=0.9, nesterov=True)

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

def train_butterfly_binary(name):
    model = create_batterfly(num_labels=1)
    model.summary()
    optimiser = sgd(momentum=0.9, nesterov=True)

    model.compile(optimizer=optimiser,
                 loss=mean_squared_error,
                metrics=['accuracy'])

    train_generator, test_generator = get_generators(train_batch=20, test_batch=50, segment_len=ecg_segment_len)
    steps_per_epoch = 30

    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=50,
                                  validation_data=test_generator,
                                  validation_steps=2)

    save_history(history, name)
    model.save(name+'.h5')

    return model

def train_batterfly(name):
    x_train, x_test, y_train, y_test = prepare_data(seg_len=None)  # вытаскиваем полный непорезанный датасет
    num_labels = y_train.shape[1]
    model = create_batterfly(num_labels=num_labels)
    model.summary()
    batch_size = 30

    train_generator = ecg_batches_generator_for_classifier(segment_len = ecg_segment_len,
                                                           batch_size=batch_size,
                                                           ecg_dataset=x_train,
                                                           diagnodses=y_train)
    test_generator = ecg_batches_generator_for_classifier(segment_len = ecg_segment_len,
                                                           batch_size=300,
                                                           ecg_dataset=x_test,
                                                           diagnodses=y_test)
    steps_per_epoch = 40
    print("батчей за эпоху будет:" + str(steps_per_epoch))
    print("в одном батче " + str(batch_size) + " кардиограмм.")
   
    #уменьшение learning rate автоматически на плато
    #learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 5, verbose = 1)

    #изменение LR по методу SGDR
    #change_lr = cosine_lr.SGDRScheduler(min_lr=0.0001, max_lr=0.1, steps_per_epoch=np.ceil(15/batch_size), lr_decay=0.8, cycle_length=1, mult_factor=1)
    
    #tb_callback = TensorBoard(log_dir='./butterfly_logs', histogram_freq=20, write_graph=True, write_grads=True)
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=50,
                                  validation_data=test_generator,
                                  validation_steps=1)


    save_history(history, name)
    model.save(name + '.h5')

    eval_generator = ecg_batches_generator_for_classifier(segment_len=ecg_segment_len,
                                                          batch_size=700,
                                                          ecg_dataset=x_test,
                                                          diagnodses=y_test)
    xy = next(eval_generator)

    # заставляем модель предсказать
    prediction = model.predict_on_batch(x=xy[0])
    print("ответ модели:")
    print(prediction)
    print("правильный ответ:")
    print(xy[1])
    return xy[1], prediction


def eval_butterfly(n_samples, trained_batterfly=None):
    if trained_batterfly is None:
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
                                                          batch_size=n_samples,
                                                          ecg_dataset=x_test,
                                                          diagnodses=y_test)
    xy = next(test_generator)

    # заставляем модель предсказать
    prediction = trained_batterfly.predict_on_batch(x=xy[0])
    print ("ответ модели:")
    print(prediction)
    print("правильный ответ:")
    print (xy[1])
    return xy[1], prediction

def visualise_result(true_labels, predicted_labels):

    # мы хотим для каждого из диагнозов отобразить 4 вещи% true_positive, true_negative, false_positive, false_negative
    rows = []
    for j in range(len(true_labels[0])):
        new_row = {"true_(right)":0,"true_(mistake)":0, "false(right)":0, "false(mistake)":0}
        true_label_column_for_desease =  true_labels[:,j]
        redicted_label_column_for_desease = predicted_labels[:,j]
        for i in range(len(true_label_column_for_desease)):
            true_label = true_label_column_for_desease[i]
            predicted_label = redicted_label_column_for_desease[i]
            if predicted_label > 0.5:
                predicted_label = 1
            else:
                predicted_label = 0

            if true_label == 1:
                if predicted_label == 1:
                    new_row["true_(right)"]+=1
                if predicted_label == 0:
                    new_row["false(mistake)"] += 1
            if true_label == 0:
                if predicted_label==1:
                    new_row["true_(mistake)"] += 1
                if predicted_label==0:
                    new_row["false(right)"] += 1
        rows.append(new_row)
    df = pd.DataFrame(data=rows)
    df.plot.bar(stacked=True)
    plt.savefig("vis.png")

def visualise_result_binary(true_labels, predicted_labels):
    rows = []
    for j in range(len(true_labels[0])):
        new_row = {"(right)": 0, "(mistake)": 0}
        true_label_column_for_desease = true_labels[:, j]
        redicted_label_column_for_desease = predicted_labels[:, j]
        for i in range(len(true_label_column_for_desease)):
            true_label = true_label_column_for_desease[i]
            predicted_label = redicted_label_column_for_desease[i]
            if predicted_label > 0.5:
                predicted_label = 1
            else:
                predicted_label = 0

            if true_label == predicted_label:
                new_row["(right)"] += 1
            else:
                new_row["(mistake)"] += 1
        rows.append(new_row)
    df = pd.DataFrame(data=rows)
    df.plot.bar(stacked=True)
    plt.savefig("vis_binary.png")

if __name__ == "__main__":
    name = "batterfly_top15_anna"
    #true_labels, predicted_labels =train_batterfly(batterfly_name)
    #true_labels, predicted_labels = eval_butterfly(n_samples=900)
    #visualise_result(true_labels, predicted_labels)
    #visualise_result_binary(true_labels, predicted_labels)
    train_butterfly_binary(name)




