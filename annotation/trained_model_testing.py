import os
import json
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import pyedflib
import numpy as np
import pickle as pkl

def zero_to_nan(values):
    """Replace every 0 with 'nan' and return a copy."""
    return [float('nan') if x==0 else x for x in values]

def draw_prediction_and_reality(ecg_signal, prediction, right_answer, plot_name):
    """

    :param ecg_signal: сигнал некотего отведения
    :param prediction: предсказаные 3 бинарные маски для этого отведения
    :param right_answer: правильная маска этого отведения (тоже три штуки)
    :param plot_name: имя картинки, куда хотим отрисовать это
    :return:
    """
    figname = plot_name + "_.png"
    assert len(prediction) == len(right_answer) == 3 #3 бинарные маски

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, sharex=True)
    ax1.plot(ecg_signal)
    ax2.plot(ecg_signal)
    ax3.plot(ecg_signal)

    ax1.plot(zero_to_nan(prediction[0]),labels="predict")
    ax2.plot(zero_to_nan(prediction[1]),labels="predict")
    ax3.plot(zero_to_nan(prediction[2]),labels="predict")

    ax1.plot(zero_to_nan(right_answer[0]),labels="right")
    ax2.plot(zero_to_nan(right_answer[1]),labels="right")
    ax3.plot(zero_to_nan(right_answer[2]),labels="right")
    plt.savefig(figname)

def test_model(model, batch, name):
    """

    :param model: бученная модель
    :param batch: батч из тестовго генератора, (x, ann)
    :param name: имя для серии картинок
    :return:
    """
    predictions = model.predict_on_batch(batch['x'])
    for i in range(len(predictions)):
        predicted = predictions[i]
        true_ans = batch['x'][i]
        signal_in_channel = batch['ann'][i][0] # i-тая кардиограмма, нулевое отведение
        draw_prediction_and_reality(ecg_signal=signal_in_channel,
                                    prediction=predicted,
                                    right_answer=true_ans, plot_name=name+"_"+str(i))
    print("картинки сохранены!")




