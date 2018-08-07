# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import numpy as np
import easygui
import pickle as pkl
from keras.models import load_model
import os
import wfdb
from sklearn.preprocessing import normalize
# plt.xkcd()


def draw_reconstruction_to_png(ecg_true, ecg_predicted, png_filename):
    """

    :param ecg_true: истинная экг
    :param ecg_predicted: предсказанное
    :param png_filename: имя для файла с картинкой
    :return:
    """
    ecg_true = reshape_ecg_tensor(ecg_true)
    ecg_predicted = reshape_ecg_tensor(ecg_predicted)

    assert ecg_true.shape == ecg_predicted.shape

    len_of_time = len(ecg_true[0])
    t = [i for i in range(len_of_time)]

    rows = len(ecg_true)  # кол-во каналов
    cols = 2              # true и predicted - 2 столбика
    f, axarr = plt.subplots(nrows=rows, ncols=cols, sharex=True, sharey=True)
    for i in range(rows):
        true_i_chanel = ecg_true[i]
        predicted_i_chanel = ecg_predicted[i]
        axarr[i, 0].plot(t, true_i_chanel)
        axarr[i, 1].plot(t, predicted_i_chanel)

    plt.savefig(png_filename+".png")


def reshape_ecg_tensor(ecg):
    # превратим (252, 1, 12) в (12, 252)
    print("форма тезора с экг =  " + str(ecg.shape))
    ecg = ecg[:, 0, :]
    ecg = np.transpose(ecg)
    print("форма тезора с экг (после напильника) =" + str(ecg))
    return ecg


def save_history(history, canterpillar_name):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(canterpillar_name+"_loss.png")

    if 'acc' in history.history.keys():
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model loss')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(canterpillar_name + "_acc.png")


def show_reconstruction_by_ae(ecg_sample, name):
    filepath = easygui.fileopenbox("выберите файл с обученной моделью .h5")
    trained_model = load_model(filepath)
    trained_model.summary()
    ecg_sample = np.array([ecg_sample])
    prediction = trained_model.predict(ecg_sample)

    draw_reconstruction_to_png(ecg_sample[0], prediction[0], name)


def get_addon_mask(annotations_in):
    addon = np.zeros((1, annotations_in.shape[1], 1))

    for i in range(0, annotations_in.shape[1]):
        sum = annotations_in[:, i, 0] + annotations_in[:, i, 1] + annotations_in[:, i, 2]
        if (sum == 0):
            addon[0:1, i, 0:1] = 1
        else:
            continue
    return addon


def open_pickle(path):
    infile = open(path, 'rb')
    load_pkl = pkl.load(infile)
    infile.close()
    return load_pkl


def get_ecg_data(record_path, segment_len):

    annotator = 'q1c'
    annotation = wfdb.rdann(record_path, extension=annotator)
    Lstannot = list(zip(annotation.sample, annotation.symbol, annotation.aux_note))

    FirstLstannot = min(i[0] for i in Lstannot)
    LastLstannot = max(i[0] for i in Lstannot)-1

    retry_flag = False

    for i, j in enumerate(Lstannot[:-1]):
        if((j[0] - Lstannot[i+1][0]) > 10):
            return -1

    if(retry_flag == True):
        return -1
    
    if((LastLstannot-FirstLstannot) > 5000):
        return -1

    print("NOW WORKING: " + record_path)

    print("DIST " + str(LastLstannot-FirstLstannot))

    if((LastLstannot-FirstLstannot) < segment_len):
        print("DIST " + str(LastLstannot-FirstLstannot))
        print("NOT ENOUGH POINTS!")
        return -1

    record_lead1 = wfdb.rdsamp(record_path, sampfrom=FirstLstannot,
                         sampto=LastLstannot, channels=[0])
    record_lead2 = wfdb.rdsamp(record_path, sampfrom=FirstLstannot,
                         sampto=LastLstannot, channels=[1])
    annotation = wfdb.rdann(record_path, annotator,
                            sampfrom=FirstLstannot, sampto=LastLstannot)

    VctAnnotations = list(zip(annotation.sample, annotation.symbol))

    mask_p = np.zeros(record_lead1[0].shape)
    mask_qrs = np.zeros(record_lead1[0].shape)
    mask_t = np.zeros(record_lead1[0].shape)

    #print(record_lead1[0].shape)
    #print(record_lead2[0].shape)
    for i in range(len(VctAnnotations)):
        try:
            if VctAnnotations[i][1] == "p":
                if VctAnnotations[i-1][1] == "(":
                    pstart = VctAnnotations[i-1][0]
                if VctAnnotations[i+1][1] == ")":
                    pend = VctAnnotations[i+1][0]
                if VctAnnotations[i+3][1] == "N":
                    rpos = VctAnnotations[i+3][0]
                    if VctAnnotations[i+2][1] == "(":
                        qpos = VctAnnotations[i+2][0]
                    if VctAnnotations[i+4][1] == ")":
                        spos = VctAnnotations[i+4][0]
                    # search for t (sometimes the "(" for the t  is missing  )
                    for ii in range(0, 8):
                        if VctAnnotations[i+ii][1] == "t":
                            tpos = VctAnnotations[i+ii][0]
                            if VctAnnotations[i+ii+1][1] == ")":
                                tendpos = VctAnnotations[i+ii+1][0]
                                mask_p[pstart-FirstLstannot:pend-FirstLstannot] = 1
                                mask_qrs[qpos-FirstLstannot:spos-FirstLstannot] = 1
                                mask_t[tpos-FirstLstannot:tendpos-FirstLstannot] = 1
        except IndexError:
            pass

    sum_p = np.sum(mask_p)
    sum_qrs = np.sum(mask_qrs)
    sum_t = np.sum(mask_t)

    lst = list(record_lead1)
    lst[0] = normalize(record_lead1[0], axis=0, norm='max')
    record_lead1 = tuple(lst)
    
    lst = list(record_lead2)
    lst[0] = normalize(record_lead2[0], axis=0, norm='max')
    record_lead2 = tuple(lst)

    if (sum_p == 0):
        return -1
    if (sum_qrs == 0):
        return -1
    if (sum_t == 0):
        return -1

    lead2 = np.array(record_lead2[0])
    lead2 = np.reshape(lead2, (1, np.size(record_lead2[0]), 1))

    record_tens = np.reshape(record_lead1[0], (1, np.size(record_lead1[0]), 1))
    record_tens = np.concatenate((record_tens, lead2), 2)
    mask_p = np.reshape(mask_p, (1, np.size(mask_p), 1))
    mask_qrs = np.reshape(mask_qrs, (1, np.size(mask_qrs), 1))
    mask_t = np.reshape(mask_t, (1, np.size(mask_t), 1))

    mask_tens = np.empty((1, np.size(mask_p), 0))
    mask_tens = np.concatenate((mask_tens, mask_p), axis=2)
    mask_tens = np.concatenate((mask_tens, mask_qrs), axis=2)
    mask_tens = np.concatenate((mask_tens, mask_t), axis=2)

    return record_tens, mask_tens
