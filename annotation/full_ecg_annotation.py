# загружаем обученную сверточную модель
# загружаем экг из датасета
# бьем экг на перекрывающиеся сегменты
# создаем из них батч
# предсказываем сетью на том батче разметку
# отрисовавыем ее очень прозрачно на всей экг (там где очень ярко, модель очень уверена, типа)
# под штриховой линией, как обучно, отрисовываем истинную разметку
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from annotation.dice_koef import (
    dice_coef, dice_coef_loss, dice_coef_loss_quad, get_custom_objects
)
from utils import open_pickle
from annotation.multiklass2 import get_test_batch
seg_len=512

def get_model():
    folder = "./"
    model_name = 'oxxy_quad_annotator_long_nwond.h5'
    model = load_model(os.path.join(folder, model_name), custom_objects=get_custom_objects())
    print ("загрузили модель " + model_name)
    return model



def get_batch_ecg(ecg, overide):
    print (ecg.shape,"=1")
    max_t = len(ecg)-seg_len
    times=range(0, max_t, overide)
    batch = []
    print ("оценим на " + str(len(times)) + " тактах")
    for t in times:
        segment=ecg[t:t+seg_len]
        batch.append([segment])
    batch = np.array(batch)
    batch = np.swapaxes(batch, 1,2)
    print (batch.shape , " итоговый батч")
    return batch, times

def draw(full_ecg, full_ann, anns, times, name):

    #рисуем ЭКГ
    plt.figure(figsize=(13, 5))
    plt.plot(full_ecg, 'm-', label="ЭКГ")
    y_base = 12

    # штриховая линия между ответами и истиной
    plt.axhline(y=y_base, linestyle='--', color='m')

    # истинная разметка
    t_all = range(0, len(full_ecg))
    plt.fill_between(t_all, 0, 10, alpha=0.6, where=full_ann[:,0] > 0.6, label="правильн.отв.0", facecolor='red')
    plt.fill_between(t_all, 0, 10, alpha=0.6, where=full_ann[:,1] > 0.6, label="правильн.отв.1", facecolor='green')
    plt.fill_between(t_all, 0, 10, alpha=0.6, where=full_ann[:,2] > 0.6, label="правильн.отв.2", facecolor='blue')

    # ответы
    for i in range(len(times)):
        ann =anns[i]
        time = times[i]
        t_ann = range(time, time+seg_len)
        draw_ann3(t_ann, ann, y_base)

    plt.savefig(name)

def draw_ann3(t, ann3, y_base):
    print (ann3.shape)
    plt.fill_between(t, y_base, y_base + 10, alpha=0.2, where=ann3[:,0] > 0.5, facecolor='red')
    plt.fill_between(t, y_base , y_base + 10, alpha=0.2, where=ann3[:,1] > 0.5, facecolor='green')
    plt.fill_between(t, y_base, y_base + 10, alpha=0.2, where=ann3[:,2] > 0.5,facecolor='blue')

def eval_test_ecgs(model,overide, num):
    offset=200
    BATCH = get_test_batch()
    ecg = BATCH['x']
    ann = BATCH['ann']
    ecg = np.swapaxes(ecg,2,1)
    ann = np.swapaxes(ann,2,1)

    for i in range(len(BATCH['x'])):
        if num<0:
            break
        ecg_i = ecg[i]
        ann_i = ann[i]
        ecg_i = ecg_i[offset:-offset,0]
        ann_i = ann_i[offset:-offset,:]
        ecgs, times = get_batch_ecg(ecg_i, overide=overide)
        anns = model.predict_on_batch(ecgs)
        draw(ecg_i, ann_i, anns, times, name="DRAW_"+str(i)+'.png')
        num -= 1

def run():
    model = get_model()
    eval_test_ecgs(model,overide=60, num=25)



if __name__ == "__main__":
    run()
