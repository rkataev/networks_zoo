# создаем базовый автокодировщик
from sklearn.model_selection import train_test_split
from pprint import pprint

import numpy as np
import easygui
import pickle as pkl
from caterpillar_feeder import ecg_batches_generator

ecg_segment_len = 500


def get_healthy_dataset():
    filename_pkl = easygui.fileopenbox("выберите файл с датасетом")

    print("загружаем 2д-датасет из файла " + filename_pkl)
    infile = open(filename_pkl, 'rb')
    new_dict = pkl.load(infile)
    infile.close()

    x = np.array(new_dict['x'])
    print(len(x), " записей ")
    x = np.expand_dims(x, axis=2)  # reshape (n1, n2) -> (n1, n2, 1)
    x = np.swapaxes(x, 1, 3)
    return x

def make_generators(x):
    batch_size = 20
    steps_per_epoch = 15
    print("батчей за эпоху будет:" + str(steps_per_epoch))
    print("в одном батче " + str(batch_size) + " кардиограмм.")
    x_train, x_test = train_test_split(x, test_size=0.33, random_state=42)
    train_generator = ecg_batches_generator(segment_len=ecg_segment_len,
                                            batch_size=batch_size,
                                            ecg_dataset=x_train)
    test_generator = ecg_batches_generator(segment_len=ecg_segment_len,
                                           batch_size=batch_size,
                                           ecg_dataset=x_test)
    return train_generator, test_generator


def make_fish():
    pass

def train_fish():
    pass

def reconstruct_fish():
    pass

def eval_fish_with_diverse():
    pass




