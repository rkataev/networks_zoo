import numpy as np
import easygui
import pickle as pkl
from sklearn.model_selection import train_test_split
from caterpillar_feeder import (
    ecg_batches_generator_for_classifier
)

only_healthy_pkl ="./fish_family/ONLY_healthy.pkl"
only_ill_pkl = "./fish_family/No_healthy90.pkl"
healthy_rythm_pkl = './fish_family/healthy.pkl'

def get_generators(train_batch=30, test_batch=50, segment_len=512):
    #вытаскиваем первый датасет
    infile = open(only_healthy_pkl, 'rb')
    x_healthy = np.array(pkl.load(infile)['x'])
    infile.close()

    #вытаскиваем второй датасет
    infile = open(healthy_rythm_pkl, 'rb')
    x_ill = np.array(pkl.load(infile)['x'])
    infile.close()

    #бъем на тест и трейн оба
    x_ill_train, x_ill_test = train_test_split(x_ill, test_size=0.33, random_state=42)
    x_healthy_train, x_healthy_test = train_test_split(x_healthy, test_size=0.33, random_state=42)

    #генерим с лейблами
    x_train, y_train = _get_xy(x_ill_train, x_healthy_train)
    x_test, y_test = _get_xy(x_ill_test, x_healthy_test)

    #делаем генераторы
    test_generetor = ecg_batches_generator_for_classifier(segment_len=segment_len, batch_size=test_batch, ecg_dataset=x_test, diagnodses=y_test)
    train_generator = ecg_batches_generator_for_classifier(segment_len=segment_len, batch_size=train_batch, ecg_dataset=x_train, diagnodses=y_train)
    return train_generator, test_generetor

def _get_xy(x_ill, x_healthy):
    X=[]
    Y=[]
    for x in x_ill:
        X.append(x)
        Y.append(1)
    for x in x_healthy:
        X.append(x)
        Y.append(0)
    X = np.array(X)
    X = np.expand_dims(X, axis=2)  # reshape (n1, n2) -> (n1, n2, 1)

    X = np.swapaxes(X, 1, 3)
    return X, np.array(Y)


train_generator, test_generetor = get_generators(train_batch=30, test_batch=50, segment_len=512)
eval_set = next(test_generetor)
