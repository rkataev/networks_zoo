# создаем базовый автокодировщик
from sklearn.model_selection import train_test_split
from pprint import pprint
from keras.models import load_model
import numpy as np
import easygui
import pickle as pkl
from caterpillar_feeder import ecg_batches_generator
from keras.losses import (
    mean_squared_error, binary_crossentropy
)
from keras.optimizers import (
    adam, sgd, adadelta
)
from fish_family.ae import AE
from utils import (
    save_history, show_reconstruction_by_ae
)

ecg_segment_len =512


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

def get_ecg_test_sample():
    x = get_healthy_dataset()
    test_generator = ecg_batches_generator(segment_len=ecg_segment_len,
                                           batch_size=1,
                                           ecg_dataset=x)
    sample = next(test_generator)[0]
    return sample[0]

def make_fish():
    ae_maker = AE()
    fish = ae_maker.make_net()
    fish.compile(optimizer= sgd(momentum=0.9, nesterov=True),
                 loss=mean_squared_error)
    return fish


def train_fish(fish, name):
    x= get_healthy_dataset()
    train_generator, test_generator = make_generators(x)
    history = fish.fit_generator(generator=train_generator,
                                  steps_per_epoch=20,
                                  epochs=50,
                                  validation_data=test_generator,
                                  validation_steps=1)

    save_history(history, name)
    fish.save(name + '.h5')

def reconstruct_fish():
    sample = get_ecg_test_sample()
    show_reconstruction_by_ae(ecg_sample=sample, name='fish.png')

def eval_fish_with_diverse():
    #собираем активации ботлнека для здоровых и нездоровых
    pass

if __name__ == "__main__":
    name = "fish"
    #train_fish(make_fish(), name)
    reconstruct_fish()

