# -*- coding: utf-8 -*
#from sklearn.model_selection import train_test_split, learning_curve, KFold
from sklearn.model_selection import train_test_split
from pprint import pprint

import numpy as np
import easygui
import pickle as pkl
def open_dataset():
    filename_pkl = easygui.fileopenbox("выберите файл с датасетом")

    print("загружаем 2д-датасет из файла " + filename_pkl)
    infile = open(filename_pkl, 'rb')
    new_dict = pkl.load(infile)
    infile.close()

    assert len(new_dict['x']) == len(new_dict['y'])
    x, y = np.array(new_dict['x']), np.array(new_dict['y'])
    print(len(new_dict['x']) ," записей ")

    # разбиваем на трейн/тест
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    # решейпим их, чтоб модель могла их сожрать
    x_train = np.expand_dims(x_train, axis=2)  # reshape (n1, n2) -> (n1, n2, 1)
    x_test = np.expand_dims(x_test, axis=2)  # reshape (n1, n2) -> (n1, n2, 1)


    print("y_train  имеет форму " + str(y_train.shape))
    print("x_train  имеет форму " + str(x_train.shape))
    print("x_test   имеет форму " + str(x_test.shape))
    print("y_test   имеет форму " + str(y_test.shape))


    return  x_train, x_test, y_train, y_test



if __name__ == "__main__":
    #x_train, x_test, y_train, y_test = open_dataset()
    import matplotlib.pyplot as plt

    #plt.xkcd()
    infile = open('series.pkl', 'rb')
    series_of_losses = pkl.load(infile)
    infile.close()
    x = [i for i in range(len(series_of_losses[0]))]

    for i in range(len(series_of_losses)):
        plt.plot(x, series_of_losses[i], label=str(i))

    leg = plt.legend(loc='best', ncol=2, mode="expand", shadow=True, fancybox=False)

    leg.get_frame().set_alpha(0.6)
    plt.show()