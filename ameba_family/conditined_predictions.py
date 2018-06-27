# визаулизируем, на что натренилась заданная базовая амеба
import easygui
import keras
from keras.datasets import mnist
from keras.models import model_from_json
from keras.utils import np_utils

import matplotlib.pyplot as plt
import numpy as np

import pickle as pkl

def load(filename_weights, model_json):
    json_file = open(model_json, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_amoeba = model_from_json(loaded_model_json)
    print("построили пустую амебу")
    print("загружаем веса в амебу с диска")
    loaded_amoeba.load_weights(filename_weights)
    loaded_amoeba.compile(optimizer='adadelta', loss='binary_crossentropy')
    return loaded_amoeba

def get_pic_and_ansver(pic_id):
    # загружаем неповрежденные картинки и метки
    (_, _), (x_test, y_test) = mnist.load_data()

    x = x_test[pic_id]
    y = y_test[pic_id]
    print("правильный ответ - " + str(y))

    x = x.astype('float32') / 255.
    x = x.reshape(28, 28, 1)

    return x, y

def get_conditions(ids=None):
    print ("будем делать списко таких гипотез: " + str(ids))
    if ids is None:
        ys = np.array(range(0, 10, 1))
    else:
        ys = np.array(ids)
    names = []
    for y in ys:
        names.append(str(y))
    ys = np_utils.to_categorical(ys, num_classes=10)
    return ys, names



def get_mixed_cond():
    names = []
    conds = []
    pair = np_utils.to_categorical([7, 5], num_classes=10)
    result = np.add(pair[0], pair[1])
    name = "7 + 5"
    names.append(name)
    conds.append(result)

    pair = np_utils.to_categorical([7, 5, 2], num_classes=10)
    result = np.add(np.add(pair[0], pair[1]), pair[2])
    name = "7 + 5 + 2"
    names.append(name)
    conds.append(result)


    return np.array(conds), names

def get_custom_cond():
    cond1 = [1., 1., 1., 1., 1., 1., 0., 1., 1., 1.]
    cond2 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    cond3 = [0., 0., 0., 0., 0., 1., 0., 1., 0., 0.]
    name1 = "all 1"
    name2 = " all 0"
    return np.array([cond1, cond2]), [name1, name2]

def make_cond_predictios(amoeba, raw_image, conditions):
    # клонируем картинку
    xs = []
    for i in range(len(conditions)):
        xs.append(raw_image)
    xs = np.array(xs)
    print("генерим сетью условные предсказания")
    MAP_OF_INPUT_TENSORS = {'code_input': conditions,
                            'raw_image_input': xs}
    predictions = amoeba.predict(MAP_OF_INPUT_TENSORS)
    reshaped_predictions = predictions.reshape(-1, 28, 28, 1)

    return reshaped_predictions

def show_predictions(reshaped_predictions , name, ax_names):
    print("рисуем предсказание в файл " + name + ".png")

    for i, prediction in enumerate(reshaped_predictions):
        ax = plt.subplot(1, len(reshaped_predictions), i + 1)
        ax.set_title("by " + ax_names[i])
        prediction = prediction.reshape(prediction.shape[0], prediction.shape[1])
        ax.imshow(prediction, cmap='gray')

    plt.savefig(name + "_predictions.png")
    plt.clf()

if __name__ == "__main__":
    # загружаем базовую амебу
    filename_weights = easygui.fileopenbox("выберите hdf5 файд с весами базовой модели")
    model_json = easygui.fileopenbox("выберите json файд с  модеью")
    amoeba = load(filename_weights, model_json)

    #загружаем картинку и правильный ответ
    pic, ans =  get_pic_and_ansver(0)

    #загружаем условие(я)
    #conditions, names_of_conditions = get_conditions([1,5,7])#get_conditions()
    #conditions, names_of_conditions = get_mixed_cond()
    conditions, names_of_conditions =  get_custom_cond()

    print (conditions)
    print(names_of_conditions)

    #генерим условную реконструкцию(и)  , сохраняем картинкой
    name_visualisation = "right_ans-"+str(ans)
    reshaped_predictions = make_cond_predictios(amoeba, pic, conditions=conditions)
    show_predictions(reshaped_predictions, name_visualisation, names_of_conditions)