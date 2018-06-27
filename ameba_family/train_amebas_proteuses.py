# треним зоопарк амеб-протеус
# -*- coding: utf-8 -*

import keras
from keras.datasets import mnist
from keras.utils import np_utils
import pickle as pkl

from ameba_family import amoeba_proteus

# загружаем неповрежденные картинки и метки
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

X_train = x_train.reshape(-1, 28, 28, 1)
X_test  = x_test.reshape(-1, 28, 28, 1)

# one hot
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


def train(name_of_experiment, epoches):
    amoeba = amoeba_proteus.AmoebaProteus(side_corrupter=4)
    ae = amoeba.get_ae()
    ae.summary()
    ae.compile(optimizer='adadelta', loss='binary_crossentropy')

    MAP_OF_INPUT_TENSORS = {'code_input': y_train,
                            'raw_image_input': X_train
                            }
    MAP_OF_INPUT_TENSORS_TEST = {'code_input': y_test,
                            'raw_image_input': X_test
                            }
    X_train_flatten = X_train.reshape(len(X_train), 784)
    X_test_flatten = X_test.reshape(len(X_test), 784)

    history = ae.fit(MAP_OF_INPUT_TENSORS,
              {'decoded': X_train_flatten},
                validation_data=(MAP_OF_INPUT_TENSORS_TEST, {'decoded': X_test_flatten}),
                epochs=epoches,
                batch_size=30)
    # охранение модели
    model_json = ae.to_json()
    name_model = "amoeba_proteus_" + name_of_experiment
    with open(name_model+ ".json", "w") as json_file:
        json_file.write(model_json)
    ae.save_weights(name_model+".hdf5", overwrite=True)
    with open(name + 'series.pkl', 'wb') as f:
        pkl.dump(history.history["loss"], f)


if __name__ == "__main__":
    names = ["little_10"]
    epoches = [5]
    assert len(names) == len(epoches)
    true_picture = X_test[0]
    right_ansver = y_test[0]

    for i in range(len(epoches)):
        name = names[i]
        epoch = epoches[i]
        train(name_of_experiment=name, epoches=epoch)


