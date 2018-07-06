import numpy as np
from dataset_getter import open_dataset

def our_train_generator(segment_len, batch_size = 10):
    x_train, _, _, _ = open_dataset()

    x_train = np.swapaxes(x_train, 1, 3)
    print("после свопа - " + str(x_train.shape))

    starting_position = np.random.randint(0, x_train.shape[1] - segment_len)
    ending_position = starting_position + segment_len
    ecg_rand = np.random.randint(0, x_train.shape[0])

    while True:
        x_train_batch = x_train[ecg_rand:ecg_rand+1, starting_position:ending_position, :, :]
        for i in range(0, batch_size-1):
            starting_position = np.random.randint(0, x_train.shape[1] - segment_len)
            ending_position = starting_position + segment_len
            ecg_rand = np.random.randint(0, x_train.shape[0])
            x_train_batch = np.concatenate((x_train_batch, x_train[ecg_rand:ecg_rand+1, starting_position:ending_position, :, :]), 0)
        yield x_train_batch, x_train_batch

def our_val_generator(segment_len, batch_size = 10):
    _, x_test, _, _ = open_dataset()

    x_test = np.swapaxes(x_test, 1, 3)
    print("после свопа - " + str(x_test.shape))

    starting_position = np.random.randint(0, x_test.shape[1] - segment_len)
    ending_position = starting_position + segment_len
    ecg_rand = np.random.randint(0, x_test.shape[0])

    while True:
        x_test_batch = x_test[ecg_rand:ecg_rand+1, starting_position:ending_position, :, :]
        for i in range(0, batch_size-1):
            starting_position = np.random.randint(0, x_test.shape[1] - segment_len)
            ending_position = starting_position + segment_len
            ecg_rand = np.random.randint(0, x_test.shape[0])
            x_test_batch = np.concatenate((x_test_batch, x_test[ecg_rand:ecg_rand+1, starting_position:ending_position, :, :]), 0)
        yield x_test_batch, x_test_batch

