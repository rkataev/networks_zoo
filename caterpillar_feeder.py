import numpy as np
from dataset_getter import prepare_data

def ecg_batches_generator(segment_len, batch_size, ecg_dataset):
    """
    генеритор батчей для автокодировщика кусков ЭКГ
    :param segment_len: длина (в тактах) кусков экг, которые будем вырезать
    :param batch_size: длина батча, возвращаемого ф-цией
    :param ecg_dataset: форма тензора (кол-во пациентов, длина единичного еэг, 1, кол-во отведений), например (274, 5000, 1, 12)
    :return:
    """
    print("делаем генератор из данных размера - " + str(ecg_dataset.shape))

    starting_position = np.random.randint(0, ecg_dataset.shape[1] - segment_len)
    ending_position = starting_position + segment_len
    ecg_rand = np.random.randint(0, ecg_dataset.shape[0])

    while True:
        batch = ecg_dataset[ecg_rand:ecg_rand+1, starting_position:ending_position, :, :]
        for i in range(0, batch_size-1):
            starting_position = np.random.randint(0, ecg_dataset.shape[1] - segment_len)
            ending_position = starting_position + segment_len
            ecg_rand = np.random.randint(0, ecg_dataset.shape[0])
            batch = np.concatenate((batch, ecg_dataset[ecg_rand:ecg_rand+1, starting_position:ending_position, :, :]), 0)
        yield (batch, batch)

if __name__ == "__main__":
    segment_len = 3
    batch_size = 1
    x_train, _, _, _ = prepare_data(seg_len=None)
    train_generator = ecg_batches_generator(segment_len=segment_len,
                                            batch_size=batch_size,
                                            ecg_dataset=x_train)
    batch = next(train_generator)
    print("батч имеет форму: " + str(batch[0].shape))

