import numpy as np
from dataset_getter import prepare_data
import pprint
from utils import open_pickle

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

def ecg_batches_generator_for_classifier(segment_len, batch_size, ecg_dataset, diagnodses):
    """
    генеритор батчей для классификатора кусков ЭКГ
    :param segment_len: длина (в тактах) кусков экг, которые будем вырезать
    :param batch_size: длина батча, возвращаемого ф-цией
    :param ecg_dataset: форма тензора (кол-во пациентов, длина единичного еэг, 1, кол-во отведений), например (274, 5000, 1, 12)
    :param diagnodses: соотвествующий список диагнозов
    :return:
    """
    print("делаем генератор из данных размера - " + str(ecg_dataset.shape))

    starting_position = np.random.randint(0, ecg_dataset.shape[1] - segment_len)
    ending_position = starting_position + segment_len
    ecg_rand = np.random.randint(0, ecg_dataset.shape[0])

    while True:
        batch_x = ecg_dataset[ecg_rand:ecg_rand+1, starting_position:ending_position, :, :]
        batch_y = np.array([diagnodses[ecg_rand]])
        for i in range(0, batch_size-1):
            starting_position = np.random.randint(0, ecg_dataset.shape[1] - segment_len)
            ending_position = starting_position + segment_len
            ecg_rand = np.random.randint(0, ecg_dataset.shape[0])
            batch_x = np.concatenate((batch_x, ecg_dataset[ecg_rand:ecg_rand+1, starting_position:ending_position, :, :]), 0)
            batch_y = np.concatenate((batch_y , diagnodses[ecg_rand:ecg_rand+1]),0)
        yield (batch_x, batch_y)

def annotated_generator(segment_len, batch_size, dataset_in=None):

    """
    батч-генератор для ЭКГ с аннотациями
    :param segment_len: длина (в тактах) кусков экг, которые будем вырезать
    :param batch_size: длина батча, возвращаемого ф-цией
    :param ecg_dataset: датасет для разрезания
    """

    #здесь желательно посмотреть по интеграции в сам код сети
    #пока по умолчанию загружается датасет, выбранный ниже
    #open_pickle находится в utils для компактности
    if dataset_in is None:
        our_data = open_pickle('./datasets/DTST_argentina.pkl')
        ecg_dataset = np.array(our_data['x'])
        ecg_annotations = np.array(our_data['ann'])
    else:
        ecg_dataset = np.array(dataset_in['x'])
        ecg_annotations = np.array(dataset_in['ann'])
    
    #отступ от начала и конца
    offset = 700

    ecg_dataset = np.swapaxes(ecg_dataset, 1, 2)
    ecg_annotations = np.swapaxes(ecg_annotations, 1, 2)

    starting_position = np.random.randint(offset, ecg_dataset.shape[1] - segment_len - offset)
    ending_position = starting_position + segment_len
    ecg_rand = np.random.randint(0, ecg_dataset.shape[0])

    while True:
        batch_x = ecg_dataset[ecg_rand:ecg_rand+1, starting_position:ending_position, :]
        batch_ann = np.array([ecg_annotations[ecg_rand, starting_position:ending_position, :]])
        for i in range(0, batch_size-1):
            starting_position = np.random.randint(offset, ecg_dataset.shape[1] - segment_len - offset)
            ending_position = starting_position + segment_len
            ecg_rand = np.random.randint(0, ecg_dataset.shape[0])
            batch_x = np.concatenate((batch_x, ecg_dataset[ecg_rand:ecg_rand+1, starting_position:ending_position, :]), 0)
            batch_ann = np.concatenate((batch_ann , ecg_annotations[ecg_rand:ecg_rand+1, starting_position:ending_position, :]), 0)
        print(batch_x.shape)
        print(batch_ann.shape)
        yield (batch_x, batch_ann)

def annotated_generator_simple(genrator_annotated):
    while True:
        (batch_x, batch_ann) = next(genrator_annotated)
        batch_x=batch_x[:,:,0]
        batch_ann=batch_ann[:,:,0]
        yield (batch_x, batch_ann)

def TEST_generator_for_ae():
    segment_len = 3
    batch_size = 1
    x_train, _, _, _ = prepare_data(seg_len=None)
    train_generator = ecg_batches_generator(segment_len=segment_len,
                                            batch_size=batch_size,
                                            ecg_dataset=x_train)
    batch = next(train_generator)
    print("батч имеет форму: " + str(batch[0].shape))

def TEST_generator_for_classifier():
    segment_len = 3
    batch_size = 10
    x_train, _, y_train, _ = prepare_data(seg_len=None)
    train_generator = ecg_batches_generator_for_classifier(segment_len=segment_len,
                                            batch_size=batch_size,
                                            ecg_dataset=x_train,
                                            diagnodses=y_train)
    batch_xy = next(train_generator)
    print("батч имеет форму: \\n x.shape=" + str(batch_xy[0].shape))
    print("y.shape=" + str(batch_xy[1].shape))
    pprint.pprint(batch_xy[1])

def TEST_generator_for_annotator():
    segment_len = 10
    batch_size = 2
    dataset_in = open_pickle('./DSET_argentina.pkl')
    my_generator = annotated_generator(segment_len, batch_size, dataset_in)

    batch_xy = next(my_generator)
    print("батч имеет форму: \\n 1.shape=" + str(batch_xy[0].shape))
    print("2.shape=" + str(batch_xy[1].shape))
    pprint.pprint(batch_xy[1])

if __name__ == "__main__":
    TEST_generator_for_annotator()

