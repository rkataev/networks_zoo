# -*- coding: utf-8 -*
import matplotlib.pyplot as plt
import numpy as np
import easygui
import pickle as pkl


dataset_path = "./DSET_argentina.pkl"
segment_len=512

def get_generators(train_batch=30, test_batch=50):
    infile = open(dataset_path, 'rb')
    dset = np.array(pkl.load(infile)['x'])
    infile.close()
    print (dset['x'].shape, " форма инпута")


