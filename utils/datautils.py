import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
import scipy as sp
from scipy.io import arff, loadmat
import random
import matplotlib.pyplot as plt
import pdb


def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0
    return data


def load_ONT(Path='G:\XY\Source_code_in_literature\QuipuNet\data'):
    train = torch.load(Path + 'train.pt')
    val = torch.load(Path + 'val.pt')
    test = torch.load(Path + 'test.pt')
    TRAIN_DATA = train['samples'].transpose(1, 2).float()
    VAL_DATA = val['samples'].transpose(1, 2).float()
    TEST_DATA = test['samples'].transpose(1, 2).float()
    
    # TRAIN_LABEL = train['barcode_labels'].long()
    # VAL_LABEL = val['barcode_labels'].long()
    # TEST_LABEL = test['barcode_labels'].long()
    TRAIN_LABEL = train['bound_labels'].long()
    VAL_LABEL = val['bound_labels'].long()
    TEST_LABEL = test['bound_labels'].long()

    ALL_DATA = torch.cat([TRAIN_DATA, VAL_DATA, TEST_DATA])
    ALL_LABEL = torch.cat([TRAIN_LABEL, VAL_LABEL, TEST_LABEL])
    print('data loaded')

    return [np.array(ALL_DATA), np.array(ALL_LABEL)], [np.array(TRAIN_DATA), np.array(TRAIN_LABEL)], \
        [np.array(VAL_DATA), np.array(VAL_LABEL)], [np.array(TEST_DATA), np.array(TEST_LABEL)]


def load_amyloid(Path, ptms):
    data = np.array([])
    label = np.array([])
    for i, ptm in enumerate(ptms):
        mat = loadmat(Path + ptm + '_BetaAmyloid.mat')
        data = np.concatenate([data, mat['samples']], axis=0) if data.size else mat['samples']
        label = np.concatenate([label, np.ones(mat['samples'].shape[0]) * i], axis=0) if label.size else np.ones(mat['samples'].shape[0]) * i
    print(f'{ptms} data loaded')

    index = np.arange(0, len(data))
    np.random.seed(3407)
    np.random.shuffle(index)
    data = data[index].reshape(-1, 500, 1).astype(np.float32)
    label = label[index].astype(np.int64)
    train_data = data[:int(len(data) * 0.8)]
    train_label = label[:int(len(data) * 0.8)]
    val_data = data[int(len(data) * 0.8):int(len(data) * 0.9)]
    val_label = label[int(len(data) * 0.8):int(len(data) * 0.9)]
    test_data = data[int(len(data) * 0.9):]
    test_label = label[int(len(data) * 0.9):]

    return [data, label], [train_data, train_label], [val_data, val_label], [test_data, test_label]


def load_amyloid_CVsplit(Path, ptms, fold=0, ratio=1):
    data = np.array([])
    label = np.array([])
    for i, ptm in enumerate(ptms):
        mat = loadmat(Path + ptm + '_BetaAmyloid.mat')
        data = np.concatenate([data, mat['samples']], axis=0) if data.size else mat['samples']
        label = np.concatenate([label, np.ones(mat['samples'].shape[0]) * i], axis=0) if label.size else np.ones(mat['samples'].shape[0]) * i
    print(f'{ptms} data loaded')
    # print(data.shape, label.shape)

    data = data.reshape(-1, 500, 1).astype(np.float32)
    label = label.astype(np.int64)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

    splits = []
    for train_index, test_index in skf.split(data, label):
        train_data, test_data = data[train_index], data[test_index]
        train_label, test_label = label[train_index], label[test_index]
        splits.append([(train_data, train_label), (test_data, test_label)])
    
    TRAIN = splits[fold][0]
    TEST = splits[fold][1]

    if ratio < 1:
        TRAIN = Split_Proportion(ptms, ratio, TRAIN)
    ALL = [np.concatenate([TRAIN[0], TEST[0]]), np.concatenate([TRAIN[1], TEST[1]])]

    return ALL, TRAIN, None, TEST


def Split_Proportion(ptms, ratio=0.8, TRAIN=None):
    train_data = np.array([])
    train_label = np.array([])
    for i in range(len(ptms)):
        index = np.where(TRAIN[1] == i)[0]
        train_data = np.concatenate([train_data, TRAIN[0][index][:int(len(index) * ratio)]], axis=0) if train_data.size else TRAIN[0][index][:int(len(index) * ratio)]
        train_label = np.concatenate([train_label, TRAIN[1][index][:int(len(index) * ratio)]], axis=0) if train_label.size else TRAIN[1][index][:int(len(index) * ratio)]
    TRAIN = [train_data, train_label]

    return TRAIN
