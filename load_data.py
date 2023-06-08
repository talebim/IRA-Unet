import os
import random
import shutil
import numpy as np

from dataGen import DataGenerator, load3d


def data_ld(loc_path, ref, batch):
    print('-' * 30)
    print('Loading and preprocessing train data...')
    print('-' * 30)
    train_dirpath = os.path.join(loc_path, 'train')
    mask_dirpath = os.path.join(loc_path, 'mask_train')
    val_dirpath = os.path.join(loc_path, 'val')
    msk_val_dirpath = os.path.join(loc_path, 'mask_val')

    nb_train = os.listdir(train_dirpath)
    random.shuffle(nb_train)
    nb_val = os.listdir(val_dirpath)

    # Parameters
    params = {'dim': (ref, ref),
              'batch_size': batch,
              'n_classes': 4,
              'n_channels': 1,
              'shuffle': True}


    acdata_train = load3d(train_dirpath, mask_dirpath, nb_train, **params)
    training_generator = DataGenerator(acdata_train, split='train')
    acdata_val = load3d(val_dirpath, msk_val_dirpath, nb_val, **params)
    validation_generator = DataGenerator(acdata_val, split='val')

    return training_generator, validation_generator
