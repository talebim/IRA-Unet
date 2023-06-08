import os
import argparse
import random
import torch
import csv
from torch.utils import data
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from load_data import data_ld
from segmentation_gan import SimpleGAN
from predict import main_out

from datetime import datetime

start_time = datetime.now()


def real_arrange_callenge(path):
    file_tot = sorted(os.listdir(path))[:100]
    random.seed(4)
    file_tot_rnd = random.sample(file_tot, len(file_tot))
    nb_data = len(file_tot_rnd)
    nb_val = round(0.15 * nb_data)
    val_data_name = file_tot_rnd[:nb_val]
    train_data_name = file_tot_rnd[nb_val:]
    test_data_name = sorted(os.listdir(path))[100:150]

    return train_data_name, val_data_name, test_data_name


def main(args):
    training_generator, validation_generator = data_ld(args.loc_root, ref=args.ref_size, batch=args.batch_size)

    loader_train = data.DataLoader(
        training_generator,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)

    loader_val = data.DataLoader(
        validation_generator,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=True)

    print('-' * 30)
    print('Creating and compiling model...')
    print('-' * 30)

    model = SimpleGAN()
    checkpoint_callback = ModelCheckpoint(dirpath=args.ckpt, monitor='g_loss_val', save_top_k=1, save_last=True,
                                          mode='min')

    logger = TensorBoardLogger("logs", name="IRAUnet")
    trainer = pl.Trainer(gpus=1, logger=logger, log_every_n_steps=1, callbacks=[checkpoint_callback],
                         max_epochs=args.num_epoch)
    trainer.fit(model, loader_train, loader_val)

    return model, trainer


if __name__ == '__main__':
    LOC_ROOT = os.getenv('LOC_ROOT', './loc192')


    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='id', help="define id")
    parser.add_argument('--unet_arch', default='loc192_IRAUnet', help="UNet architecture")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--loc_root', type=str, default=LOC_ROOT)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--ckpt', default='./ckpt', help='folder to output checkpoints')
    parser.add_argument('--num_epoch', default=2, type=int, help='epochs to train for')
    parser.add_argument('--batch_size', default=15, type=int, help='input batch size')
    parser.add_argument('--ref_size', default=192, type=int, help='ROI size')
    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # Model ID
    args.id += '-' + str(args.unet_arch)
    args.id += '-batchSize' + str(args.batch_size)
    args.id += '-epoch' + str(args.num_epoch)
    args.id += '-size' + str(args.ref_size)
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    random.seed(220)
    torch.manual_seed(220)

    train_data_name, val_data_name, test_data_name = real_arrange_callenge(args.data_root)


    loaded_model, trainer = main(args)

    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    print('test data name:', test_data_name)

    main_out(args.data_root, args.loc_root, test_data_name, args.save_path, loaded_model,
             target_resolution=(1.25, 1.25), ref=args.ref_size)
