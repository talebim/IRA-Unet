import torch
import os
import argparse
import random
import numpy as np
import shutil
import csv
import glob
import nibabel as nib
from skimage import transform

import image_utils
from segmentation_gan import SimpleGAN

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

def read_csv(loc_path):
    with open(os.path.join(loc_path, 'ROI_spec', 'out.csv'), "r", newline='') as f:
        reader = csv.reader(f, delimiter=',')
        for l in reader:
            if reader.line_num == 1:
                pat_name = l
            elif reader.line_num == 2:
                spec = l

    return pat_name, spec


def change_size(patient, imgs, imgs_big, pat_list, spec, change):
    ind = pat_list.index(patient)
    str = spec[ind]
    str2 = str.split('[')
    str3 = str2[-1].split(']')
    cu_spec = str3[0].split(',')
    x_new_var, y_new_var, w_new_var, h_new_var = int(cu_spec[0]), int(cu_spec[1]), int(cu_spec[2]), int(cu_spec[3])
    if change == 'small':
        imgs_crop = imgs[y_new_var:y_new_var + h_new_var, x_new_var:x_new_var + w_new_var, :]
        return imgs_crop
    if change == 'big':
        imgs_big[y_new_var:y_new_var + h_new_var, x_new_var:x_new_var + w_new_var, :] = imgs
        return imgs_big


def main_out(main_path, loc_path, test_name, model_path, model, target_resolution, ref):
    model.eval()
    model.cuda()

    save_path = model_path

    pat_list, spec = read_csv(loc_path)
    img_rows, img_cols = 224, 224

    for patient in test_name:
        folder_path = os.path.join(main_path, patient)

        infos = {}
        for line in open(os.path.join(folder_path, 'Info.cfg')):
            label, value = line.split(':')
            infos[label] = value.rstrip('\n').lstrip(' ')

        systole_frame = int(infos['ES'])
        diastole_frame = int(infos['ED'])

        if os.path.exists(os.path.join(save_path + '/' + patient)):
            shutil.rmtree(os.path.join(save_path + '/' + patient))
        os.makedirs(os.path.join(save_path + '/' + patient))

        for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):
            file_base = file.split('.')[0]
            frame = int(file_base.split('frame')[-1])

            img_nii = nib.load(file)
            img_dat = img_nii.get_fdata()
            img = img_dat.copy()

            head = img_nii.header
            pixel_size = (head.structarr['pixdim'][1],
                          head.structarr['pixdim'][2])

            scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1], 1]
            slice_rescaled = transform.rescale(img,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               mode='constant')

            img_cropped = image_utils.crop_or_pad_to_size(slice_rescaled, img_rows, img_cols)

            img_cropped2 = (img_cropped - np.min(img_cropped)) * 255 / (np.max(img_cropped) - np.min(img_cropped))

            for z in range(img_cropped.shape[-1]):
                if img_cropped[:, :, z].min() > 0:
                    img_cropped[:, :, z] -= img_cropped[:, :, z].min()

                img_tmp = img_cropped[:, :, z]

                mu = img_tmp.mean()
                sigma = img_tmp.std()
                img_tmp = (img_tmp - mu) / (sigma + 1e-10)
                img_cropped2[:, :, z] = img_tmp

            imgs_test = np.array(img_cropped2, np.float32)
            imgs_test_crop = np.zeros((ref, ref, imgs_test.shape[-1]))
            imgs_test_crop = change_size(patient, imgs_test, imgs_test_crop, pat_list, spec, change='small')
            img_exp = np.expand_dims(imgs_test_crop, 0)
            img_test_fin = torch.from_numpy(img_exp).float()
            img_test_fin2 = img_test_fin.permute(3, 0, 1, 2).cuda()
            with torch.no_grad():
                pre = model(img_test_fin2)

                _, pred = torch.max(pre, dim=1)

                imgs_mask_test_final = pred.squeeze(0).cpu().numpy()

                imgs_mask_test_final_reshaped = np.transpose(imgs_mask_test_final, (1, 2, 0))

            imgs_mask_big = np.zeros(imgs_test.shape).astype('int64')
            imgs_test_big = change_size(patient, imgs_mask_test_final_reshaped, imgs_mask_big, pat_list, spec,
                                        change='big')

            mask_real_size = image_utils.crop_or_pad_to_size(imgs_test_big, slice_rescaled.shape[0],
                                                             slice_rescaled.shape[1])
            scale_vector = [target_resolution[0] / pixel_size[0], target_resolution[1] / pixel_size[1], 1]
            # mask_real_size = mask_real_size.astype('uint8')
            mask_real_scale = transform.rescale(mask_real_size,
                                                scale_vector,
                                                order=0,
                                                preserve_range=True,
                                                mode='constant',
                                                anti_aliasing=False)

            mask_real_scale = mask_real_scale.astype('uint8')
            if img.shape != mask_real_scale.shape:
                raise ValueError('the shape of image and mask is not equal')
            if frame == systole_frame:
                nimg = nib.Nifti1Image(mask_real_scale, affine=img_nii.affine, header=img_nii.header)
                nib.save(nimg,
                         os.path.join(save_path + '/' + patient + '/' + '%s_ES.nii.gz' % patient))
            elif frame == diastole_frame:
                nimg = nib.Nifti1Image(mask_real_scale, affine=img_nii.affine, header=img_nii.header)
                nib.save(nimg,
                         os.path.join(save_path + '/' + patient + '/' + '%s_ED.nii.gz' % patient))


if __name__ == "__main__":
    LOC_ROOT = os.getenv('LOC_ROOT', './loc192')

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='id', help="define id")
    parser.add_argument('--unet_arch', default='loc192_IRAUnet', help="UNet architecture")
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--loc_root', type=str, default=LOC_ROOT)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--ref_size', default=192, type=int, help='ROI size')
    args = parser.parse_args()


    _, _, test_data_name = real_arrange_callenge(args.data_root)

    loaded_model = SimpleGAN.load_from_checkpoint(
        "./ckpt/last.ckpt")

    print('test data name:', test_data_name)

    main_out(args.data_root, args.loc_root, test_data_name, args.save_path, loaded_model,
             target_resolution=(1.25, 1.25), ref=args.ref_size)
