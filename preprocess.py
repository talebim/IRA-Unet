import os
import shutil
import glob
import numpy as np
import random
import csv
import cv2
import nibabel as nib
import logging
from skimage import transform
import argparse

import image_utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

img_rows, img_cols = 224, 224
target_resolution = (1.25, 1.25)


def creat_folds(dirpath):
    if os.path.exists(os.path.join(dirpath)):
        shutil.rmtree(os.path.join(dirpath))
    os.makedirs(os.path.join(dirpath))

    os.makedirs(dirpath + '/' + 'val')
    os.makedirs(dirpath + '/' + 'mask_val')
    os.makedirs(dirpath + '/' + 'train')
    os.makedirs(dirpath + '/' + 'mask_train')
    os.makedirs(dirpath + '/' + 'ROI_spec')

    return dirpath


def find_special_image(path, im_case):
    folder_path = os.path.join(path, im_case, '%s_4d.nii.gz' % im_case)

    nimg = nib.load(folder_path)

    slc_num = round(nimg.shape[2] / 2)
    print('middle slice is', slc_num)

    img_ph_tot = []

    for phase in range(nimg.shape[3]):
        phase_data = nimg.get_data()
        phase_img = np.squeeze(phase_data[:, :, :, phase])

        head = nimg.get_header()
        pixel_size = (head.structarr['pixdim'][1],
                      head.structarr['pixdim'][2])

        scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]

        slice_img = np.squeeze(phase_img[:, :, slc_num])

        slice_rescaled = transform.rescale(slice_img,
                                           scale_vector,
                                           order=1,
                                           preserve_range=True,
                                           mode='constant')

        slice_cropped = transform.resize(slice_rescaled, (img_rows, img_cols), preserve_range=True)

        pixs = slice_cropped * 255 / (np.max(slice_cropped) - np.min(slice_cropped))
        pixs = pixs.astype('uint8')
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl1 = clahe.apply(pixs)

        img_ph_tot.append(cl1)

    return img_ph_tot


def remove_particles(image):
    new_image = image.copy()
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            neighbor = np.zeros((5, 5), np.uint8)
            if image[i][j] == 255:
                for ii in range(5):
                    for jj in range(5):
                        if i - 2 + ii < image.shape[0] and j - 2 + jj < image.shape[1]:
                            neighbor[ii][jj] = image[i - 2 + ii][j - 2 + jj] / 255
                if np.sum(neighbor) < 20:
                    new_image[i][j] = 0

    return new_image


def compute_variance(img_path, img_cases):
    empty = 0
    x = 0
    y = 0
    img_pixs_tot = find_special_image(img_path, img_cases)
    var_img = np.var(img_pixs_tot, axis=0)

    mean_img = np.mean(img_pixs_tot, axis=0)
    var_img_nrm = var_img * 255 / (np.max(var_img) - np.min(var_img))
    var_img_nrm = var_img_nrm.astype('uint8')

    equ = cv2.equalizeHist(var_img_nrm)

    ret, thresh = cv2.threshold(equ, 215, 255, cv2.THRESH_BINARY)
    rm_part = remove_particles(thresh)
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(rm_part, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_org = sorted(contours, key=len, reverse=True)

    image = dilation.copy()

    for cn in range(len(cnts_org)):
        mean_tot = np.mean(cnts_org[cn], axis=0, dtype=np.float32)
        mask = np.ones(image.shape[:2], dtype="uint8") * 255
        if mean_tot[0][0] > img_rows - round(img_rows / 4) or mean_tot[0][1] > img_cols - round(img_cols / 4) or \
                mean_tot[0][0] < round(img_rows / 4) or mean_tot[0][1] < round(img_cols / 4):
            cv2.drawContours(mask, cnts_org, cn, 0, -1)
            image = cv2.bitwise_and(image, image, mask=mask)

    contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_org = sorted(contours, key=len, reverse=True)

    for cn in range(len(cnts_org) - 1):
        mean_1 = np.mean(cnts_org[0], axis=0, dtype=np.float32)
        mean_2 = np.mean(cnts_org[cn + 1], axis=0, dtype=np.float32)
        mask = np.ones(image.shape[:2], dtype="uint8") * 255
        if (abs(mean_1[0][1] - mean_2[0][1]) > 60 or abs(mean_1[0][0] - mean_2[0][0]) > 60) and len(
                cnts_org[cn + 1]) <= 40:
            cv2.drawContours(mask, cnts_org, cn + 1, 0, -1)
            image = cv2.bitwise_and(image, image, mask=mask)

    x, y, width, height = cv2.boundingRect(image)
    color_img2 = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(color_img2, (x, y), (x + width, y + height), (0, 255, 0), 2)

    return x, y, width, height, empty, color_img2


def correct_coordin(x, y, w, h, ref, image):
    if w < ref and x - round((ref - w) / 2) > 0:
        x_new = x - round((ref - w) / 2)
    elif w < ref and x - round((ref - w) / 2) <= 0:
        x_new = 0
    elif w > ref and x - round((w - ref) / 2) > 0:
        y_new = x - round((w - ref) / 2)
    elif w > ref and x - round((w - ref) / 2) <= 0:
        y_new = x + round((w - ref) / 2)
    if x_new + ref > image.shape[1]:
        x_new2 = image.shape[1] - ref
    else:
        x_new2 = x_new
    w_new = ref
    if h <= ref and y - round((ref - h) / 2) > 0:
        y_new = y - round((ref - h) / 2)
    elif h <= ref and y - round((ref - h) / 2) <= 0:
        y_new = 0
    elif h > ref and y - round((h - ref) / 2) > 0:
        y_new = y - round((h - ref) / 2)
    elif h > ref and y - round((h - ref) / 2) <= 0:
        y_new = y + round((h - ref) / 2)
    if y_new + ref > image.shape[0]:
        y_new2 = image.shape[0] - ref
    else:
        y_new2 = y_new
    h_new = ref
    cv2.rectangle(image, (x_new2, y_new2), (x_new2 + w_new, y_new2 + h_new), (0, 0, 255), 2)

    return x_new2, y_new2, w_new, h_new


def compute_ROI_slc_mid(img_path, case, ref):
    print('\ncomputing variance for ', case)
    x_var, y_var, w_var, h_var, empty, flag = compute_variance(img_path, case)
    x_new_var, y_new_var, w_new_var, h_new_var = correct_coordin(x_var, y_var, w_var, h_var, ref, flag)
    var_spec = [x_new_var, y_new_var, w_new_var, h_new_var]
    return var_spec


def compute_ROI_ACDC(img_path, dirpath, case, ROI_spec, ref, mode, data_type):
    folder_path = os.path.join(img_path, case)

    infos = {}
    for line in open(os.path.join(folder_path, 'Info.cfg')):
        label, value = line.split(':')
        infos[label] = value.rstrip('\n').lstrip(' ')

    mid_slc_spec = compute_ROI_slc_mid(img_path, case, ref)

    ROI_spec.append(mid_slc_spec)

    if data_type != 'test':
        for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):
            file_base = file.split('.')[0]
            file_mask = file_base + '_gt.nii.gz'
            img_name = file_base.split('/')[-1]

            print('image is for the case:', img_name)
            img_nii = nib.load(file)
            mask_nii = nib.load(file_mask)

            img_dat = img_nii.get_data()
            mask_dat = mask_nii.get_data()

            img = img_dat.copy()
            mask = mask_dat.copy()

            if mode == 5:
                mask[mask == 1] = 1
                mask[mask == 2] = 0
                mask[mask == 3] = 0
            elif mode == 1:
                mask[mask == 1] = 0
                mask[mask == 2] = 1
                mask[mask == 3] = 0
            elif mode == 0:
                mask[mask == 1] = 0
                mask[mask == 2] = 0
                mask[mask == 3] = 1

            head = img_nii.get_header()
            pixel_size = (head.structarr['pixdim'][1],
                          head.structarr['pixdim'][2])

            scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1], 1]

            slice_rescaled = transform.rescale(img,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               mode='constant')

            mask_rescaled = transform.rescale(mask,
                                              scale_vector,
                                              order=0,
                                              preserve_range=True,
                                              anti_aliasing=False,
                                              mode='constant')

            slice_cropped = image_utils.crop_or_pad_to_size(slice_rescaled, img_rows, img_cols)
            mask_cropped = image_utils.crop_or_pad_to_size(mask_rescaled, img_rows, img_cols)

            slice_cropped2 = np.zeros(slice_cropped.shape)

            for z in range(slice_cropped.shape[-1]):
                if slice_cropped[:, :, z].min() > 0:
                    slice_cropped[:, :, z] -= slice_cropped[:, :, z].min()

                img_tmp = slice_cropped[:, :, z]

                mu = img_tmp.mean()
                sigma = img_tmp.std()
                img_tmp = (img_tmp - mu) / (sigma + 1e-10)
                slice_cropped2[:, :, z] = img_tmp

            mask_cropped2 = np.round(mask_cropped).astype('uint8')

            [x_new_var, y_new_var, w_new_var, h_new_var] = mid_slc_spec

            mask_crop = mask_cropped2[y_new_var:y_new_var + h_new_var, x_new_var:x_new_var + w_new_var]

            img_crop = slice_cropped2[y_new_var:y_new_var + h_new_var, x_new_var:x_new_var + w_new_var]
            if data_type != 'test':
                nimg = nib.Nifti1Image(mask_crop, affine=mask_nii.affine, header=mask_nii.header)
                nib.save(nimg,
                         os.path.join(dirpath, 'mask_' + '%s', '%s_mask.nii.gz') % (data_type, img_name))
                nimg = nib.Nifti1Image(img_crop, affine=img_nii.affine, header=img_nii.header)
                nib.save(nimg, os.path.join(dirpath, '%s', '%s.nii.gz') % (data_type, img_name))
    return ROI_spec


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


def compute_ROI(img_path, dir_path, file_tot, ROI_spec, ref, mode, data_type):
    for case in file_tot:
        compute_ROI_ACDC(img_path, dir_path, case, ROI_spec, ref, mode, data_type)

    return ROI_spec


if __name__ == '__main__':
    LOC_ROOT = os.getenv('LOC_ROOT', './loc192')

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--loc_root', type=str, default=LOC_ROOT)
    parser.add_argument('--ref_size', default=192, type=int, help='ROI size')
    args = parser.parse_args()

    print('-' * 30)
    print('Preprocessed begins....')
    print('-' * 30)

    train_data_name, val_data_name, test_data_name = real_arrange_callenge(args.data_root)
    data_tot = list(train_data_name)

    for i in range(len(val_data_name)):
        data_tot.append(val_data_name[i])
    for i in range(len(test_data_name)):
        data_tot.append(test_data_name[i])

    loc_path = creat_folds(args.loc_root)

    ROI_spec = []
    ROI_spec = compute_ROI(args.data_root, loc_path, train_data_name, ROI_spec, ref=args.ref_size,
                           mode=type, data_type='train')
    ROI_spec = compute_ROI(args.data_root, loc_path, val_data_name, ROI_spec, ref=args.ref_size,
                           mode=type, data_type='val')
    ROI_spec = compute_ROI(args.data_root, loc_path, test_data_name, ROI_spec, ref=args.ref_size,
                           mode=type, data_type='test')
    with open(os.path.join(loc_path, 'ROI_spec', 'out.csv'), 'w') as fh:
        writer = csv.writer(fh, delimiter=',')
        writer.writerow(data_tot)
        writer.writerow(ROI_spec)

    print('-' * 30)
    print('Preprocessed finished.')
    print('-' * 30)