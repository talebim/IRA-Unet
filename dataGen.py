import numpy as np
import cv2
import torch
import nibabel as nib
import os
from torch.utils import data

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def random_elastic_deformation(image, mask, alpha=500, sigma=20, mode='nearest',
                               random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
..  [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
    Convolutional Neural Networks applied to Visual Document Analysis", in
    Proc. of the International Conference on Document Analysis and
    Recognition, 2003.
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

    height, width = image.shape

    dx = gaussian_filter(2 * random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(2 * random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    indices = ((x + dx),
               (y + dy))

    img_new = map_coordinates(image, indices, order=1, mode=mode)
    mask_new = map_coordinates(mask, indices, order=0, mode=mode)

    return img_new, mask_new


def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0625, 0.0625),
                           scale_limit=(-0.1, 0.1),
                           rotate_limit=(-45, 45), aspect_limit=(-0.5, 0.5),
                           borderMode=cv2.BORDER_CONSTANT):
    height = image.shape[0]
    width = image.shape[1]

    angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
    scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
    aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
    sx = scale * aspect / (aspect ** 0.5)
    sy = scale / (aspect ** 0.5)
    dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
    dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

    cc = np.math.cos(angle / 180 * np.math.pi) * sx
    ss = np.math.sin(angle / 180 * np.math.pi) * sy
    rotate_matrix = np.array([[cc, -ss], [ss, cc]])

    box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
    box1 = box0 - np.array([width / 2, height / 2])
    box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

    box0 = box0.astype(np.float32)
    box1 = box1.astype(np.float32)
    mat = cv2.getPerspectiveTransform(box0, box1)
    image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                borderValue=(
                                    0, 0,
                                    0,))
    mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                               borderValue=(
                                   0, 0,
                                   0,))

    return image, mask


def randomHorizontalFlip(image, mask):
    image = cv2.flip(image, 1)
    mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticalFlip(image, mask):
    image = cv2.flip(image, 0)
    mask = cv2.flip(mask, 0)

    return image, mask


class load3d(torch.utils.data.Dataset):
    def __init__(self, train_dirpath, mask_dirpath, list_IDs, batch_size, dim, n_channels,
                 n_classes, shuffle):
        'Initialization'
        self.train_dirpath = train_dirpath
        self.mask_dirpath = mask_dirpath
        self.dim = dim
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, item):

        file_base = self.list_IDs[item].split('.')[0]

        img_nii = nib.load(os.path.join(self.train_dirpath, self.list_IDs[item]))
        img_dat = img_nii.get_fdata()
        img = img_dat.copy()

        mask_nii = nib.load(os.path.join(self.mask_dirpath, file_base + '_mask' + '.nii.gz'))
        mask_dat = mask_nii.get_fdata()
        mask = mask_dat.copy()

        data_dict = {
            "name": file_base,
            "image": img,
            "mask": mask,
        }
        return data_dict


class DataGenerator():
    def __init__(self, dataset, split):
        self.data = []
        self.split = split

        for i in range(dataset.__len__()):
            d = dataset[i]
            for x in range(d["image"].shape[-1]):
                entry = {}
                entry["image"] = d["image"].transpose(2, 0, 1)[x]
                entry["mask"] = d["mask"].transpose(2, 0, 1)[x]
                entry["name"] = d["name"] + "_z" + str(x)
                self.data.append(entry)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img = self.data[i]["image"]
        mask = self.data[i]["mask"]

        if self.split == 'train':
            rnd = np.random.randint(0, 5, 1)
            if rnd == 0:
                img, mask = randomShiftScaleRotate(img, mask)
            elif rnd == 1:
                img, mask = randomHorizontalFlip(img, mask)
            elif rnd == 2:
                img, mask = randomVerticalFlip(img, mask)
            elif rnd == 3:
                img, mask = random_elastic_deformation(img, mask)
            else:
                img, mask = img, mask
        else:
            img, mask = img, mask

        mask_round = np.round(mask).astype('uint8')

        mask_lv = np.zeros(mask.shape).astype('uint8')
        mask_rv = np.zeros(mask.shape).astype('uint8')
        mask_myo = np.zeros(mask.shape).astype('uint8')
        mask_bg = np.zeros(mask.shape)
        mask_lv[mask_round == 3] = 1
        mask_rv[mask_round == 1] = 1
        mask_myo[mask_round == 2] = 1
        mask_bg[mask_round == 0] = 1

        mask_exp = np.stack((mask_bg, mask_rv, mask_myo, mask_lv), axis=0)
        img_exp = np.expand_dims(img, 0)

        dd = {"image": torch.from_numpy(img_exp).float(),
              "mask": (torch.from_numpy(mask_exp)),
              "name": self.data[i]["name"]}
        return dd

