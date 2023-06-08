# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import numpy as np
from skimage import measure
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

try:
    import cv2
except:
    logging.warning('Could not import opencv. Augmentation functions will be unavailable.')
else:
    def rotate_image(img, angle, interp=cv2.INTER_LINEAR):

        rows, cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        return cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)


    def resize_image(im, size, interp=cv2.INTER_LINEAR):

        im_resized = cv2.resize(im, (size[1], size[0]), interpolation=interp)  # swap sizes to account for weird OCV API
        return im_resized


def convert_to_uint8(image):
    image = image - image.min()
    image = 255.0*np.divide(image.astype(np.float32),image.max())
    return image.astype(np.uint8)

def normalise_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)

def normalise_images(X):
    '''
    Helper for making the images zero mean and unit standard deviation i.e. `white`
    '''

    X_white = np.zeros(X.shape, dtype=np.float32)

    for ii in range(X.shape[0]):

        Xc = X[ii,:,:,:]
        mc = Xc.mean()
        sc = Xc.std()

        Xc_white = np.divide((Xc - mc), sc)

        X_white[ii,:,:,:] = Xc_white

    return X_white.astype(np.float32)


def reshape_2Dimage_to_tensor(image):
    return np.reshape(image, (1,image.shape[0], image.shape[1],1))


def keep_largest_connected_components(mask):
    '''
    Keeps only the largest connected components of each label for a segmentation mask.
    '''

    out_img = np.zeros(mask.shape, dtype=np.uint8)

    for struc_id in [1, 2, 3]:

        binary_img = mask == struc_id
        blobs = measure.label(binary_img, connectivity=1)

        props = measure.regionprops(blobs)

        if not props:
            continue

        area = [ele.area for ele in props]
        largest_blob_ind = np.argmax(area)
        largest_blob_label = props[largest_blob_ind].label

        out_img[blobs == largest_blob_label] = struc_id

    return out_img

def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped

def crop_or_pad_to_size(slice, nx, ny):
    x = slice.shape[0]
    y = slice.shape[1]

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny,:]
    else:
        slice_cropped = np.zeros((nx, ny ,slice.shape[-1])).astype(slice.dtype)
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :,:] = slice[:, y_s:y_s + ny,:]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y, :] = slice[x_s:x_s + nx, :, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y, :] = slice[:, :, :]

    return slice_cropped

def crop_or_pad_to_size_3d(slice, nx, ny, nz):
    x = slice.shape[0]
    y = slice.shape[1]
    z = slice.shape[2]

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    z_s = (z - nz) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2
    z_c = (nz - z) // 2

    if x > nx and y > ny and z > nz:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny, z_s:z_s + nz]
    else:
        slice_cropped = np.zeros((nx, ny, nz))
        if x <= nx and y > ny and z > nz:
            slice_cropped[x_c:x_c + x, :,:] = slice[:, y_s:y_s + ny, z_s:z_s + nz]
        elif x > nx and y <= ny and z > nz:
            slice_cropped[:, y_c:y_c + y, :] = slice[x_s:x_s + nx, :, z_s:z_s + nz]
        elif x > nx and y > ny and z <= nz:
            slice_cropped[:, :, z_c:z_c + z] = slice[x_s:x_s + nx, y_s:y_s + ny, :]
        elif x > nx and y <= ny and z <= nz:
            slice_cropped[:, y_c:y_c + y, z_c:z_c + z] = slice[x_s:x_s + nx, :, :]
        elif x <= nx and y > ny and z <= nz:
            slice_cropped[x_c:x_c + x, :, z_c:z_c + z] = slice[:, y_s:y_s + ny, :]
        elif x <= nx and y <= ny and z > nz:
            slice_cropped[x_c:x_c + x, y_c:y_c + y, :] = slice[:, :, z_s:z_s + nz]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y, z_c:z_c + z] = slice[:, :, :]

    return slice_cropped

def crop_or_pad_to_size_4d(slice, nx, ny, nz):
    x = slice.shape[0]
    y = slice.shape[1]
    z = slice.shape[2]

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    z_s = (z - nz) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2
    z_c = (nz - z) // 2

    if x > nx and y > ny and z > nz:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny, z_s:z_s + nz, :]
    else:
        slice_cropped = np.zeros((nx, ny, nz, slice.shape[-1]))
        if x <= nx and y > ny and z > nz:
            slice_cropped[x_c:x_c + x, :,:, :, :] = slice[:, y_s:y_s + ny, z_s:z_s + nz, :]
        elif x > nx and y <= ny and z > nz:
            slice_cropped[:, y_c:y_c + y, :, :] = slice[x_s:x_s + nx, :, z_s:z_s + nz, :]
        elif x > nx and y > ny and z <= nz:
            slice_cropped[:, :, z_c:z_c + z, :] = slice[x_s:x_s + nx, y_s:y_s + ny, :, :]
        elif x > nx and y <= ny and z <= nz:
            slice_cropped[:, y_c:y_c + y, z_c:z_c + z, :] = slice[x_s:x_s + nx, :, :, :]
        elif x <= nx and y > ny and z <= nz:
            slice_cropped[x_c:x_c + x, :, z_c:z_c + z, :] = slice[:, y_s:y_s + ny, :, :]
        elif x <= nx and y <= ny and z > nz:
            slice_cropped[x_c:x_c + x, y_c:y_c + y, :, :] = slice[:, :, z_s:z_s + nz, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y, z_c:z_c + z, :] = slice[:, :, :, :]

    return slice_cropped
