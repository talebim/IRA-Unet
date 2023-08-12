# IRA-Unet
## **Introduction**

This repository contains the  implementation for automated cardiac segmentation introduced in the following paper: "[IRA-Unet: Inception Residual Attention Unet in Adversarial Network for Cardiac MRI Segmentation](https://www.techrxiv.org/articles/preprint/IRA-Unet_Inception_Residual_Attention_Unet_in_Adversarial_Network_for_Cardiac_MRI_Segmentation/23896641)"

## **Steps to train and test the model:**

1.Register and download ACDC-2017 dataset from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html

2.Run the script preprocess.py.
```
python preprocess.py --data-root your DATA_DIR
```

3.A folder named loc192 will be created which contain preprocessed and croped train and validation dataset.

4.Run the script main.py.

```
python main.py --data-root your DATA_DIR --save-path your OUT_DIR
```

The segmented image of test set will be saved in outputs

## **Steps to test the pretrained model:**
1.To reproduce the results, download weights of our best model from **[here](https://drive.google.com/file/d/1iMSjN4b1y_uBoCqYYazqd33tP7uWjvCq/view?usp=drive_link)**
 
2.Put the last.ckpt file in ckpt folder

2.Run the script predict.py.
```
python predict.py --data-root your DATA_DIR --save-path your OUT_DIR
```

## **Requirements**
The code is tested on Ubuntu 20.04 with the following components:

Software
Python 3.8
pytorch 1.13
CUDA 11.8 

## Logs
To launch the tensorboard instance run
```
tensorboard --logdir 'logs/IRA-Unet'
```
It will give a view on the evolution of the loss for both the training and validation data.
