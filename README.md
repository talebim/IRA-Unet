# IRA-Unet
## **Introduction**

This repository contains the  implementation for automated cardiac segmentation introduced in the following paper: "IRA-Unet: Inception Residual Attention Unet In Adversarial Network For Cardiac MRI Segmentation"

## **Steps to train and test the model:**

1.Register and download ACDC-2017 dataset from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html

2.Run the script preprocess.py.

3.A folder named loc192 will be created which contain preprocessed and croped train and validation dataset.

4.Run the script main.py.

```
python main.py --Data-root --Save-path
```
Data-root: Folder to which you put training data

Save-path: Where to save segmented images

The segmented image of test set will be saved in outputs

## **Steps to test the pretrained model:**
1.To reproduce the results, download weights of our best model from
 
2.Put the last.ckpt file in ckpt folder

2.Run the script predict.py.

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
