# IRA-Unet
## **Introduction**

This repository contains the  implementation for automated cardiac segmentation introduced in the following paper: "IRA-Unet: Inception Residual Attention Unet In Adversarial Network For Cardiac MRI Segmentation"

## **Steps to train and test the model:**

1.Register and download ACDC-2017 dataset from https://www.creatis.insa-lyon.fr/Challenge/acdc/index.html

2.Run the script main.py.

```
python main.py --Data-root --Loc-path --Save-path
```
Data-root: Folder to which you put training data

Loc-path: Where to save preprocessed and localized images

Save-path: Where to save segmented images
