# Semantic Segmentation with Adversarial Networks for Masked Style Transfer

## Description 
Code for the final project for CS231N (Spring 2018). We trained a semantic segmentation network adversarially to generate segmentation masks and use them to apply different styles to an image using masked style transfer. The final model was trained on the `animal` super-category in the MS-COCO dataset. 

Details can be found in the report (to be posted) and [poster](https://drive.google.com/file/d/1blNW0WKBjmzc5Uv2hgqOX-Jt2cw2fwsS/view?usp=sharing).

## Setup 

In Unix environment: 
```
> source setup.sh 
```

To test the API, launch the jupyter notebook `cocostuff/PythonAPI/pycocoDemo.ipynb`.

Note: If the 1st cell raises an error when importing pycoco, then do `make` in the PythonAPI repo and restart the kernel. 
Second Note: The jupyter notebook is set to look for the 2014 dataset so this needs to be modified.
