# cs231n-segmentation
CS231N Spring 2018 Project

## Setup 

On Unix environment: 
`source setup.sh `
This downloads the COCO-Stuff API, and the dataset (training and validation set 2017).

## Test API 

There's a Jupyter Notebook in the cocostuff/PythonAPI subfolder that shows how to use the COCO API to load annotations and images
and display the annotations. 

Note: If the 1st cell raises an error when importing pycoco, then do `make` in the PythonAPI repo and restart the kernel. 
Second Note: The jupyter notebook is set to look for the 2014 dataset so this needs to be modified.
