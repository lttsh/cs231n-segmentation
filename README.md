# cs231n-segmentation
CS231N Spring 2018 Project

## Setup 

In Unix environment: 
`> source setup.sh `

This downloads the [COCO-Stuff API](https://github.com/nightrome/cocoapi) and the dataset (training and validation set 2017).

After running the script, to test the API, launch the jupyter notebook `cocostuff/PythonAPI/pycocoDemo.ipynb`.
Note: If the 1st cell raises an error when importing pycoco, then do `make` in the PythonAPI repo and restart the kernel. 
Second Note: The jupyter notebook is set to look for the 2014 dataset so this needs to be modified.
