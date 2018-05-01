## Download COCO-Stuff API and install Python API
## Not sure if needed/ how to use yet.
# wget https://github.com/nightrome/cocoapi/archive/master.zip
# unzip master.zip
# rm master.zip
# mv cocoapi-master/ coco/
# cd coco/PythonAPI && make && cd ../..

## Downloqd COCO-Stuff dataset and baseline model
wget https://github.com/nightrome/cocostuff/archive/master.zip
unzip master.zip
rm master.zip
mv cocostuff-master/ cocostuff/
cd cocostuff

# Download everything
# Images
# wget --directory-prefix=downloads http://images.cocodataset.org/zips/train2017.zip
wget --directory-prefix=downloads http://images.cocodataset.org/zips/val2017.zip

# PNG annotations
wget --directory-prefix=downloads http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# COCO style annotations
wget --directory-prefix=downloads http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuff_trainval2017.zip
wget --directory-prefix=downloads http://images.cocodataset.org/annotations/annotations_trainval2017.zip

# Unpack everything
mkdir -p dataset/images
mkdir -p dataset/annotations

# unzip downloads/train2017.zip -d dataset/images/
unzip downloads/val2017.zip -d dataset/images/
unzip downloads/stuffthingmaps_trainval2017.zip -d dataset/annotations/
unzip downloads/stuff_trainval2017.zip -d dataset/annotations/
unzip downloads/annotations_trainval2017.zip -d dataset/annotations/
cd ..
