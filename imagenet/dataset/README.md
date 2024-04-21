## Download ImageNet1K dataset

### Downloading the ImageNet Dataset

The ImageNet dataset is a large dataset of annotated photographs intended primarily for use in visual object recognition software research. To access the ImageNet dataset, you must follow several steps to ensure compliance with its usage policies.

### Prerequisites

1. **Register for Access:** You need to have an approved account to access ImageNet. You can apply for access at the ImageNet website. Please follow their guidance for registration and acceptance of terms and conditions.
   
   Link: [ImageNet Registration](http://image-net.org/download)

### Steps to Download Manually

After receiving your approval, you can download the dataset by following these steps:

1. **Login to Your Account:** Once your registration is approved, log in to your account on the ImageNet website.

2. **Access the Download Section:** Navigate to the download section where you can find links to download the dataset.

3. **Download the Correct Version:** ImageNet comes in several versions and formats, make sure you download:
   - ILSVRC2012_img_val.tar (about 6.3 GB). MD5: 29b22e2961454d5413ddabcf34fc5622
     - ```bash
       # Alternatively you can use wget to download the file
       wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
       ```
   - ILSVRC2012_img_train.tar (about 138 GB). MD5: 1d675b47d978889d74fa0da5fadfb00e
     - ```bash
       # Alternatively you can use wget to download the file
       wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
       ```


### Post-Download

After downloading the dataset, verify the integrity of the files using checksums (MD5) above. Consider setting up appropriate data storage and backup strategies due to the size of the dataset before extracting the files.

### Extract ImageNet1K dataset
```bash
#!/bin/bash
# Taken from https://gist.github.com/bonlime/4e0d236cf98cd5b15d977dfa03a63643
# script to fully prepare ImageNet dataset

## 1. Download the data
# get ILSVRC2012_img_val.tar (about 6.3 GB). MD5: 29b22e2961454d5413ddabcf34fc5622
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
# get ILSVRC2012_img_train.tar (about 138 GB). MD5: 1d675b47d978889d74fa0da5fadfb00e
# wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar

## 2. Extract the training data:
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
cd ..

## 3. Extract the validation data and move images to subfolders:
mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash


## 4. Delete corrupted image
# there is one png under JPEG name. some readers fail on this image so need to remove it
# this line is commented by default to avoid such unexpected behaviour
# rm train/n04266014/n04266014_10835.JPEG

```

After extracing the files, place the train, val and test (if you downloaded it) folders, in the `data` directory inside the imagenet directory. Create it if it doesn't exist.

### Usage and License

Please be sure to review the terms of use provided by ImageNet. These terms dictate what you can and cannot do with the dataset and may include requirements for attribution, restrictions on commercial use, etc.

For more detailed information and latest updates, visit the [official ImageNet website](http://image-net.org/).
