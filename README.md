# RU-Net: An implementation of training and prediction for tumor classification

## Usage

First, clone the repository locally:
```
$ git clone https://github.com/LMMMEng/LLD-MMRI2023.git
$ cd RU-net
```
Dependencies are listed as follows:
```
$ conda env create -f environment.yml
```

## Data file directory preparation

The data are stored in the following structure(Note that the radiomics and label files each contain five folds for cross-validation):   
```
Data
    ├── Image
        ├── type_a_tumor_x.nii.gz
        ├── type_a_tumor_y.nii.gz

        ├── type_b_tumor_x.nii.gz
        ├── type_b_tumor_y.nii.gz

    ├── Label
        ├── exp1
            ├── exp1_fold1
                ├── fold_1_train_label.txt
                ├── fold_1_val_label.txt
                ├── fold_1_test_label.txt

    ├── Radiomics_feat
        ├── exp1
            ├── exp1_fold1
                ├── fold_1_feat.csv

```
Descriptions: 


**Note**: The "Image" folder contains all image samples at the tumor level; the "Label" folder organizes the dataset at the case level and provides ground truth annotations. Based on this organization, corresponding data is extracted from the "Image" and "Radiomics_feat" folders.

## Implementation
### Training and Validation

```
bash do_train_liver_K.sh
```

### Training and Validation

```
bash do_test_liver_K.sh
```