

## Train
### 1. Prepare training data
Download Brats18 dataset and IXI dataset.  

process dataset by processDataset.py (data_tools/).
### 2. Begin to train
python train.py -v "version" -p train --train_yaml "xxx.yaml"

## Quick Test 
python test.py -v "version" -s 153 -t tester_Matlab --test_dataset_name "dataset"

