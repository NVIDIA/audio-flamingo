# Data Preparation

Data preparation and loading is a challenging part in this codebase as complex formats are used. Below are the instructions to prepare dataset manifests.

## Step 1: Download raw datasets

Download datasets from their original sources, or prepare your own datasets. For simplicity, in this repo, we assume datasets are stored under ```YOUR_DATA_ROOT_DIR/datasets/<dataset_name>```.

## Step 2: Prepare dialogues

Follow the instructions in Appendix B in our paper to generate dialogues from rich metadata and filter for quality.

## Step 3: Prepare raw datasets into manifests

- Modify the ```prepare_files()``` function in ```prepare_each_dataset.py``` based on your raw dataset files.
- For each dataset, this function generates manifests for each split (train/val/test). The manifest is stored under ```YOUR_DATA_ROOT_DIR/audio-flamingo-data/dataset_files/```. The filenames are in the format of ```<dataset_name>-Dialog/train.json```.
- The ```<dataset_name>``` used in Audio Flamingo can be found in ```configs/*.yaml``` --> data_config --> dataset_blending_config.
- The structure of manifests can be found within the ```prepare_files()``` function.
- Usage: ```python prepare_each_dataset.py -d <dataset_name>```.
