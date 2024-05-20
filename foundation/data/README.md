# Data Preparation

Data preparation and loading is a challenging part in this codebase as a large number of heterogeneous datasets are used. Below are the instructions to prepare dataset manifests.

## Step 1: Download raw datasets

Download datasets from their original sources, or prepare your own datasets. For simplicity, in this repo, we assume datasets are stored under ```YOUR_DATA_ROOT_DIR/datasets/<dataset_name>```. 

## Step 2: Prepare raw datasets into manifests

- Complete the ```prepare_files()``` function in ```prepare_each_dataset.py``` based on your raw dataset files. An example is presented in the code.
- For each dataset, this function generates manifests for each split (train/val/test). The manifest is stored under ```YOUR_DATA_ROOT_DIR/audio-flamingo-data/dataset_files/```. The filenames are in the format of ```<dataset_name>-<flamingo_task>/<split>.json```.
- The ```<dataset_name>``` and ```<flamingo_task>``` used in Audio Flamingo can be found in ```configs/*.yaml``` --> data_config --> dataset_blending_config. We named ```<flamingo_task>``` with camel case style (e.g. AudioCaptioning, SceneClassification, etc.) with no "-" in the middle. 
- The structure of manifests can be found within the ```prepare_files()``` function.
- Usage: ```python prepare_each_dataset.py -d <dataset_name> -f <flamingo_task>```. For example: ```python prepare_each_dataset.py -d audiocaps -f AudioCaptioning```.

## Step 3 (Optional): Prepare retrieval-based in-context learning (ICL) manifests

- This step is needed if you want to train an ICL model that can take >1 samples in-context (e.g. for retrieval-augmented audio captioning). This step is not needed if you just want to train an audio understanding model that outputs one text answer for one audio. 
- Laion-CLAP ([official repo](https://github.com/LAION-AI/CLAP)) is needed to compute audio embeddings. Download pretrained checkpoints to ```<YOUR_LAION_CLAP_ROOT_DIR>```, which you can modify in the ```load_clap_model()``` function.
- ```faiss-gpu``` package is needed to build hashing and querying kNN samples. Unfortunately, we couldn't find a version that is compatible with the docker image used for training. We ended up with copying the docker image to a new image and installing ```faiss-gpu``` just for data preparation.
- The datasets and tasks that used ICL manifests can be found in ```configs/*_sft_*.yaml``` --> data_config --> dataset_blending_config, wherever there is the ```interleaved_knn-train``` flag.
- Usage ```python prepare_each_dataset.py -d <dataset_name> -f <flamingo_task> --interleave```. For example: ```python prepare_each_dataset.py -d audiocaps -f AudioCaptioning --interleave```. This will also execute manifest preparation in Step 2. 

# Additional notes

As it's hard to define an epoch in this complicated data mixing scenario, we used a "virtual epoch" concept where we extracted a fraction from each dataset and mix them to form a virtual epoch. The fractions are computed based on each dataset weight (number of training epochs based on Table 7 and 8 from our paper). The virtual epoch size is determined by ```configs/*.yaml``` --> data_config --> dataset_blending_global_weight. Then, the num_epochs in ```configs/*.yaml``` --> train_config is its reciprocal. The final checkpoint will be ```checkpoint_{num_epochs-1}.pt```.