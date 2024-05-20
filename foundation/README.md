# Audio Flamingo Training (Foundation Model)

## Get paths ready

Let ```YOUR_REPO_ROOT_DIR``` be the absolute path to this repo. We use the following structure

```
YOUR_REPO_ROOT_DIR/
  - foundation/  # you are here
  - chat/
  - inference/
```

Replace ```YOUR_REPO_ROOT_DIR``` to your absolute path in the following places:
- ```configs/*.yaml --> clap_config --> config_root```


Let ```YOUR_DATA_ROOT_DIR``` be the absolute path to store all data, checkpoints, etc. We use the following structure
```
YOUR_DATA_ROOT_DIR/
  - datasets/
    - <dataset_name_i>/
      - files: raw data of this dataset, including raw waveforms, metadata, etc.
  
  - audio-flamingo-data/
    - dataset_files/
      - <dataset_name_i>-<flamingo_task_i>/
        - files: dataset manifests, precomputed embeddings, etc.

    - checkpoint/
      - <experiment_name>/  # same as the config file name, and train_config --> run_name in each config
        - tensorboard/
        - checkpoint_xxx.pt
        - other cached files
    
    - clap/
      - files: pretrained Microsoft-CLAP checkpoints
    
    - laion-clap-pretrained/laion_clap
      - files: pretrained Laion-CLAP checkpoints
    
    - LLM_pretrained/.cache/  # place to store HuggingFace cache instead of the default ~/.cache
```

Replace ```YOUR_DATA_ROOT_DIR``` to your absolute path in the following places:
- ```configs/*.yaml```
- ```prepare_each_dataset.py --> __main__```
- ```inference/inference.sh```

## Get data ready

Please read ```data/README.md``` for instructions on data preparation.

## Training

The following code is tested on 1 node (8 GPUs per node) of A100 (80G) GPUs. 
```
export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1 
cd train/
torchrun --nproc_per_node 8 train.py -c ../configs/<CONFIG_NAME>.yaml
```

For pretraining, use ```foundation_pretrain.yaml```. For SFT, use ```foundation_sft_*.yaml``` (these configs differ by number of ICL samples). Remember to set ```configs/foundation_sft_*.yaml --> sft_config --> pretrained_path``` and ```pretrained_ckpt``` to be the checkpoint of the pretrained model.