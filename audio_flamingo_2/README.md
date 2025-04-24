# Main training and inference code of Audio Flamingo 2


# Get data and paths ready

- Step 1, download and pre-process all datasets from their original sources based on Table 21 of the paper, and store at ```DATA_ROOT/```. 
- Step 2, create all manifests in the following format:
  - All manifests are stored at ```DATA_MANIFEST_FOLDER/```.
  - The name of manifest is ```Name-Task/Split.json```, such as ```audiocaps-AudioCaptioning/train.json, GTZAN-GenreClassification/test.json, Clotho-AQA-AQA/val.json ```. 
  - The content of each manifest is in the format of 
  ```
    {
        "split": "train",                               # train or test or val
        "split_path": "audiocaps/audio/train",          # absolute or relative path of the split,
        "flamingo_task": "audiocaps-AudioCaptioning",   # should match manifest name
        "total_num": 49273,                             # total number of samples
        "data": {                                       # note the keys are strings of 0, 1, 2, ...
            "0": {
                "name": "5228.flac",                    # filename
                "prompt": "Caption the input audio.",   # prompt
                "output": "An engine ....",             # output
                "duration": 10.0                        # audio duration in seconds
            },
            "1": {
                "name": "19105.flac",
                "prompt": "Briefly describe ....",
                "output": "A woman speaking ....",
                "duration": 10.0
            },

            ...

            "49272": {
                ...
            }
        }
    }
  ```

  - For AQA, all options are provided in the prompt as natural language: ```Question? (A) xxx. (B) yyy. (C) zzz. (D) uuu.```
  - Make sure the path to each audio file is ```os.path.join(DATA_ROOT, split_path, data["i"]["name"])```.

# Get config ready

- Modify the paths in ```configs/*.yaml``` to where you store data and model, including 
  - ```train_config.expdir, train_config.run_name```
  - ```sft_config.pretrained_path, sft_config.pretrained_ckpt```
  - ```data_config.dataset_file_root, data_config.data_root, data_config.dataset_blending_output```
  - ```clap_config.checkpoint```
  - ```model_config.cache_dir```
- Include all training and validation (or test) datasets in the ```data_config```. The key follows the manifest name (```Name-Task/Split```) and the weight is the blending weight based on Table 21 of the paper.
- Adjust ```train_config.batch_size``` and ```train_config.gradient_accumulation_steps``` based on your number of GPUs and memory.
- Adjust ```train_config.num_epochs``` and ```data_config.dataset_blending_global_weight``` based on your job time. Reduce ```data_config.dataset_blending_global_weight``` for shorter job time. Make sure their multiplication equals 1. ```checkpoint_{num_epochs-1}.pt``` is the last checkpoint at the end of training.

# Scripts

The training script is 
```
cd train/
python -u train.py -c ../configs/pretrain.yaml  # Pretrain
python -u train.py -c ../configs/sft.yaml       # SFT
```

The evaluation script is
```
cd eval/
sh inference.sh audiocaps-AudioCaptioning/test  # make sure the 'Name-Task/test' key is in data_config.valid_dataset_config
```