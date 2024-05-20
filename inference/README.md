# Audio Flamingo Inference Code

This folder contains inference code of Audio Flamingo. The code supports both the foundation and chat models.

- The foundation model is trained on a number of audio captioning, audio question answering, and audio classification datasets. The model takes an audio and text prompt as input, and outputs the answer. The config file is ```configs/foundation.yaml```. 

- The chat model is finetuned on our generated multi-turn dialogue dataset. It can chat with a human up to 4 rounds. The config file is ```configs/chat.yaml```.

## Get paths ready

Let ```YOUR_REPO_ROOT_DIR``` be the absolute path to this repo. We use the following structure

```
YOUR_REPO_ROOT_DIR/
  - foundation/
  - chat/
  - inference/  # you are here
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
- ```inference_examples.py --> __main__```
- ```launch_gradio.py --> __main__```


## Usage of inference code

The example inference code is in ```inference_examples.py```. Within its ```__main__``` function, 
- ```data_root``` is the root folder of audio datasets.
- ```checkpoint_path``` is the path to the model checkpoint. 
- ```inference_kwargs``` is used to setup the inference kwargs used in the HuggingFace transformers package. [Here](https://huggingface.co/blog/how-to-generate) is a simple tutorial on inference algorithms by HuggingFace. 
- ```items``` contains a list of samples to inference. 

## Gradio code

The sample code to launch an interactive gradio interface is in ```launch_gradio.py```. It requires the gradio package. For now it only supports one round of dialogue with the chat model. Different from the ```inference_examples.py``` code, the gradio code uses additional CLAP filtering for better generation quality. 

## Retrieval-based ICL inference code

As the inference code for retrieval-based ICL is more complex, we leave the instructions to ```foundation/inference/README.md``` and the inference code to ```foundation/inference/inference.py```.