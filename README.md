
<div align="center" style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <a href="https://github.com/NVIDIA/audio-flamingo" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
    <img src="static/logo-no-bg.png" alt="Audio Flamingo 3 🔥🚀🔥" width="120">
  </a>
</div>
<div align="center" style="display: flex; justify-content: center; align-items: center; text-align: center;">
    <h2>
    Audio Flamingo Sound-CoT Technical Report: Improving Chain-of-Thought Reasoning in Sound Understanding
    </h2>
</div>

## AF-Reasoning-Eval

The AF-Reasoning-Eval benchmark is included in ```AF_Reasoning_Eval/```. 

- The ```AQA``` subset is derived from [Clotho-AQA](https://zenodo.org/records/6473207), and our filenames point to Clotho-AQA filenames. Note that the Clotho-AQA audio files follow their original license. 
- The ```Classification``` subset is derived from [FSD50K](https://zenodo.org/records/4060432), and our filenames point to FSD50K filenames. Note that the FSD50K audio files follow their original license. 
- Our metadata are released under ```CC-BY 4.0```.

## AF-CoT-Train

The ```AF-CoT-Train``` generation pipelines are included in ```AF_CoT_Train/```. It is recommended to use ```alg_6_AQA_subquestions.py``` to generate reasoning chains for AQA samples and ```alg_8_Classification_MCQ.py``` to generate reasoning chains for classification samples.

The dataset is released in [this link](https://huggingface.co/datasets/nvidia/AF-Think/tree/main/af_cot_train).

## Inference code
The instruction to run inference of the CoT model is [here](https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_2/inference_HF_pretrained#steps-of-inference-of-the-cot-model).


## License

- The code in this repo is under MIT license.
- ```AF-Reasoning-Eval``` is released under ```CC-BY 4.0```.
- ```AF-CoT-Train``` is for non-commercial use only (see NVIDIA OneWay Noncommercial License).
- The checkpoints are for non-commercial use only (see NVIDIA OneWay Noncommercial License). They are also subject to other restrictions (see ``` README``` and ```incl_licenses``` within each branch).
- Notice: Audio Flamingo Sound-CoT models are built with Qwen-2.5. Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.


## Citation

- Audio Flamingo
```
@inproceedings{kong2024audio,
  title={Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities},
  author={Kong, Zhifeng and Goel, Arushi and Badlani, Rohan and Ping, Wei and Valle, Rafael and Catanzaro, Bryan},
  booktitle={International Conference on Machine Learning},
  pages={25125--25148},
  year={2024},
  organization={PMLR}
}
```

- Audio Flamingo 2
```
@inproceedings{
  ghosh2025audio,
  title={Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities},
  author={Ghosh, Sreyan and Kong, Zhifeng and Kumar, Sonal and Sakshi, S and Kim, Jaehyeon and Ping, Wei and Valle, Rafael and Manocha, Dinesh and Catanzaro, Bryan},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=xWu5qpDK6U}
}
```

- Audio Flamingo 3
```
@article{goel2025audio,
  title={Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models},
  author={Goel, Arushi and Ghosh, Sreyan and Kim, Jaehyeon and Kumar, Sonal and Kong, Zhifeng and Lee, Sang-gil and Yang, Chao-Han Huck and Duraiswami, Ramani and Manocha, Dinesh and Valle, Rafael and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2507.08128},
  year={2025}
}
```
