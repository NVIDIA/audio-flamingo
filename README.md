
<div style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <h2>
    <a href="https://github.com/NVIDIA/audio-flamingo" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
      <img src="assets/af_logo.png" alt="Audio Flamingo 2 🔥🚀🔥" width="100">
    </a>
    Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities
  </h2>
</div>

<div style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2503.03983"><img src="https://img.shields.io/badge/arXiv-2503.03983-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/AF2/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo"><img src='https://img.shields.io/badge/Github-Audio Flamingo 2-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social"></a>
</div>

<div style="display: flex; justify-content: center; margin-top: 10px;">
<a href="https://huggingface.co/nvidia/audio-flamingo-2"><img src="https://img.shields.io/badge/🤗-Checkpoints (3B)-ED5A22.svg" style="margin-right: 5px;"></a>
<a href="https://huggingface.co/nvidia/audio-flamingo-2-1.5B"><img src="https://img.shields.io/badge/🤗-Checkpoints (1.5B)-ED5A22.svg" style="margin-right: 5px;"></a>
<a href="https://huggingface.co/nvidia/audio-flamingo-2-0.5B"><img src="https://img.shields.io/badge/🤗-Checkpoints (0.5B)-ED5A22.svg" style="margin-right: 5px;"></a>
</div>

<div style="display: flex; justify-content: center; margin-top: 10px;">
<a href="https://huggingface.co/spaces/nvidia/audio-flamingo-2"><img src="https://img.shields.io/badge/🤗-Gradio Demo (3B)-5F9EA0.svg" style="margin-right: 5px;"></a>
<a href="https://huggingface.co/spaces/nvidia/audio-flamingo-2-1.5B"><img src="https://img.shields.io/badge/🤗-Gradio Demo (1.5B)-5F9EA0.svg" style="margin-right: 5px;"></a>
<a href="https://huggingface.co/spaces/nvidia/audio-flamingo-2-0.5B"><img src="https://img.shields.io/badge/🤗-Gradio Demo (0.5B)-5F9EA0.svg" style="margin-right: 5px;"></a>
  
</div>

## Overview

This repo contains the PyTorch implementation of [Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities](https://arxiv.org/abs/2503.03983). Audio Flamingo 2 achieves the state-of-the-art performance across over 20 benchmarks, with only a 3B parameter small language model. It is improved from our previous [Audio Flamingo](https://arxiv.org/abs/2402.01831). 

- We introduce two datasets, AudioSkills for expert audio reasoning, and LongAudio for long audio understanding, to advance the field of audio understanding.

- Audio Flamingo 2 has advanced audio understanding and reasoning capabilities. Especially, Audio Flamingo 2 has expert audio reasoning abilities, and can understand long audio up to 5 minutes.

- Audio Flamingo 2 outperforms larger and proprietary LALMs across 20+ benchmarks, despite being smaller (3B) and trained exclusively on public datasets.

## Main Results

Audio Flamingo 2 outperforms prior SOTA models including GAMA, Audio Flamingo, Qwen-Audio, Qwen2-Audio, LTU, LTU-AS, SALMONN, AudioGPT, Gemini Flash v2, Gemini Pro v1.5, and GPT-4o-audio on a number of understanding and reasoning benchmarks.

<div align="center">
  <img class="img-full" src="assets/af2_radar.png" width="300">
</div>

<div align="center">
  <img class="img-full" src="assets/af2_table2.png" width="400">
</div>

## Audio Flamingo 2 Architecture

Audio Flamingo 2 uses a cross-attention architecture similar to [Audio Flamingo](https://arxiv.org/abs/2402.01831) and [Flamingo](https://arxiv.org/abs/2204.14198). Audio Flamingo 2 can take up to 5 minutes of audio inputs. 

<div align="center">
  <img class="img-full" src="assets/af2_arch.png" width="800">
</div>


## Code Structure

- The folder ```inference/``` contains inference code of Audio Flamingo 2.

The structure is highly based on the [Open Flamingo](https://github.com/mlfoundations/open_flamingo) repo (commit ```a05dcba```).

<!-- Within each folder, the structure is highly based on the [Open Flamingo](https://github.com/mlfoundations/open_flamingo) repo (commit ```a05dcba```). Each folder is self-contained and we expect no cross dependencies between these folders. -->

## References

<!-- The main training and inferencing code within each folder (```foundation/```, ```chat/```, ```inference/```), including ```train/```, ```src/```, ```data/```, and ```configs/```,  -->
The code under ```src/``` and ```configs/``` are modified from [Open Flamingo](https://github.com/mlfoundations/open_flamingo) (commit ```a05dcba```) (MIT license), which borrows from [flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch) (MIT license), [flamingo-mini](https://github.com/dhansmair/flamingo-mini) (MIT license), and [open_clip](https://github.com/mlfoundations/open_clip) (MIT license). ```src/helpers.py``` also includes self-attention implementations based on [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch) (MIT license), which borrows from [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) (MIT license). ```my_laion_clap``` is adapted from [LAION-AI/CLAP](https://github.com/LAION-AI/CLAP) (CC0-1.0 license).

## License

- The code in this repo is under MIT license.
- The checkpoints are for non-commercial use only (see NVIDIA OneWay Noncommercial License). They are also subject to the [Qwen Research license](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE), the [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and the original licenses accompanying each training dataset.
- Notice: Audio Flamingo 2 is built with Qwen-2.5. Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.


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
@article{ghosh2025audio,
  title={Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities},
  author={Ghosh, Sreyan and Kong, Zhifeng and Kumar, Sonal and Sakshi, S and Kim, Jaehyeon and Ping, Wei and Valle, Rafael and Manocha, Dinesh and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2503.03983},
  year={2025}
}
```