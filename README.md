
<div align="center" style="display: flex; justify-content: center; align-items: center; text-align: center;">
  <a href="https://github.com/NVIDIA/audio-flamingo" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
    <img src="static/logo-no-bg.png" alt="Audio Flamingo 3 🔥🚀🔥" width="120">
  </a>
</div>
<div align="center" style="display: flex; justify-content: center; align-items: center; text-align: center;">
    <h2>
    Audio Flamingo: Series of Advanced Audio Understanding Language Models
    </h2>
</div>

## Overview
In this repo, we present the **Audio Flamingo** series of advanced audio understanding Language models:
- [Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities](https://arxiv.org/abs/2402.01831) (ICML 2024)
- [Audio Flamingo 2: An Audio-Language Model with Long-Audio Understanding and Expert Reasoning Abilities](https://arxiv.org/abs/2503.03983) (ICML 2025)
- [Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models](https://arxiv.org/abs/2507.08128) (arxiv)

## Audio Flamingo

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2402.01831"><img src="https://img.shields.io/badge/arXiv-2402.01831-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://audioflamingo.github.io/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/tree/legacy_audio_flamingo_1"><img src='https://img.shields.io/badge/Github-Audio Flamingo 1-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social"></a>
  <a href="https://huggingface.co/nvidia/audio-flamingo"><img src="https://img.shields.io/badge/🤗-Checkpoints_(1.3B)-ED5A22.svg" style="margin-right: 5px;"></a>
</div>

<br>

Audio Flamingo is our first audio language model based on the [Flamingo](https://arxiv.org/abs/2204.14198) architecture. It is based on an 1.3B language model and has in-context few-shot learning and multi-turn dialogue abilities (see [Audio Dialogues](https://arxiv.org/abs/2404.07616) for details of dialogue data). We curated about 5.9M audio-text pairs to train our model. It achieves the SOTA results on several zero-shot, few-shot, and in-distribution benchmarks of captioning, classification, and question answering.

<br>
<br>

<div align="center">
  <img class="img-full" src="static/af1_radial.png" width="300">
</div>

<br>

<div align="center">
  <img class="img-full" src="https://github.com/NVIDIA/audio-flamingo/raw/legacy_audio_flamingo_1/assets/audio_flamingo_arch.png" width="600">
</div>

## Audio Flamingo 2

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2503.03983"><img src="https://img.shields.io/badge/arXiv-2503.03983-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/AF2/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_2"><img src='https://img.shields.io/badge/Github-Audio Flamingo 2-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social"></a>
  <a href="https://huggingface.co/spaces/nvidia/audio-flamingo-2"><img src="https://img.shields.io/badge/🤗-Gradio Demo (3B)-5F9EA0.svg" style="margin-right: 5px;"></a>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
<a href="https://huggingface.co/nvidia/audio-flamingo-2"><img src="https://img.shields.io/badge/🤗-Checkpoints_v20250625_(3B)-ED5A22.svg" style="margin-right: 5px;"></a>
<a href="https://huggingface.co/nvidia/audio-flamingo-2-1.5B"><img src="https://img.shields.io/badge/🤗-Checkpoints_v20250625_(1.5B)-ED5A22.svg" style="margin-right: 5px;"></a>
<a href="https://huggingface.co/nvidia/audio-flamingo-2-0.5B"><img src="https://img.shields.io/badge/🤗-Checkpoints_v20250625_(0.5B)-ED5A22.svg" style="margin-right: 5px;"></a>
</div>

<br>

Audio Flamingo 2 significantly improves Audio Flamingo in several aspects. First, we re-trained a better CLAP for with stronger text understanding abilities. Second, we scaled up the training set to about 10M audio-text pairs with a focus on several understanding skills (AudioSkills) and understanding of longer audio (LongAudio). Third, we carefully ablate the training recipes and curriculums and found a 3-stage training strategy yields the best results. Audio Flamingo 2 is based on a 3B langauge model. It achieves the SOTA results on several individual and mixed audio understanding benchmarks of captioning, classification, and question answering. It can also understand longer audio up to 5 minutes. 

<br>
<br>

<div align="center">
  <img class="img-full" src="https://github.com/NVIDIA/audio-flamingo/raw/audio_flamingo_2/assets/af2_radar.png" width="300">
</div>

<div align="center">
  <img class="img-full" src="https://github.com/NVIDIA/audio-flamingo/raw/audio_flamingo_2/assets/af2_table2.png" width="400">
</div>

<div align="center">
  <img class="img-full" src="https://github.com/NVIDIA/audio-flamingo/raw/audio_flamingo_2/assets/af2_arch.png" width="800">
</div>


## Audio Flamingo 3

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2507.08128"><img src="https://img.shields.io/badge/arXiv-2507.08128-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/AF3/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_3"><img src='https://img.shields.io/badge/Github-Audio Flamingo 3-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social"></a>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/nvidia/audio-flamingo-3">
    <img src="https://img.shields.io/badge/🤗-Checkpoints-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/nvidia/audio-flamingo-3-chat">
    <img src="https://img.shields.io/badge/🤗-Checkpoints (Chat)-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/spaces/nvidia/audio_flamingo_3">
    <img src="https://img.shields.io/badge/🤗-Gradio Demo (7B)-5F9EA0.svg" style="margin-right: 5px;">
  </a>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://huggingface.co/datasets/nvidia/AudioSkills">
    <img src="https://img.shields.io/badge/🤗-Dataset: AudioSkills--XL-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/LongAudio">
    <img src="https://img.shields.io/badge/🤗-Dataset: LongAudio--XL-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Chat">
    <img src="https://img.shields.io/badge/🤗-Dataset: AF--Chat-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Think">
    <img src="https://img.shields.io/badge/🤗-Dataset: AF--Think-ED5A22.svg">
  </a>
</div>

<br>

Audio Flamingo 3 is our latest model based on a 7B language model and the [LLaVA](https://arxiv.org/abs/2304.08485) architecture. We trained our unified AF-Whisper audio encoder based on [Whisper](https://arxiv.org/abs/2212.04356) to handle understanding beyond speech recognition. We included speech-related tasks in Audio Flamingo 3 and scaled up the training dataset to about 50M audio-text pairs. Therefore, Audio Flamingo 3 is able to handle all three modalities in audio: **sound**, **music**, and **speech**. It outperforms prior SOTA models including GAMA, Audio Flamingo, Audio Flamingo 2, Qwen-Audio, Qwen2-Audio, Qwen2.5-Omni, LTU, LTU-AS, SALMONN, AudioGPT, Gemini Flash v2 and Gemini Pro v1.5 on a number of understanding and reasoning benchmarks.

Audio Flamingo 3 can take up to 10 minutes of audio inputs, and has a streaming TTS module (AF3-Chat) to output voice. 

<br>
<br>


<div align="center">
  <img class="img-full" src="static/af3_radial-1.png" width="400">
</div>

<div align="center">
  <img class="img-full" src="static/af3_sota.png" width="600">
</div>


<div align="center">
  <img class="img-full" src="static/af3_main_diagram-1.png" width="800">
</div>

## Code Structure

Each branch includes the individual code to train and inference Audio Flamingo.

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
<a href="https://github.com/NVIDIA/audio-flamingo/tree/legacy_audio_flamingo_1"><img src='https://img.shields.io/badge/Github-Audio Flamingo 1-9C276A' style="margin-right: 5px;"></a>
<a href="https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_2"><img src='https://img.shields.io/badge/Github-Audio Flamingo 2-9C276A' style="margin-right: 5px;"></a>
<a href="https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_3"><img src='https://img.shields.io/badge/Github-Audio Flamingo 3-9C276A' style="margin-right: 5px;"></a>
</div>


## License

- The code in this repo is under MIT license.
- The checkpoints are for non-commercial use only (see NVIDIA OneWay Noncommercial License). They are also subject to other restrictions (see ``` README``` and ```incl_licenses``` within each branch).
- Notice: Audio Flamingo is built with OPT-IML and is subject to the OPT-IML license. 
- Notice: Audio Flamingo 2 and Audio Flamingo 3 are built with Qwen-2.5. Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All Rights Reserved.


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
