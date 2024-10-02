# Model Overview

## Description:
Audio Flamingo is a novel audio-understanding language model for

- understanding audio,
- quickly adapting to unseen tasks via in-context learning and retrieval, and
- understanding and responding to multi-turn dialogues

We introduce a series of training techniques, architecture design, and data strategies to enhance our model with these abilities. Extensive evaluations across various audio understanding tasks confirm the efficacy of our method, setting new state-of-the-art benchmarks.

<center><img src="https://github.com/NVIDIA/audio-flamingo/raw/main/assets/audio_flamingo_arch.png" width="800"></center>

**This model is ready for non-commercial research-only.**
<br>


## References(s):
* [Audio Flamingo: A Novel Audio Language Model with Few-Shot Learning and Dialogue Abilities](https://arxiv.org/abs/2402.01831)  <br>
* [Project Page](https://github.com/NVIDIA/audio-flamingo)  <br> 
* [Demo Website](https://audioflamingo.github.io/)  <br> 

## Model Architecture:
**Architecture Type:** Transformer <br>
**Network Architecture:** Audio Flamingo 

Audio Flamingo is a Flamingo-style architecture with frozen audio feature extractor, trainable transformation layers and xattn-dense layers, and language model layers. 

## Input:
**Input Types:** Audio, Text <br>
**Input Format:** Wav/MP3/Flac, String <br>
**Input Parameters:** None <br>
**Maximum Audio Input Lengths:** 33.25 seconds <br>
**Maximum Text Input Lengths:** 512 tokens <br>

## Output:
**Output Type:** Text <br>
**Output Format:** String <br>
**Output Parameters:** None <br>

## Software Integration:
**Runtime Engine(s):** PyTorch

**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Ampere <br>
* NVIDIA Hopper <br>

## Preferred/Supported Operating System(s):
* Linux


## Model Version(s):
* v1.0

## Training, Testing, and Evaluation Datasets:

### Training Dataset:
Audio Flamingo is trained with **publicly available** datasets under various licenses, with the most restricted ones being non-commercial/research-only. The dataset contains diverse audio types including speech, environmental sounds, and music.


* [OpenAQA	](https://github.com/YuanGongND/ltu?tab=readme-ov-file): Data collection method - [Human]; Labeling method - [Synthetic]
* [Laion630K	](https://github.com/LAION-AI/audio-dataset/blob/main/laion-audio-630k/README.md)
* [LP-MusicCaps	](https://github.com/seungheondoh/lp-music-caps)
* [SoundDescs  	](https://github.com/akoepke/audio-retrieval-benchmark)
* [WavCaps](https://github.com/XinhaoMei/WavCaps)
* [AudioSet    	](https://research.google.com/audioset/download.html)
* [AudioSet Strong Labeled	](https://research.google.com/audioset/download_strong.html)
* [WavText5K   	](https://github.com/microsoft/WavText5K)
* [MSP-Podcast 	](https://ecs.utdallas.edu/research/researchlabs/msp-lab/MSP-Podcast.html)
* [ClothoAQA   	](https://zenodo.org/records/6473207)
* [Clotho-v2   	](https://github.com/audio-captioning/clotho-dataset/tree/master)
* [MACS        	](https://zenodo.org/records/5114771)
* [FSD50k      	](https://zenodo.org/records/4060432)
* [CochlScene  	](https://github.com/cochlearai/cochlscene)
* [NonSpeech 7k	](https://zenodo.org/records/6967442)
* [Chime-home  	](https://code.soundsoftware.ac.uk/projects/chime-home-dataset-annotation-and-baseline-evaluation-code)
* [Sonyc-UST   	](https://zenodo.org/records/3966543)
* [Emov-DB     	](https://github.com/numediart/EmoV-DB)
* [JL-Corpus   	](https://github.com/tli725/JL-Corpus)
* [Tess        	](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
* [OMGEmotion  	](https://github.com/knowledgetechnologyuhh/OMGEmotionChallenge)
* [MELD        	](https://github.com/declare-lab/MELD)
* [MusicAVQA   	](https://gewu-lab.github.io/MUSIC-AVQA/)
* [MusicQA     	](https://github.com/shansongliu/MU-LLaMA?tab=readme-ov-file)
* [MusicCaps   	](https://www.kaggle.com/datasets/googleai/musiccaps)
* [NSynth      	](https://magenta.tensorflow.org/datasets/nsynth)
* [MTG-Jamendo 	](https://github.com/MTG/mtg-jamendo-dataset)
* [MusDB-HQ    	](https://zenodo.org/records/3338373)
* [FMA         	](https://github.com/mdeff/fma)

For all of these datasets, the data collection method is [human]. For OpenAQA, Laion630k, LP-MusicCaps, WavCaps, MusicQA, the data labeling method is [synthetic]. For the rest, the data labeling method is [human].

### Evaluating Dataset:
Audio Flamingo is evaluated on the test split of the following datasets.

* [ClothoAQA   	](https://zenodo.org/records/6473207)
* [MusicAVQA   	](https://gewu-lab.github.io/MUSIC-AVQA/)
* [Clotho-v2   	](https://github.com/audio-captioning/clotho-dataset/tree/master)
* [FSD50k      	](https://zenodo.org/records/4060432)
* [CochlScene  	](https://github.com/cochlearai/cochlscene)
* [NonSpeech 7k	](https://zenodo.org/records/6967442)
* [NSynth      	](https://magenta.tensorflow.org/datasets/nsynth)
* [AudioCaps   	](https://github.com/cdjkim/audiocaps)
* [CREMA-D     	](https://github.com/CheyneyComputerScience/CREMA-D)
* [Ravdess     	](https://zenodo.org/records/1188976)
* [US8K        	](https://urbansounddataset.weebly.com/urbansound8k.html)
* [GTZAN       	](https://www.tensorflow.org/datasets/catalog/gtzan)
* [Medley-solos-DB	](https://zenodo.org/records/3464194)

For all of these datasets, the data collection method is [human] and the data labeling method is [human].

## Inference

**Engine:** HuggingFace Transformers <br>
**Test Hardware [Name the specific test hardware model]:** A100 80GB <br>