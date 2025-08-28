
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

<div align="center" style="display: flex; justify-content: center; margin-top: 10px;">
  <a href="https://arxiv.org/abs/2508.11818"><img src="https://img.shields.io/badge/arXiv-2508.11818-AD1C18" style="margin-right: 5px;"></a>
  <a href="https://research.nvidia.com/labs/adlr/AF3/"><img src="https://img.shields.io/badge/Demo page-228B22" style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/tree/soundCoT"><img src='https://img.shields.io/badge/Github-Audio Flamingo SoundCoT-9C276A' style="margin-right: 5px;"></a>
  <a href="https://github.com/NVIDIA/audio-flamingo/stargazers"><img src="https://img.shields.io/github/stars/NVIDIA/audio-flamingo.svg?style=social"></a>
</div>

<div align="center" style="display: flex; justify-content: center; margin-top: 10px; flex-wrap: wrap; gap: 5px;">
  <a href="https://huggingface.co/nvidia/audio-flamingo-2-SoundCoT">
    <img src="https://img.shields.io/badge/🤗-Audio_Flamingo_2_SoundCoT_(3B)-ED5A22.svg">
  </a>
  <a href="https://huggingface.co/datasets/nvidia/AF-Think/tree/main/afcot">
    <img src="https://img.shields.io/badge/🤗-Dataset: AF--CoT--Train-ED5A22.svg">
  </a>
  <a href="https://github.com/NVIDIA/audio-flamingo/tree/soundCoT/AF_Reasoning_Eval">
    <img src="https://img.shields.io/badge/Dataset: AF--Reasoning--Eval-ED5A22.svg">
  </a>
</div>

## AF-Reasoning-Eval

The AF-Reasoning-Eval benchmark is included in ```AF_Reasoning_Eval/```. 

- The ```AQA``` subset is derived from [Clotho-AQA](https://zenodo.org/records/6473207), and our filenames point to Clotho-AQA filenames. Note that the Clotho-AQA audio files follow their original license. 
- The ```Classification``` subset is derived from [FSD50K](https://zenodo.org/records/4060432), and our filenames point to FSD50K filenames. Note that the FSD50K audio files follow their original license. 
- Our metadata are released under ```CC-BY 4.0```.

## AF-CoT-Train

The ```AF-CoT-Train``` generation pipelines are included in ```AF_CoT_Train/```. It is recommended to use ```alg_6_AQA_subquestions.py``` to generate reasoning chains for AQA samples and ```alg_8_Classification_MCQ.py``` to generate reasoning chains for classification samples.

The dataset is released in [this link](https://huggingface.co/datasets/nvidia/AF-Think/tree/main/afcot).

## Inference code
The instruction to run inference of the CoT model is [here](https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_2/inference_HF_pretrained#steps-of-inference-of-the-cot-model).

## Sound Demos

[▶️ Sample 1](https://github.com/NVIDIA/audio-flamingo/tree/soundCoT/static/audio/UrbanHerringGulls.wav)
<div>
  <table>
    <tr>
      <td>
        <audio controls preload="none">
          <source src="static/audio/UrbanHerringGulls.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <!-- <td> -->
        <b> (Sample 1) Question: </b> Which could be the natural enemy of these animals? 
        <br>
        (A) eagle. (B) tiger. (C) octopus. (D) shark.
      <!-- </td> -->
    </tr>
    <tr>
      <td>
        <b> Output: </b> 
        <span style="color: #218C21;">
        <br> &lt;summary&gt; The question asks to identify the natural enemy of the animals in the audio. I will first determine the specific animals present, then identify their typical natural predators, and finally select the option that matches one of those predators. &lt;/summary&gt;
        </span>
        <span style="color: #21568C;">
        <br> &lt;caption&gt; A bird is singing. &lt;/caption&gt;
        </span>
        <span style="color: #8C2121;">
        <br> &lt;reasoning&gt; The audio contains bird vocalizations. Birds are typically preyed upon by various animals, including eagles. Among the options, 'eagle' (A) is the natural predator of birds, making it the correct answer. &lt;/reasoning&gt;
        </span>
        <br> &lt;conclusion&gt; (A) eagle &lt;/conclusion&gt;
      </td>
    </tr>
  </table>
</div>

<br>

[▶️ Sample 2](https://github.com/NVIDIA/audio-flamingo/tree/soundCoT/static/audio/ambient text.wav)
<div>
  <table>
    <tr>
      <td>
        <audio controls preload="none">
          <source src="static/audio/ambient text.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <!-- <td> -->
        <b> (Sample 2) Question: </b> What genre of movie could this sound be played in?
        <br>
        (A) romance. (B) horror. (C) comedy. (D) action.
      <!-- </td> -->
    </tr>
    <tr>
      <td>
        <b> Output: </b> 
        <span style="color: #218C21;">
        <br> &lt;summary&gt; The question asks to identify the movie genre based on the sound. I will first analyze the sound's characteristics (e.g., eerie, suspenseful, chaotic) and then match them to the typical soundscapes of each genre (e.g., horror's eerie tones, action's intense rhythms, comedy's lightheartedness, romance's emotional tones). &lt;/summary&gt;
        </span>
        <span style="color: #21568C;">
        <br> &lt;caption&gt; This is a dark ambient piece. &lt;/caption&gt;
        </span>
        <span style="color: #8C2121;">
        <br> &lt;reasoning&gt; The sound's eerie and suspenseful characteristics align with the typical soundscapes of horror movies. Horror films often use such sounds to create tension and fear, making it the most fitting genre. &lt;/reasoning&gt;
        </span>
        <br> &lt;conclusion&gt; (B) horror &lt;/conclusion&gt;
      </td>
    </tr>
  </table>
</div>

<br>

[▶️ Sample 3](https://github.com/NVIDIA/audio-flamingo/tree/soundCoT/static/audio/58969.wav)
<div>
  <table>
    <tr>
      <td>
        <audio controls preload="none">
          <source src="static/audio/58969.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <!-- <td> -->
        <b> (Sample 3) Question: </b> Classify the sound.
        <br>
        (A) bass guitar. (B) electric guitar. (C) acoustic guitar. (D) boiling.
      <!-- </td> -->
    </tr>
    <tr>
      <td>
        <b> Output: </b> 
        <span style="color: #218C21;">
        <br> &lt;summary&gt; The question is to classify the sound to be one of bass guitar, electric guitar, acoustic guitar, boiling. For each choice, I will describe its acoustic properties, and then decide whether the sound fits that label. &lt;/summary&gt;
        </span>
        <span style="color: #21568C;">
        <br> &lt;caption&gt; A guitar is being played. &lt;/caption&gt;
        </span>
        <span style="color: #8C2121;">
        <br> &lt;reasoning&gt; The bass guitar produces low-frequency sounds, typically ranging from the lowest a1 note (22 hz) to the highest e4 note (659 hz), contributing a rich, deep tone to the overall sound of a band or ensemble. The sound does not fit with the label 'bass guitar' as the audio features a guitar being played, not a bass guitar.<br> The electric guitar produces a sound that is characterized by its bright, clear, and resonant tone, which can be further shaped and modified by various electronic effects and amplification. The sound fits with the label 'electric guitar' as it is a musical instrument that produces sound through electric amplification.<br> Acoustic guitars produce sound through the vibration of strings, amplified by the guitar's body, resulting in a warm, resonant tone that is often described as natural and clear. The sound fits with the label 'acoustic guitar' as it is a musical instrument that produces sound through the vibration of its strings, which can be played with a pick or fingers.<br> Boiling produces a continuous, high-pitched hissing sound as water vapor bubbles rapidly rise to the surface. The sound does not fit with the label 'boiling'. &lt;/reasoning&gt;
        </span>
        <br> &lt;conclusion&gt; acoustic guitar &lt;/conclusion&gt;
      </td>
    </tr>
  </table>
</div>

<br>

[▶️ Sample 4](https://github.com/NVIDIA/audio-flamingo/tree/soundCoT/static/audio/KYGj2xHDQAg_00-00-00_00-00-17.wav)
<div>
  <table>
    <tr>
      <td>
        <audio controls preload="none">
          <source src="static/audio/KYGj2xHDQAg_00-00-00_00-00-17.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <!-- <td> -->
        <b> (Sample 4) Question: </b> What is this competition venue?
        <br>
        (A) Tennis, (B) Badminton, (C) Table Tennis, (D) Table Soccer.
      <!-- </td> -->
    </tr>
    <tr>
      <td>
        <b> Output: </b> 
        <span style="color: #218C21;">
        <br> &lt;summary&gt; The question asks to identify the competition venue based on audio cues. I will first analyze the specific sounds or cues present in the audio, then determine which venue (tennis, badminton, table tennis, table soccer) is most associated with those sounds, and finally check for any conflicting sounds that might indicate a different venue. &lt;/summary&gt;
        </span>
        <span style="color: #21568C;">
        <br> &lt;caption&gt; A ball is bouncing and people are talking. &lt;/caption&gt;
        </span>
        <span style="color: #8C2121;">
        <br> &lt;reasoning&gt; The audio contains distinct sounds of a ball bouncing and people talking. Table tennis is uniquely characterized by the distinct sound of a ball hitting a table and the rhythmic bounce, which aligns with the described audio cues. While table soccer may involve similar sounds, the specific bounce and impact noises are more characteristic of table tennis. No conflicting sounds are present to suggest alternative venues. &lt;/reasoning&gt;
        </span>
        <br> &lt;conclusion&gt; (c) table tennis &lt;/conclusion&gt;
      </td>
    </tr>
  </table>
</div>

<br>

[▶️ Sample 5](https://github.com/NVIDIA/audio-flamingo/tree/soundCoT/static/audio/NePo2M4Ckjg_00-00-07_00-00-37.wav)
<div>
  <table>
    <tr>
      <td>
        <audio controls preload="none">
          <source src="static/audio/NePo2M4Ckjg_00-00-07_00-00-37.wav" type="audio/wav">
        </audio>
      </td>
    </tr>
    <tr>
      <!-- <td> -->
        <b> (Sample 5) Question: </b> What kind of movie is this music suitable for?
        <br>
        (A) Romance, (B) Comedy, (C) Epic, (D) Mystery.
      <!-- </td> -->
    </tr>
    <tr>
      <td>
        <b> Output: </b> 
        <span style="color: #218C21;">
        <br> &lt;summary&gt; The question asks to determine the movie genre based on the music's characteristics. I will first analyze the music's tempo, instrumentation, and mood. Then, I will compare these elements to the typical traits of each genre (romance, comedy, epic, mystery) to identify the best fit. &lt;/summary&gt;
        </span>
        <span style="color: #21568C;">
        <br> &lt;caption&gt; This is an instrumental electronic piece with a slow tempo, featuring a blend of ambient and experimental sounds. The music creates a dark and mysterious atmosphere, with a sense of tension and suspense. &lt;/caption&gt;
        </span>
        <span style="color: #8C2121;">
        <br> &lt;reasoning&gt; The music's slow tempo, ambient and experimental instrumentation, and dark, suspenseful mood align with the characteristics of a mystery movie. These elements create an atmosphere of intrigue and tension, which are hallmark features of mystery films. &lt;/reasoning&gt;
        </span>
        <br> &lt;conclusion&gt; (d) mystery &lt;/conclusion&gt;
      </td>
    </tr>
  </table>
</div>


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

- Audio Flamingo Sound-CoT
```
@article{kong2025audio,
  title={Audio Flamingo Sound-CoT Technical Report: Improving Chain-of-Thought Reasoning in Sound Understanding},
  author={Kong, Zhifeng and Goel, Arushi and Santos, Joao Felipe and Ghosh, Sreyan and Valle, Rafael and Ping, Wei and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2508.11818},
  year={2025}
}
```
