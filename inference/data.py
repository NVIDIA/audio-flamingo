# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import functools
import io
import json
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable the tokenizer parallelism warning
import random
import re
import string
import subprocess
import sys
import yaml

import numpy as np

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pydub import AudioSegment
from tqdm import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler


from transformers import AutoTokenizer

import librosa
import soundfile as sf


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


class AudioTextDataProcessor:
    def __init__(
        self,
        data_root: str,
        clap_config: dict,
        tokenizer,
        max_tokens: int,
        **kwargs
    ):
        self.data_root = data_root
        self.clap_config = clap_config
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.max_tokens = max_tokens

    def get_num_windows(self, T, sr):
        clap_config = self.clap_config
        window_length  = int(float(clap_config["window_length"]) * sr)
        window_overlap = int(float(clap_config["window_overlap"]) * sr)
        max_num_window = int(clap_config["max_num_window"])

        num_windows = 1
        if T <= window_length:
            num_windows = 1
            full_length = window_length
        elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
            num_windows = max_num_window
            full_length = (max_num_window * window_length - (max_num_window - 1) * window_overlap)
        else:
            num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
            full_length = num_windows * window_length - (num_windows - 1) * window_overlap
        
        return num_windows, full_length

    def load_audio(self, file_path, target_sr=44100, duration=30.0, start=0.0):
        if file_path.endswith('.mp3'):
            audio = AudioSegment.from_file(file_path)
            if len(audio) > (start + duration) * 1000:
                audio = audio[start * 1000:(start + duration) * 1000]

            if audio.frame_rate != target_sr:
                audio = audio.set_frame_rate(target_sr)

            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            data = np.array(audio.get_array_of_samples())
            if audio.sample_width == 2:
                data = data.astype(np.float32) / np.iinfo(np.int16).max
            elif audio.sample_width == 4:
                data = data.astype(np.float32) / np.iinfo(np.int32).max
            else:
                raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))

        else:
            with sf.SoundFile(file_path) as audio:
                original_sr = audio.samplerate
                channels = audio.channels

                max_frames = int((start + duration) * original_sr)

                audio.seek(int(start * original_sr))
                frames_to_read = min(max_frames, len(audio))
                data = audio.read(frames_to_read)

                if data.max() > 1 or data.min() < -1:
                    data = data / max(abs(data.max()), abs(data.min()))
            
            if original_sr != target_sr:
                if channels == 1:
                    data = librosa.resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
                else:
                    data = librosa.resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
            else:
                if channels != 1:
                    data = data.T[0]
        
        if data.min() >= 0:
            data = 2 * data / abs(data.max()) - 1.0
        else:
            data = data / max(abs(data.max()), abs(data.min()))
        
        assert len(data.shape) == 1, data.shape
        return data

    def compute_sliding_window(self, audio_file, audio_start=0.0):
        if type(audio_start) == str:
            audio_start = float(audio_start)

        clap_config = self.clap_config

        if clap_config["method"] == 'laion-clap':
            sr = 48000
        elif clap_config["method"] == 'microsoft-clap':
            sr = 44100
        else:
            raise NotImplementedError

        window_length  = int(float(clap_config["window_length"]) * sr)
        window_overlap = int(float(clap_config["window_overlap"]) * sr)
        max_num_window = int(clap_config["max_num_window"])
        duration = max_num_window * (clap_config["window_length"] - clap_config["window_overlap"]) + clap_config["window_overlap"]

        audio_data = self.load_audio(audio_file, sr, duration, audio_start)
        T = len(audio_data)
        num_windows, full_length = self.get_num_windows(T, sr)

        if full_length > T:
            audio_data = np.append(audio_data, np.zeros(full_length - T))
        audio_data = audio_data.reshape(1, -1)
        audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()

        audio_clips = []
        audio_embed_mask = torch.zeros(max_num_window)
        for i in range(num_windows):
            start = i * (window_length - window_overlap)
            audio_clips.append(audio_data_tensor[:, start:start+window_length])            
            audio_embed_mask[i] = 1

        assert sum(audio_embed_mask) == num_windows

        if num_windows < max_num_window:
            for _ in range(max_num_window - num_windows):
                audio_clips.append(torch.zeros_like(audio_clips[-1]))
        
        audio_clips = torch.cat(audio_clips)  # (max_num_window, window_length * sr) cuda tensor

        return audio_clips, audio_embed_mask

    def preprocess_string_for_eval(self, x):
        x = x.rstrip().lstrip()
        x = x.lower()
        return x

    def process(self, item):
        if type(item['name']) is str:
            audio_files = [os.path.join(self.data_root, item['name'])]
            audio_starts = [0 if 'audio_start' not in item else float(item['audio_start'])]
        else:
            audio_files = [os.path.join(self.data_root, name) for name in item['name']]
            audio_starts = [0] * len(audio_files) if 'audio_start' not in item else item['audio_start']
        
        audio_clips, audio_embed_mask = [], []
        for audio_file, audio_start in zip(audio_files, audio_starts):
            this_audio_clips, this_audio_embed_mask = self.compute_sliding_window(audio_file, audio_start)
            audio_clips.append(this_audio_clips)
            audio_embed_mask.append(this_audio_embed_mask)

        audio_clips = torch.cat(audio_clips)
        audio_embed_mask = torch.cat(audio_embed_mask)

        correct_num_windows = int(self.clap_config["max_num_window"]) * int(self.clap_config["max_num_fewshot"])
        if len(audio_clips) < correct_num_windows:
            audio_clips = torch.cat([
                audio_clips, 
                torch.zeros(correct_num_windows - len(audio_clips), audio_clips.shape[1])
            ])
            audio_embed_mask = torch.cat([
                audio_embed_mask,
                torch.zeros(correct_num_windows - len(audio_embed_mask))
            ])
    
        audio_clips.requires_grad = False
        audio_embed_mask.requires_grad = False

        assert type(item['name']) is str

        # simple data - 1 audio, 1 text
        if 'prompt' in item:
            text_prompt = item['prompt'].lower()
            prefix = item['prefix'].lower()  # the task is xxx.
            sample = "{}{} <audio>{}\nanswer:{}".format(
                self.tokenizer.bos_token,
                self.preprocess_string_for_eval(prefix),
                self.preprocess_string_for_eval(text_prompt),
                self.tokenizer.sep_token
            )
        
        # dialog data - 1 audio, multiple text
        elif 'dialogue' in item:
            dialogue = item['dialogue']
            prefix = item['prefix'].lower()  # the task is dialog.
            sample = f"{self.tokenizer.bos_token}{prefix}<audio>"
            for each_round in dialogue:
                sample = sample + f"user: {each_round['user']} \nassistant: {self.tokenizer.sep_token}"
                if 'assistant' in each_round:
                    sample = sample + f"{each_round['assistant']}<|endofchunk|>{self.tokenizer.eos_token}\n"

        text = self.tokenizer(
            sample,
            max_length=self.max_tokens*5,
            padding="longest",
            truncation="only_first",
            return_tensors="pt"
        )

        return (item['name'], audio_clips, audio_embed_mask, text["input_ids"], text["attention_mask"])
