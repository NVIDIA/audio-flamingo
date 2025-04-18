# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.


import os
import io 
import sys
import string
import yaml
import json
import shutil
from copy import deepcopy
from contextlib import suppress
import warnings
from tqdm import tqdm
import random

import librosa
import soundfile as sf

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, set_seed 

seed = 42
set_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# import laion_clap
import my_laion_clap.CLAP.src.laion_clap as laion_clap
# from my_laion_clap.CLAP.src.laion_clap.clap_module.htsat import create_htsat_model


class CLAPAudioCfp:
    model_type: str = "HTSAT"
    model_name: str = "afclap"
    sample_rate: int = 16000
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 160
    fmin: int = 50
    fmax: int = 14000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 160000


class CustomDataset(Dataset):
    def __init__(self, audio_files, ground_truth):
        self.audio_files = audio_files
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        return self.audio_files[idx], self.ground_truth[idx]


def custom_collate_fn(batch):
    audio_batch, ground_truth_batch = zip(*batch)
    return audio_batch, ground_truth_batch


@torch.no_grad()
def compute_clap_text_audio_sim(afclap_model, audio_file_list, all_texts):
    # compute audio embedding
    audio_embed = afclap_model.get_audio_embedding_from_filelist(audio_file_list, sr=16000, use_tensor=True)

    # compute text embedding
    text_embed = afclap_model.get_text_embedding(all_texts, use_tensor=True)

    # compute similarities
    similarities = torch.tensor(audio_embed) @ torch.tensor(text_embed).t()
    return similarities.cpu().numpy()


def load_afclap(ckpt_path):
    method = "afclap"
    audio_embed_dim = 2048

    model = laion_clap.CLAP_Module(
        enable_fusion=True, 
        amodel='HTSAT-afclap',
        tmodel='t5'
    ).cuda()

    model.load_afclap_ckpt(ckpt=ckpt_path, verbose=True)
    return model


if __name__ == "__main__":
    ckpt_path = "LOCAL_PATH_TO_CLAP_CHECKPOINT"  # location to store epoch_15.pt from https://huggingface.co/nvidia/audio-flamingo-2/tree/main/clap_ckpt
    afclap_model = load_afclap(ckpt_path)

    # ========== GTZAN example ==========
    audio_file_list = [
        "GTZAN/gtzan/data/genres/classical/classical.00000.wav",
        "GTZAN/gtzan/data/genres/blues/blues.00000.wav",
        "GTZAN/gtzan/data/genres/pop/pop.00000.wav",
        "GTZAN/gtzan/data/genres/rock/rock.00000.wav",
    ]
    all_texts = [
        "This is a classical song.", 
        "This is a blues song.", 
        "This is a pop song.", 
        "This is a rock song.",
        "This is a disco song.",
        "This is a metal song.",
    ]
    similarities = compute_clap_text_audio_sim(afclap_model, audio_file_list, all_texts)
    print(similarities)
