# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/mlfoundations/open_flamingo under the MIT license.
#   LICENSE is in incl_licenses directory.

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


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        filenames, audio_clips, audio_embed_mask, input_ids, attention_masks = zip(*batch)

        audio_clips = torch.cat([x.unsqueeze(0) for x in audio_clips], dim=0)
        audio_embed_mask = torch.cat([x.unsqueeze(0) for x in audio_embed_mask], dim=0)

        max_length = max([ids.shape[1] for ids in input_ids])

        padded_input_ids = []
        padded_attention_masks = []
        for ids, mask in zip(input_ids, attention_masks):
            if ids.shape[1] < max_length:
                padded_input_ids.append(
                    torch.cat([ids, torch.LongTensor([self.tokenizer.pad_token_id] * (max_length - ids.shape[1])).unsqueeze(0)], dim=1)
                )
                padded_attention_masks.append(
                    torch.cat([mask, torch.LongTensor([0] * (max_length - mask.shape[1])).unsqueeze(0)], dim=1)
                )
            else:
                padded_input_ids.append(ids)
                padded_attention_masks.append(mask)
        
        padded_input_ids = torch.cat(padded_input_ids, dim=0)
        padded_attention_masks = torch.cat(padded_attention_masks, dim=0).bool()
        
        out_dict = dict(
            filenames=filenames,
            audio_clips=audio_clips,
            audio_embed_mask=audio_embed_mask,
            input_ids=padded_input_ids,
            attention_mask=padded_attention_masks
        )
        return out_dict


class AudioTextData(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_file_root: str,
        data_root: str,
        clap_config: dict,
        dataset_blending_global_weight: float,
        dataset_blending_config: dict,
        dataset_blending_output: str,
        tokenizer,
        max_tokens: int,
        split: str = 'train',
        epoch: int = 0,
        force_reblend: bool = False,
        **kwargs
    ):
        self.dataset_file_root = dataset_file_root
        self.data_root = data_root
        self.clap_config = clap_config
        self.dataset_blending_global_weight = dataset_blending_global_weight
        self.dataset_blending_config = dataset_blending_config
        
        self.split = split
        self.epoch = epoch
        self.force_reblend = force_reblend

        assert self.split == 'train'
        self.data = self.blend_dataset(dataset_blending_config, dataset_blending_output)
        
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.max_tokens = max_tokens

    @staticmethod
    def shuffle_dict_fixed_rand(dic, seed=0):
        print('randomly shuffling key-value pairs')
        
        local_random = np.random.default_rng(seed)
        original_keys = list(dic.keys())
        shuffled_keys = deepcopy(original_keys)
        local_random.shuffle(shuffled_keys)
        shuffling_mapping = {x: y for (x, y) in zip(original_keys, shuffled_keys)}

        shuffled_dic = {}
        for idx in original_keys:
            shuffled_idx = shuffling_mapping[idx]
            shuffled_dic[idx] = dic[shuffled_idx]
        return shuffled_dic

    @staticmethod
    def is_broken_file(audiopath):
        # write your broken file paths here
        BROKEN_FILES = []
        return audiopath in BROKEN_FILES

    def _read_dataset_file(self, dataset_file):
        print("reading", dataset_file)
        with open(dataset_file) as f:
            contents = f.read()
        contents = json.loads(contents)

        assert contents["dataset_path"].startswith(self.data_root)
        rel_path = contents["dataset_path"][len(self.data_root):]
        if rel_path.startswith('/'):
            rel_path = rel_path[1:]
        if contents['split_path'] is not None:
            rel_path = os.path.join(rel_path, contents['split_path'])

        """
        contents["data"] = {
            "0": {'name': name (xxx.wav), 'dialogue': [
                    {"user": question 1, "assistant": answer 1}, 
                    ...
                    {"user": question k, "assistant": answer k}
                ]
            },
            "1": {'name': name (xxx.wav), 'dialogue': [
                    {"user": question 1, "assistant": answer 1}, 
                    ...
                    {"user": question k, "assistant": answer k}
                ]
            },
            ...
            "total_num-1": {'name': name (xxx.wav), 'dialogue': [
                    {"user": question 1, "assistant": answer 1}, 
                    ...
                    {"user": question k, "assistant": answer k}
                ]
            }
        }
        """

        for idx in contents["data"]:
            contents["data"][idx]['task'] = contents["flamingo_task"]
            contents["data"][idx]['name'] = os.path.join(
                rel_path, contents["data"][idx]['name']
            )
        return contents

    def blend_dataset(self, dataset_blending_config, dataset_blending_output):
        if os.path.exists(dataset_blending_output) and not self.force_reblend:
            print("loading blended dataset file from:", dataset_blending_output)
            with open(dataset_blending_output) as f:
                contents = f.read()
            self_data = json.loads(contents)
        
        else:
            if not self.force_reblend:
                print("no blended dataset file found; reading all dataset files")
            else:
                print("force reblending dataset at epoch {}; reading all dataset files".format(self.epoch))

            all_data = {}
            for dataset_name in dataset_blending_config:
                dataset_file = os.path.join(self.dataset_file_root, '{}.json'.format(dataset_name))
                contents = self._read_dataset_file(dataset_file)
                contents['data'] = self.shuffle_dict_fixed_rand(
                    contents['data'], 
                    seed=sum(list(map(ord, dataset_name)))
                )

                weight_global = float(self.dataset_blending_global_weight)
                weight_dataset = float(dataset_blending_config[dataset_name]["weight"])
                weight = weight_global * weight_dataset

                all_data[dataset_name] = {
                    "contents": contents,
                    "weight": weight
                }

            self_data = {
                "dataset_path": self.data_root,
                "split_path": None,
                "total_num": 0,
                "data": {}
            }

            for dataset_name in all_data:
                print('blending {}'.format(dataset_name))

                contents = all_data[dataset_name]["contents"]
                shuffled_contents_data = contents['data']
                weight = all_data[dataset_name]["weight"]
                assert type(weight) == float and weight > 0.0

                dataset_total_num = contents['total_num']
                start_idx = int(self.epoch * dataset_total_num * weight)
                end_idx = int((self.epoch + 1) * dataset_total_num * weight)

                for idx in range(start_idx, end_idx):
                    if idx > 0 and idx % dataset_total_num == 0:
                        print('force shuffling at new epoch {} for dataset {}'.format(idx // dataset_total_num, dataset_name))
                        shuffled_contents_data = self.shuffle_dict_fixed_rand(
                            contents['data'], 
                            seed=sum(list(map(ord, '{}-epoch-{}'.format(dataset_name, idx // dataset_total_num))))
                        )

                    key = str(idx % dataset_total_num)
                    item = shuffled_contents_data[key]

                    found_broken = False
                    assert type(item['name']) is str
                    audiopath = os.path.join(self.data_root, item['name'])
                    if self.is_broken_file(audiopath):
                        print('cannot read {}'.format(audiopath))
                        found_broken = True 

                    if found_broken:
                        continue 
                    
                    self_data['data'][self_data['total_num']] = item
                    self_data['total_num'] += 1 

            if not self.force_reblend:
                print('writing blended dataset file to:', dataset_blending_output)
                with open(dataset_blending_output, 'w') as json_file:
                    json.dump(self_data, json_file)
            else:
                print('writing reblended dataset file to:', dataset_blending_output.replace('.json', '-reblended.json'))
                with open(dataset_blending_output.replace('.json', '-reblended.json'), 'w') as json_file:
                    json.dump(self_data, json_file)

        return self_data

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

        audio_data = self.load_audio(os.path.join(self.data_root, audio_file), sr, duration, audio_start)
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

    def __getitem__(self, i):
        try:
            item = self.data['data'][str(i)]
        except:
            item = self.data['data'][i]

        assert type(item['name']) is str
        audio_files = [os.path.join(self.data_root, item['name'])]
        audio_starts = [0 if 'audio_start' not in item else float(item['audio_start'])]

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

        assert 'dialogue' in item
        dialogue = item['dialogue']
        prefix = 'The task is dialog. '
        sample = f"{self.tokenizer.bos_token}{prefix}<audio>"
        for each_round in dialogue:
            user_content, assistant_content = each_round['user'], each_round['assistant']
            sample = sample + f"user: {user_content} \nassistant: {self.tokenizer.sep_token}{assistant_content}<|endofchunk|>{self.tokenizer.eos_token}\n"

        text = self.tokenizer(
            sample,
            max_length=self.max_tokens,
            padding="longest",
            truncation="only_first",
            return_tensors="pt"
        )
        
        return (item['name'], audio_clips, audio_embed_mask, text["input_ids"], text["attention_mask"])

    def __len__(self):
        return len(list(self.data['data'].keys()))


@dataclass
class DataInfo:
    dataset: Dataset
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_audiotext_dataloader(data_config, clap_config, text_tokenizer, batch_size, split='train', epoch=0, force_reblend=False):
    assert split == 'train'

    data_collator = DataCollator(text_tokenizer)
    dataloader_shuffle = False

    trainset = AudioTextData(
        **data_config, 
        clap_config=clap_config, 
        tokenizer=text_tokenizer, 
        split=split,
        epoch=epoch,
        force_reblend=force_reblend
    )
    sampler = DistributedSampler(trainset, shuffle=True)
    trainloader = DataLoader(
        trainset, 
        sampler=sampler, 
        batch_size=batch_size, 
        shuffle=dataloader_shuffle, 
        collate_fn=data_collator, 
        num_workers=data_config["num_workers"]
    )
    return DataInfo(dataset=trainset, dataloader=trainloader, sampler=sampler)
