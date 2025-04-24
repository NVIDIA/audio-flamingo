# Copyright (c) 2025 NVIDIA CORPORATION. 
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
import torch.nn.functional as F
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
    def __init__(self, tokenizer, clap_config):

        self.tokenizer = tokenizer
        self.clap_config = clap_config
        self.max_num_window = clap_config["max_num_window"]

    def __call__(self, batch):

        filenames, audio_clips, audio_embed_masks, input_ids, attention_masks = zip(*batch)

        num_windows_all = [sum(audio_embed_mask) for audio_embed_mask in audio_embed_masks]
        max_window_batch = int(max(num_windows_all))

        if max_window_batch > self.max_num_window:
            max_window_batch = self.max_num_window

        padded_audio_clips = []
        padded_audio_embed_masks = []
        for audio_clip, audio_embed_mask in zip(audio_clips,audio_embed_masks):
            this_audio_clip_clips = [clip for clip in audio_clip]
            num_windows = len(this_audio_clip_clips)
            if num_windows < max_window_batch:
                for _ in range(max_window_batch - num_windows):
                    this_audio_clip_clips.append(torch.zeros_like(this_audio_clip_clips[-1]))
                audio_clip = torch.cat(this_audio_clip_clips)
                audio_embed_mask = torch.zeros(max_window_batch)
                audio_embed_mask[:num_windows] = 1
            elif num_windows > max_window_batch:
                audio_clip = this_audio_clip_clips[:max_window_batch]
                audio_clip = torch.cat(this_audio_clip_clips)
                audio_embed_mask = audio_embed_mask[:max_window_batch]
            else:
                audio_clip = torch.cat(this_audio_clip_clips)
                
            padded_audio_clips.append(audio_clip)
            padded_audio_embed_masks.append(audio_embed_mask)

        audio_clips = torch.cat([x.unsqueeze(0) for x in padded_audio_clips], dim=0)
        audio_embed_mask = torch.cat([x.unsqueeze(0) for x in padded_audio_embed_masks], dim=0)

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
        valid_dataset_config: dict = {},
        valid_dataset_name: str = '',
        epoch: int = 0,
        force_reblend: bool = False,
        sr = 16000,
        **kwargs
    ):
        self.dataset_file_root = dataset_file_root
        self.data_root = data_root
        self.clap_config = clap_config
        self.dataset_blending_global_weight = dataset_blending_global_weight
        self.dataset_blending_config = dataset_blending_config
        self.sr = sr
        
        self.split = split
        self.epoch = epoch
        self.force_reblend = force_reblend

        assert self.split in ['train', 'val', 'test']

        if self.split == 'train':
            self.data = self.blend_dataset(dataset_blending_config, dataset_blending_output)

        elif self.split in ['val', 'test']:
            self.valid_data = self.validation_dataset(valid_dataset_config, valid_dataset_name)
        
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
        BROKEN_FILES = []
        return audiopath in BROKEN_FILES

    def _read_dataset_file(self, dataset_file):
        print("reading", dataset_file)
        with open(dataset_file) as f:
            contents = f.read()
        contents = json.loads(contents)

        if contents['split_path'] is not None:
            abs_path = contents['split_path']

        """
        for normal data
        contents['data'] = {idx: {
                'name': rel_path/name, 
                'prompt': prompt, 
                'output': output, 
                [optional] 'audio_start': audio_start,
                'task': task,
            }}
        """

        if 'interleaved' not in dataset_file:
            for idx in contents["data"]:
                contents["data"][idx]['task'] = contents["flamingo_task"]
                contents["data"][idx]['name'] = os.path.join(
                    abs_path, contents["data"][idx]['name']
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
                "data": {}  # {id: {'name': rel_path/name or [rel_path/names], 'prompt': prompt or [prompts], 'output': output or [outputs], 'task': task, 'interleaved': interleave_method}}
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
                    if type(item['name']) is str:
                        audiopath = item['name']
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

    def load_audio(self, file_path, target_sr=16000, duration=30.0, start=0.0):
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

    def compute_sliding_window(self, audio_file, audio_start=0.0, audio="sound"):
        if type(audio_start) == str:
            audio_start = float(audio_start)

        if audio == "sound":
            encoder_config = self.clap_config
        else:
            raise NotImplementedError

        if encoder_config["method"] == 'afclap-large':
            sr = 16000
        else:
            raise NotImplementedError

        window_length  = int(float(encoder_config["window_length"]) * sr)
        window_overlap = int(float(encoder_config["window_overlap"]) * sr)
        max_num_window = int(encoder_config["max_num_window"])
        duration = max_num_window * (encoder_config["window_length"] - encoder_config["window_overlap"]) + encoder_config["window_overlap"]

        audio_data = self.load_audio(os.path.join(self.data_root, audio_file), sr, duration, audio_start) # already cuts to max duration
        T = len(audio_data)
        num_windows, full_length = self.get_num_windows(T, sr)

        # pads to the nearest multiple of window_length
        if full_length > T:
            audio_data = np.append(audio_data, np.zeros(full_length - T))

        audio_data = audio_data.reshape(1, -1)
        audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()

        audio_clips = []
        audio_embed_mask = torch.ones(num_windows)
        for i in range(num_windows):
            start = i * (window_length - window_overlap)
            audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
            audio_clips.append(audio_data_tensor_this)            
        
        return audio_clips, audio_embed_mask

    def validation_dataset(self, valid_dataset_config, valid_dataset_name):
        dataset_file = os.path.join(self.dataset_file_root, '{}.json'.format(valid_dataset_name))
        contents = self._read_dataset_file(dataset_file)

        contents['data'] = self.shuffle_dict_fixed_rand(
            contents['data'], 
            seed=sum(list(map(ord, valid_dataset_name)))
        )

        return contents

    def preprocess_string_for_eval(self, x):
        x = x.rstrip().lstrip()
        x = x.lower()
        return x

    def _actual_getitem(self, i):
        if self.split == 'train':
            try:
                item = self.data['data'][str(i)]
            except:
                item = self.data['data'][i]

            if type(item['name']) is str:
                audio_file = item['name']
                audio_start = 0 if 'audio_start' not in item else float(item['audio_start'])
            else:
                raise Exception(f"The item has a {type(item['name'])}. Only single path as a string is supported")

            # compute window for long audios
            audio_clips, audio_embed_mask = self.compute_sliding_window(audio_file, audio_start, audio="sound")
        
            # make the text prompt
            text_prompt = str(item['prompt']).lower()
            text_output = str(item['output']).lower()

            sample = f"<audio>{text_prompt.strip()}{self.tokenizer.sep_token}{text_output.strip()}<|endofchunk|>{self.tokenizer.eos_token}"

            text = self.tokenizer(
                sample,
                max_length=self.max_tokens,
                padding="longest",
                truncation="only_first",
                return_tensors="pt"
            )
        
        elif self.split in ['val', 'test']:
            try:
                item = self.valid_data['data'][str(i)]
            except:
                item = self.valid_data['data'][i]

            if type(item['name']) is str:
                audio_file = os.path.join(self.data_root, item['name'])
                audio_start = 0 if 'audio_start' not in item else float(item['audio_start'])
            else:
                raise Exception(f"The item has a {type(item['name'])}. Only single path as a string is supported")

            # compute window for long audios
            audio_clips, audio_embed_mask = self.compute_sliding_window(audio_file, audio_start, audio="sound")
        
            # make the text prompt
            text_prompt = self.preprocess_string_for_eval(str(item['prompt']).lower())
            text_output = self.preprocess_string_for_eval(str(item['output']).lower())

            sample = f"<audio>{text_prompt.strip()}{self.tokenizer.sep_token}{text_output.strip()}<|endofchunk|>{self.tokenizer.eos_token}"

            text = self.tokenizer(
                sample,
                max_length=self.max_tokens,
                padding="longest",
                truncation="only_first",
                return_tensors="pt"
            )
            
        # audio_clips_clap, audio_embed_mask_clap, audio_clips_speech, audio_embed_mask_speech, audio_clips_music, audio_embed_mask_music,
        return (item['name'], audio_clips, audio_embed_mask, text["input_ids"], text["attention_mask"])

    def __getitem__(self, i):
        try: 
            return self._actual_getitem(i)
        except Exception as e:
            print('batch {} failed with reason {}'.format(i, e))
            try:
                return self._actual_getitem((i-42)%99)
            except:
                return self._actual_getitem((i-84)%99)

    def __len__(self):
        if self.split == 'train':
            return len(list(self.data['data'].keys()))

        elif self.split == 'val':
            return min(len(list(self.valid_data['data'].keys())), 64)

        elif self.split == 'test':
            return len(list(self.valid_data['data'].keys()))


@dataclass
class DataInfo:
    dataset: Dataset
    dataloader: DataLoader
    sampler: DistributedSampler = None

    def set_epoch(self, epoch):
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_audiotext_dataloader(data_config, clap_config, text_tokenizer, batch_size, split='train', epoch=0, force_reblend=False):
    assert split in ['train', 'val', 'test']

    data_collator = DataCollator(text_tokenizer, clap_config)
    dataloader_shuffle = False

    if split == 'train':
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
    
    elif split in ['val', 'test']:
        all_DataInfo = {}
        for valid_dataset_name in list(data_config["valid_dataset_config"].keys()):
            valid_dataset_name = valid_dataset_name.strip()
            validset = AudioTextData(
                **data_config, 
                clap_config=clap_config,
                tokenizer=text_tokenizer, 
                split=split, 
                valid_dataset_name=valid_dataset_name
            )
            if split == 'val':
                # distributed sampler
                all_DataInfo[valid_dataset_name] = DataInfo(
                    dataset=validset,
                    dataloader=DataLoader(
                        validset, 
                        sampler=DistributedSampler(validset, shuffle=False),
                        batch_size=batch_size, 
                        shuffle=dataloader_shuffle, 
                        collate_fn=data_collator, 
                        num_workers=data_config["num_workers"]
                ))
            else:
                # single GPU
                all_DataInfo[valid_dataset_name] = DataInfo(
                    dataset=validset,
                    dataloader=DataLoader(
                        validset, 
                        batch_size=batch_size, 
                        shuffle=dataloader_shuffle, 
                        collate_fn=data_collator, 
                        num_workers=data_config["num_workers"]
                ))

        return all_DataInfo
    

