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

EMOTION_MAP_DICT = {
    'amused':       'amused'      , 
    'anger':        'angry'       , 'angry':        'angry'       , 
    'anxious':      'anxious'     , 
    'apologetic':   'apologetic'  , 
    'assertive':    'assertive'   ,
    'calm':         'calm'        , 
    'concerned':    'concerned'   , 
    'contempt':     'contempt'    , 
    'disgust':      'disgusted'   , 'disgusted':    'disgusted'   , 
    'encouraging':  'encouraging' , 
    'excited':      'excited'     , 
    'fear':         'fearful'     , 'fearful':      'fearful'     , 
    'frustated':    'frustated'   ,
    'happy':        'happy'       , 'joy':          'happy'       , 
    'neutral':      'neutral'     , 
    'sad':          'sad'         , 'sadness':      'sad'         , 
    'sleepy':       'sleepy'      , 
    'surprise':     'surprised'   , 'surprised':    'surprised'   ,
    'pleasantly surprised': 'pleasantly surprised' ,
}


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


class Augmentation:
    def __init__(
        self, 
        split: str, 
        dataset_blending_config: dict, 
        valid_dataset_config: dict = {}, 
        valid_dataset_name: str = ''
    ):
        self.split = split 
        self.dataset_blending_config = dataset_blending_config
        self.valid_dataset_config = valid_dataset_config
        self.valid_dataset_name = valid_dataset_name
    
    @staticmethod
    def either_startswith(task, candidates):
        for cand in candidates:
            if task.startswith(cand):
                return True 
        return False

    def augment(self, name, prompt, output, task, no_options=False):
        suffix='\nAnswer:'

        if self.split == 'train':
            augmentations = self.dataset_blending_config[task]["augmentations"]
            augmentation_fn = np.random.choice(list(augmentations.keys()), p=list(augmentations.values()))
            if 'prefix_prob' in self.dataset_blending_config[task].keys():
                prefix_prob = float(self.dataset_blending_config[task]['prefix_prob'])
            else:
                prefix_prob = 0.0
            
        else: 
            if self.valid_dataset_config[self.valid_dataset_name] is True:
                task = task.replace("/val", "/train")
                task = task.replace("/test", "/train")
                task = task.replace("-val", "-train")
                task = task.replace("-test", "-train")

                augmentations = self.dataset_blending_config[task]["augmentations"]
                prefix_prob = 1.0
                if 'provide_all_labels' in augmentations:
                    augmentations = {'provide_all_labels': 1.0}

            else:
                augmentations = self.valid_dataset_config[self.valid_dataset_name]["augmentations"]
                if 'prefix_prob' in self.valid_dataset_config[self.valid_dataset_name].keys():
                    prefix_prob = float(self.valid_dataset_config[self.valid_dataset_name]['prefix_prob'])
                else:
                    prefix_prob = 1.0

            self.seed = sum(list(map(ord, name + prompt + output + task)))
            local_random = np.random.default_rng(self.seed)
            augmentation_fn = local_random.choice(list(augmentations.keys()), p=list(augmentations.values()))

        if no_options and augmentation_fn == "provide_all_labels":
            augmentation_fn = "default"

        prefix = self.add_prefix(task, prefix_prob)

        output = output.strip()
        if augmentation_fn == "do_nothing":
            return (prefix, prompt + suffix, output)

        augmented = False 

        # AQA augmentation
        if self.either_startswith(task, ["Clotho-AQA-AQA", "Music-AVQA", "MU-LLAMA"]):
            if augmentation_fn == "AQA_binary_instruction":
                prompt, output, augmented = self.augment_binary_AQA(prompt, output)
            else:
                prompt, output, augmented = self.augment_AQA(prompt, output)

        # Audio Captioning augmentation
        if "AudioCaptioning" in task:
            if augmentation_fn == "AC_short":
                prompt, output, augmented = self.augment_AC_short(prompt, output)
            elif augmentation_fn == "AC_long":
                prompt, output, augmented = self.augment_AC_long(prompt, output)
            elif augmentation_fn == "AC_paragraph":
                prompt, output, augmented = self.augment_AC_paragraph(prompt, output)
            
            if "MusicCaps" in task:
                prompt = prompt.replace('sound', 'music')

        # Classification-single augmentation

        if self.either_startswith(task, [
            "CREMA-D", "emov-db", "jl-corpus", 
            "MELD", "MSP-PODCAST-Publish-1.9", 
            "ravdess", "tess", "OMGEmotion",
            "Medley-solos-DB", "musdbhq",
            "NSynth-InstrClassification",
            "NSynth-SourceClassification",
            "UrbanSound8K", "CochlScene", "NonSpeech7k",
        ]) or "GenreClassification" in task:
            if augmentation_fn in ["default", "provide_all_labels"]:
                prompt, output, augmented = self.augment_single_classification(prompt, output, task)
            if augmentation_fn == "provide_all_labels":
                prompt, output, augmented = self.augment_provide_all_labels(prompt, output, task)

        # Classification-multiple augmentation
        if self.either_startswith(task, [
            "WavText5K-Tagging", "Clotho-AQA-EventClassification",
            "AudioSet-EventClassification", "AudioSetFull",
            "chime-home", "SONYC-UST"
        ]):
            prompt = "use a few labels to describe this sound."
            if output.endswith('.'):
                output = output[:-1]
            
            if augmentation_fn == "num_words":
                prompt, output, augmented = self.augment_num_words(prompt, output)
            
            if no_options and self.either_startswith(task, ["chime-home", "SONYC-UST"]):
                augmented = True

            if not no_options:
                all_classes = []

                if task.startswith("chime-home"):
                    all_classes = [
                        'child speaking', 'male speaking', 'female speaking', 
                        'human activity', 'television', 'household appliances', 'silence',
                    ]
                
                elif task.startswith("SONYC-UST"):
                    all_classes = [
                        'small sounding engine', 'medium sounding engine', 'large sounding engine',
                        'rock drill', 'jackhammer', 'hoe ram', 'pile driver', 
                        'non machinery impact', 
                        'chainsaw', 'small medium rotating saw', 'large rotating saw', 
                        'car horn', 'car alarm', 'siren', 'reverse beeper',
                        'stationary music', 'mobile music', 'ice cream truck',
                        'person or small group talking', 'person or small group shouting', 'large crowd', 'amplified speech',
                        'dog barking whining'
                    ]
                
                if len(all_classes) > 0:
                    prompt = prompt + ' All labels are: ' + ', '.join(all_classes) + '.'
                augmented = True
        
        if task.startswith("FSD50k"):
            if augmentation_fn == 'default':
                prompt = "describe this sound in the order from specific to general."
                augmented = True

        if task.startswith("NSynth-QualityClassification"):
            if augmentation_fn == 'default':
                prompt = "what are the qualities of the music?"
                augmented = True
        
        if task.startswith("mtg-jamendo-MusicTagging"):
            if augmentation_fn == 'default':
                prompt = "what are the genres, instruments, and mood/theme of the music?"
                output = output.replace(';', '.')
                augmented = True
        
        if not augmented:
            print("This sample should be augmented but it is not: prompt is <{}>, output is <{}>".format(
                prompt, output
            ))
        
        return (prefix, prompt + suffix, output)

    def add_prefix(self, task, prefix_prob):
        if np.random.rand() > prefix_prob:
            return ''

        if 'interleaved' in task:
            for mode in ['random', 'knn']:
                task = task.replace('interleaved_{}-'.format(mode), '')

        DATASET = '-'.join(task.split('/')[0].split('-')[:-1]).lower()
        if DATASET.startswith('audiocaps'): DATASET = 'audiocaps'
        elif DATASET.startswith('audioset'): DATASET = 'audioset'
        elif DATASET.startswith('msp-podcast'): DATASET = 'msp-podcast'
        elif DATASET.startswith('wavcaps'): DATASET = 'wavcaps'
        elif DATASET.startswith('lp-musiccaps'): DATASET = 'lp-musiccaps'
        DATASET = DATASET.replace('_', ' ').replace('-', ' ')

        
        TASK = task.split('/')[0].split('-')[-1]
        TASK = ''.join([' ' + c if c.isupper() else c for c in TASK]).lstrip().lower()
        if ('musiccaps' in DATASET) and TASK == 'audio captioning':
            TASK = 'music captioning'
        if TASK.startswith('a q a'): TASK = 'audio question answering'
        elif TASK.startswith('a v q a'): TASK = 'audio vision question answering'
        elif TASK.startswith('instr classification'): TASK = 'instrument classification'
        elif TASK.startswith('m i r'): TASK = 'music information retrieval'
        
        prefix = f'The task is {TASK}. '
        return prefix

    # AQA

    def augment_AQA(self, prompt, output):
        selected = "Please answer this question:".lower()
        if prompt.startswith('question:'):
            prompt = prompt.replace('question:', selected)

        return prompt, output, True

    def augment_binary_AQA(self, prompt, output):
        if all([x.strip().lower() in ["yes", "no"] for x in output.split(',')]):
            selected = "Please answer this question:".lower()
            prompt = prompt.replace('question:', selected)
            prompt = prompt + '\nOPTIONS:\nyes.\nno.'

            return prompt, output, True
        
        else:
            return self.augment_AQA(prompt, output)

    # Audio Captioning

    def augment_AC_short(self, prompt, output):
        prompt = "Describe the sound in a sentence.".lower()
        return prompt, output, True
    
    def augment_AC_long(self, prompt, output):
        prompt = "Describe the sound at length.".lower()
        return prompt, output, True
    
    def augment_AC_paragraph(self, prompt, output):
        prompt = "Describe the sound in a paragraph.".lower()
        output = output.strip()
        last_dot = output.rfind('.')
        output = output.replace('.', ';')
        output = output[:last_dot] + '.' + output[last_dot+1:]

        return prompt, output, True
    
    # Classification-single

    def augment_single_classification(self, prompt, output, task):
        # Emotion classification
        if self.either_startswith(task, [
            "CREMA-D", "emov-db", "jl-corpus", "MELD-EmotionClassification", 
            "MSP-PODCAST-Publish-1.9", "ravdess", "tess", "OMGEmotion",
        ]):
            prompt = "what is the emotion of this speech?"
        
        if task.startswith("MELD-SentimentClassification"):
            prompt = "what is the sentiment of this speech?"

        # Music classification
        if "InstrClassification" in task:
            prompt = "what is the instrument of this music?"
        
        if "SourceClassification" in task:
            prompt = "what is the source of this music?"

        if "GenreClassification" in task:
            prompt = "what is the genre of this music?"

        # Sound classification
        if self.either_startswith(task, ["UrbanSound8K", "CochlScene"]):
            prompt = "classify this sound."

        return prompt, output, True

    def augment_provide_all_labels(self, prompt, output, task):
        all_classes = []

        # Emotion classification
        if task.startswith("CREMA-D"):
            all_classes = ['sad', 'fearful', 'neutral', 'disgusted', 'angry', 'happy']
        
        elif task.startswith("emov-db"):
            all_classes = ['amused', 'sleepy', 'neutral', 'disgusted', 'angry']
        
        elif task.startswith("jl-corpus"):
            all_classes = ['sad', 'apologetic', 'neutral', 'concerned', 'excited', 'anxious', 'encouraging', 'angry', 'assertive', 'happy']
        
        elif task.startswith("MELD"):
            if "EmotionClassification" in task:
                all_classes = ['neutral', 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust']
            
            elif "SentimentClassification" in task:
                all_classes = ['positive', 'neutral', 'negative']
        
        elif task.startswith("MSP-PODCAST-Publish-1.9"):
            all_classes = ['angry', 'sad', 'happy', 'surprise', 'fear', 'disgust', 'contempt', 'neutral']
        
        elif task.startswith("ravdess"):
            all_classes = ['sad', 'fearful', 'calm', 'neutral', 'disgusted', 'angry', 'happy', 'surprised']
        
        elif task.startswith("tess"):
            all_classes = ['pleasantly surprised', 'sad', 'fearful', 'neutral', 'disgusted', 'angry', 'happy']
        
        elif task.startswith("OMGEmotion"):
            all_classes = ['fearful', 'angry', 'disgusted', 'neutral', 'sad', 'happy', 'surprised']
        
        if "EmotionClassification" in task:
            all_classes = [EMOTION_MAP_DICT[cl] for cl in all_classes]

        # Music classification
        if task.startswith("NSynth"):
            if task.startswith("NSynth-InstrClassification"):
                all_classes = ['organ', 'mallet', 'brass', 'vocal', 'keyboard', 'reed', 'flute', 'guitar', 'bass', 'synth_lead', 'string']
            
            elif task.startswith("NSynth-SourceClassification"):
                all_classes = ['synthetic', 'electronic', 'acoustic']

        elif task.startswith("musdbhq"):
            all_classes = ['mixture', 'other', 'vocals', 'bass', 'drums']
        
        elif task.startswith("Medley-solos-DB"):
            all_classes = ['clarinet', 'flute', 'distorted electric guitar', 'trumpet', 'violin', 'piano', 'female singer', 'tenor saxophone']
        
        elif task.startswith("GTZAN"):
            all_classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

        # Sound classification
        elif task.startswith("UrbanSound8K"):
            all_classes = ['air conditioner', 'car horn', 'children playing', 'dog bark', 'drilling', 'engine idling', 'gun shot', 'jackhammer', 'siren', 'street music']
        
        elif task.startswith("CochlScene"):
            all_classes = ['Bus', 'Cafe', 'Car', 'CrowdedIndoor', 'Elevator', 'Kitchen', 'Park', 'ResidentialArea', 'Restaurant', 'Restroom', 'Street', 'Subway', 'SubwayStation']
            all_classes = [x.lower() for x in all_classes]
        
        elif task.startswith("NonSpeech7k"):
            all_classes = ['cough', 'breath', 'screaming', 'laugh', 'sneeze', 'yawn', 'crying']

        if len(all_classes) == 0:
            return prompt, output, True

        all_classes.sort()
        prompt = prompt + "\nOPTIONS:\n - {}.".format('.\n - '.join(all_classes))
        return prompt, output, True

    # Classification-multiple

    def augment_num_words(self, prompt, output):
        classes = output.split(', ')
        prompt = "Describe the sound in {} {}.".format(
            len(classes),
            'label' if len(classes) == 1 else 'labels'
        )
        if self.split == 'train':
            np.random.shuffle(classes)
        else:
            local_random = np.random.default_rng(self.seed)
            local_random.shuffle(classes)
        output = ', '.join(classes)
        return prompt, output, True
    
    def augment_alphabetic(self, prompt, output):
        prompt = "describe this sound in the alphabetic order."
        output = ', '.join(sorted(output.split(', ')))
        return prompt, output, True


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

        assert self.split in ['train', 'val', 'test']

        if self.split == 'train':
            self.data = self.blend_dataset(dataset_blending_config, dataset_blending_output)
            self.augmentor = Augmentation(split, dataset_blending_config)

        elif self.split in ['val', 'test']:
            self.valid_data = self.validation_dataset(valid_dataset_config, valid_dataset_name)
            self.augmentor = Augmentation(split, dataset_blending_config, valid_dataset_config, valid_dataset_name)
        
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
                    rel_path, contents["data"][idx]['name']
                )
            return contents
        
        """
        for interleaved data
        formatted_contents['data'] = {idx: {
                'name': list of rel_path/name for [fewshot_1, ..., fewshot_n, this], 
                'prompt': list of prompt for [fewshot_1, ..., fewshot_n, this], 
                'output': list of output for [fewshot_1, ..., fewshot_n, this], 
                [optional] 'audio_start': list of output for [fewshot_1, ..., fewshot_n, this],
                'task': task
            }}
        """

        if 'interleaved' in dataset_file:
            split_filename = dataset_file.replace('interleaved_random-', '').replace('interleaved_knn-', '')
            with open(split_filename) as f:
                contents_split = f.read()
            contents_split = json.loads(contents_split)
            
            train_filename = split_filename.replace('test.json', 'train.json').replace('val.json', 'train.json')
            with open(train_filename) as f:
                contents_train = f.read()
            contents_train = json.loads(contents_train)
            
            formatted_contents = {
                "dataset_path": contents["dataset_path"],
                "split": contents["split"],
                "split_path": contents["split_path"],
                "flamingo_task": contents["flamingo_task"],
                "total_num": contents["total_num"],
                "data": {},
            }

            for idx in contents["interleaved_data"]:
                formatted_contents["data"][idx] = {
                    'task': contents["flamingo_task"],
                    'name': [],
                    'prompt': [],
                    'output': [],
                    'audio_start': [],
                    "interleaved": None
                }

                for fewshot_idx in contents["interleaved_data"][idx]["fewshot_indices_in_train"]:
                    train_rel_path = contents_train["dataset_path"][len(self.data_root):]
                    if train_rel_path.startswith('/'):
                        train_rel_path = train_rel_path[1:]
                    if contents_train['split_path'] is not None:
                        train_rel_path = os.path.join(train_rel_path, contents_train['split_path'])

                    _contents_train_idx = contents_train["data"][str(fewshot_idx)]
                    formatted_contents["data"][idx]['name'].append(
                        os.path.join(train_rel_path, _contents_train_idx['name'])
                    )
                    formatted_contents["data"][idx]['prompt'].append(
                        _contents_train_idx['prompt']
                    )
                    formatted_contents["data"][idx]['output'].append(
                        _contents_train_idx['output']
                    )
                    formatted_contents["data"][idx]['audio_start'].append(
                        0.0 if 'audio_start' not in _contents_train_idx else _contents_train_idx['audio_start']
                    )
                
                split_idx = contents["interleaved_data"][idx]["generation_index_in_split"]
                
                split_rel_path = contents_split["dataset_path"][len(self.data_root):]
                if split_rel_path.startswith('/'):
                    split_rel_path = split_rel_path[1:]
                if contents_split['split_path'] is not None:
                    split_rel_path = os.path.join(split_rel_path, contents_split['split_path'])

                _contents_split_idx = contents_split["data"][str(split_idx)]
                formatted_contents["data"][idx]['name'].append(
                    os.path.join(split_rel_path, _contents_split_idx['name'])
                )
                formatted_contents["data"][idx]['prompt'].append(
                    _contents_split_idx['prompt']
                )
                formatted_contents["data"][idx]['output'].append(
                    _contents_split_idx['output']
                )
                formatted_contents["data"][idx]['audio_start'].append(
                    0.0 if 'audio_start' not in _contents_split_idx else _contents_split_idx['audio_start']
                )

                if 'interleaved_knn' in dataset_file.split('/')[-1]:
                    formatted_contents["data"][idx]['interleaved'] = 'similar'
                elif 'interleaved_random' in dataset_file.split('/')[-1]:
                    formatted_contents["data"][idx]['interleaved'] = 'random'

            return formatted_contents
    
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
                        audiopath = os.path.join(self.data_root, item['name'])
                        if self.is_broken_file(audiopath):
                            print('cannot read {}'.format(audiopath))
                            found_broken = True 

                    elif type(item['name']) is list:
                        for each_name in item['name']:
                            audiopath = os.path.join(self.data_root, each_name)
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

    def __getitem__(self, i):
        if self.split == 'train':
            try:
                item = self.data['data'][str(i)]
            except:
                item = self.data['data'][i]

            if type(item['name']) is str:
                audio_files = [os.path.join(self.data_root, item['name'])]
                audio_starts = [0 if 'audio_start' not in item else float(item['audio_start'])]
            else:
                audio_files = [os.path.join(self.data_root, name) for name in item['name']]
                audio_starts = ([0] * len(audio_files)) if ('audio_start' not in item) else item['audio_start']
            
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

            if type(item['name']) is str:
                text_prompt = item['prompt'].lower()
                text_output = item['output'].lower()
                prefix, text_prompt, text_output = self.augmentor.augment(
                    item['name'], text_prompt, text_output, 
                    task='{}/{}'.format(item['task'], self.split)
                )
                sample = f"{self.tokenizer.bos_token}{prefix}<audio>{text_prompt.strip()}{self.tokenizer.sep_token}{text_output.strip()}<|endofchunk|>{self.tokenizer.eos_token}"

            else:
                text_prompt = [x.lower() for x in item['prompt']]
                text_output = [x.lower() for x in item['output']]

                sample = f"{self.tokenizer.bos_token}Here are {item['interleaved']} examples. "

                for i in range(len(audio_files)):
                    no_options = (i < len(audio_files) - 1)
                    prefix, text_prompt_i, text_output_i = self.augmentor.augment(
                        item['name'][i], text_prompt[i], text_output[i], 
                        task='{}/{}'.format(item['task'], self.split),
                        no_options=no_options,
                    )
                    if i == 0:
                        sample = sample + prefix

                    sample = sample + f"<audio>{text_prompt_i.strip()}{self.tokenizer.sep_token}{text_output_i.strip()}<|endofchunk|>"
                
                sample = sample + f"{self.tokenizer.eos_token}"

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

            if type(item['name']) is str:
                text_prompt = item['prompt'].lower()
                text_output = item['output'].lower()
                prefix, text_prompt, text_output = self.augmentor.augment(
                    item['name'], text_prompt, text_output, 
                    task='{}/{}'.format(item['task'], self.split)
                )
                sample = "{}{}<audio>{}{}{}<|endofchunk|>{}".format(
                    self.tokenizer.bos_token,
                    self.preprocess_string_for_eval(prefix),
                    self.preprocess_string_for_eval(text_prompt),
                    self.tokenizer.sep_token,
                    self.preprocess_string_for_eval(text_output),
                    self.tokenizer.eos_token
                )

            else:
                text_prompt = [x.lower() for x in item['prompt']]
                text_output = [x.lower() for x in item['output']]

                sample = f"{self.tokenizer.bos_token}Here are {item['interleaved']} examples. "

                for i in range(len(audio_files)):
                    no_options = (i < len(audio_files) - 1)
                    prefix, text_prompt_i, text_output_i = self.augmentor.augment(
                        item['name'][i], text_prompt[i], text_output[i], 
                        task='{}/{}'.format(item['task'], self.split),
                        no_options=no_options,
                    )
                    if i == 0:
                        sample = sample + prefix
                    
                    sample = sample + "<audio>{}{}{}<|endofchunk|>".format(
                        self.preprocess_string_for_eval(text_prompt_i),
                        self.tokenizer.sep_token,
                        self.preprocess_string_for_eval(text_output_i)
                    )
                
                sample = sample + f"{self.tokenizer.eos_token}"
            
            text = self.tokenizer(
                sample,
                max_length=self.max_tokens*5,
                padding="longest",
                truncation="only_first",
                return_tensors="pt"
            )

        return (item['name'], audio_clips, audio_embed_mask, text["input_ids"], text["attention_mask"])

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

    data_collator = DataCollator(text_tokenizer)
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
    
