# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os 
import json
import csv
import yaml
from collections import defaultdict
import pickle
import glob
import math
from functools import partial
import sys
import io
import warnings
import random

import numpy as np
import torch

import librosa
from pydub import AudioSegment
import soundfile as sf

import faiss

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

try:
    from tqdm import tqdm 
except:
    tqdm = lambda x: x


def filter_file(file_path, file_list, filename):
    if file_list is not None:
        if filename not in file_list:
            print(filename, 'not exist')
            return True 
    else:
        if not os.path.exists(os.path.join(file_path, filename)):
            print(filename, 'not exist')
            return True 

    if os.path.getsize(os.path.join(file_path, filename)) < 16000:
        print(filename, 'less than 0.5 to 1 second')
        return True
    
    return False


def filter_response(response):
    filter_phrases_LLARK = [
        'metadata', 'is not provided', 'based on theprovided metadata', 
        'based on the providedbeat', 'based on the provided chord', 
        'basedon the provided information', 'based on theprovided annotations', 
        'no specific mood,there is no mention of', 
        'there is no specificmention of any', 'as an ai assistant', 
        'iam unable to', 'as an ai assistant', 'i donot', 
        'it is difficult to determine', 'it isnot possible to determine', 
        'no informationis available about the album', 'cannotdetermine', 
        'violin 1', 'violin 2', 'violin 3,viola 1', 'viola 2', 'viola 3', 'pack'
    ]

    filter_phrases_LTU = [
        'cannot determine', 'not provided', 'cannot be determined', 'sorry', 'i cannot',
        'without more information', 'enough information',
        'not possible', 'more context', 'enough', 'impossible', 'cannot be determined',
        'without additional information',
        'unclear', 'cannot', 'not clear', 'do not provide sufficient', 'does not provide',
        'difficult to determine', 'no information provided',
        "can't infer", "difficult to infer", "not specified", "no specific", "no information",
        "without additional", 'it is difficult to',
        "no indication"
    ]

    filter_phrases_ours = ["doesn't provide", "doesn't specify", "doesn't indicate", "based on"]

    for phrase in filter_phrases_LLARK + filter_phrases_LTU + filter_phrases_ours:
        if phrase in response.lower():
            return True 
    return False


# !!!Important!!! please write your own code to create dataset manifests based on your stored datasets
# The list of dataset_name and flamingo_task can be found in configs/*.yaml --> data_config --> dataset_blending_config
def prepare_files(dataset_name, dataset_path, split, flamingo_task, output_file):
    
    assert not os.path.exists(output_file)
    dataset_dic = {
        "dataset_path": dataset_path,
        "split": split,
        "split_path": None,
        "flamingo_task": "{}-{}".format(dataset_name, flamingo_task),
        "total_num": 0,
        "data": {}
    }

    """
    dataset_dic has the format
    {
        "dataset_path": YOUR_DATA_ROOT_DIR/datasets/dataset_name/,
        "split": "train" or "test",
        "split_path": ./,
        "flamingo_task": <dataset_name>-Dialog,
        "total_num": total number of samples,
        "data": a dictionary of data manifest (see below)
    }

    dataset_dic["data"] has the format
    {
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

    Note that os.path.join(dataset_path, split_path, name) is the absolute path to the audio file. 
    Note that audio files are not restricted to wav. However, mp3 is not recommended due to a different seeking mechanism. 
    """

    if dataset_name == 'dialog_AudioSetSL':
        assert flamingo_task == "Dialog"
        assert split == 'train'
        map_split = lambda split: './'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None 

        json_filename = 'dialogues_audioset_thresholded.json'
        with open(os.path.join(dataset_path, json_filename)) as f:
            data_list = f.read()
        data_list = json.loads(data_list)
        
        for data in tqdm(data_list):
            filename = data["audio_id"]
            if filter_file(file_path, file_list, filename):
                continue
                
            dialogue = data['dialogue']

            # filter bad dialog
            discard = False
            for each_round in dialogue:
                if filter_response(each_round['assistant']):
                    discard = True 
                    break

            if not discard:
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "dialogue": dialogue
                }
                dataset_dic["total_num"] += 1

    elif dataset_name == 'dialog_MusicCaps':
        assert flamingo_task == "Dialog"
        assert split == 'train'
        map_split = lambda split: './'
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = None 

        json_filename = 'dialogues_musiccaps_thresholded.json'
        with open(os.path.join(dataset_path, json_filename)) as f:
            data_list = f.read()
        data_list = json.loads(data_list)
        
        for data in tqdm(data_list):
            filename = data["audio_id"]
            if filter_file(file_path, file_list, filename):
                continue
                
            dialogue = data['dialogue']

            # filter bad dialog
            discard = False
            for each_round in dialogue:
                if filter_response(each_round['assistant']):
                    discard = True 
                    break

            if not discard:
                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "dialogue": dialogue
                }
                dataset_dic["total_num"] += 1

    with open(output_file, 'w') as json_file:
        json.dump(dataset_dic, json_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, help='dataset name')
    parser.add_argument('-f', '--flamingo_task', type=str, default='Dialog', help='flamingo task')
    args = parser.parse_args()

    global DATA_ROOT_DIR
    DATA_ROOT_DIR = "YOUR_DATA_ROOT_DIR"
    dataset_root = os.path.join(DATA_ROOT_DIR, "datasets")
    output_root = os.path.join(DATA_ROOT_DIR, "audio-flamingo-data/dataset_files")
    os.makedirs(output_root, exist_ok=True)

    dataset_name = args.dataset_name  # dialog_AudioSetSL, dialog_MusicCaps
    flamingo_task = args.flamingo_task  # Dialog

    split = 'train'
    dataset_path = os.path.join(dataset_root, dataset_name)

    output_folder = '{}-{}'.format(dataset_name, flamingo_task)
    os.makedirs(os.path.join(output_root, output_folder), exist_ok=True)

    dataset_file = os.path.join(output_root, output_folder, '{}.json'.format(split))
    if not os.path.exists(dataset_file):
        try:
            prepare_files(dataset_name, dataset_path, split, flamingo_task, dataset_file)
        except AssertionError as e:
            print('split {} not exist for {}: {}'.format(split, dataset_name, e))
            continue
    else:
        print('{} exists; exiting'.format(dataset_file))
        

