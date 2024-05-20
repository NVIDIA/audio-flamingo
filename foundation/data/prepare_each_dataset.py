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
import laion_clap

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


def suppress_all_output(func):
    def wrapper(*args, **kwargs):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        old_fd_out = os.dup(1)
        old_fd_err = os.dup(2)
        null_fd = os.open(os.devnull, os.O_RDWR)
        
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                result = func(*args, **kwargs)
            finally:
                os.dup2(old_fd_out, 1)
                os.dup2(old_fd_err, 2)
                os.close(null_fd)
                os.close(old_fd_out)
                os.close(old_fd_err)
        
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        
        return result
    return wrapper


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
        print(filename, 'file size too small')
        return True
    
    return False


# ==================== Prepare dataset files from each data folder ====================

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

def load_dataset_file(dataset_file):
    with open(dataset_file) as f:
        contents = f.read()
    contents = json.loads(contents)

    audio_files = [
        os.path.join(
            contents["dataset_path"],
            contents["split_path"],
            contents["data"][str(i)]["name"]
        ) for i in range(contents["total_num"])
    ]

    return contents, audio_files


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
        "split": "train" or "val" or "test",
        "split_path": ./ or sub_folder for this split,
        "flamingo_task": <dataset_name>-<flamingo_task> (e.g. audiocaps-AudioCaptioning, Clotho-AQA-AQA),
        "total_num": total number of samples,
        "data": a dictionary of data manifest (see below)
    }

    dataset_dic["data"] has the format
    {
        "0": {'name': name (xxx.wav), 'prompt': prompt, 'output': output},
        "1": {'name': name (xxx.wav), 'prompt': prompt, 'output': output},
        ...
        "total_num-1": {'name': name (xxx.wav), 'prompt': prompt, 'output': output},
    }

    Note that os.path.join(dataset_path, split_path, name) is the absolute path to the audio file. 
    Note that audio files are not restricted to wav. However, mp3 is not recommended due to a different seeking mechanism. 


    Prompts are selected from the following based on the task of each dataset:

        "generate audio caption"
        "generate audio description"
        "describe the sound in a sentence."
        "describe the sound at length."
        "describe the sound in a paragraph."

        "question: <question>"
        "<question>"
        "please answer this question: <question>"

        "classify this sound."
        "generate tags"
        "use a few labels to describe this sound."
        "describe this sound in the order from specific to general."
        "this acoustic scene is"
        "what is the emotion of this speech?"
        "what is the sentiment of this speech?"

        "describe the music in a sentence."
        "describe the music at length."
        "describe the music in a paragraph."

        "what is the instrument of this music?"
        "what is the source of this music?"
        "what are the qualities of the music?"
        "this music note is produced by"
        "what is the genre of this music?"
        "what are the genres, instruments, and mood/theme of the music?"

    """

    # below is an example code to dump audiocaps manifest from its original metadata
    if dataset_name == "audiocaps":
        assert flamingo_task == 'AudioCaptioning'
        assert split in ["train", "test", "val"]

        map_split = lambda split: 'audio/{}'.format(split if split in ['train', 'test'] else 'valid')
        file_path = os.path.join(
            dataset_path,
            map_split(split)
        )
        assert os.path.exists(file_path), '{} not exist'.format(file_path)

        dataset_dic["split_path"] = map_split(split)
        file_list = list(filter(lambda x: x.endswith('.flac'), os.listdir(file_path)))

        for filename in tqdm(file_list):
            if filter_file(file_path, file_list, filename):
                continue
            
            with open(os.path.join(file_path, filename.replace('.flac', '.json')), 'r') as f:
                data = json.load(f)
            
            for text_output in data['text']:
                if len(text_output) <= 1:
                    continue

                dataset_dic["data"][dataset_dic["total_num"]] = {
                    "name": filename,
                    "prompt": "describe the sound in a sentence.",
                    "output": text_output.replace('\n', ' ')
                }
                dataset_dic["total_num"] += 1

    with open(output_file, 'w') as json_file:
        json.dump(dataset_dic, json_file)


# ==================== Precompute CLAP and build Hashing ====================

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


@suppress_all_output
def load_clap_model(checkpoint):
    if checkpoint in ['630k-audioset-best.pt', '630k-best.pt', '630k-audioset-fusion-best.pt', '630k-fusion-best.pt']:
        amodel = 'HTSAT-tiny'
    elif checkpoint in ['music_speech_audioset_epoch_15_esc_89.98.pt']:
        amodel = 'HTSAT-base'
    else:
        raise NotImplementedError
    
    model = laion_clap.CLAP_Module(
        enable_fusion=('fusion' in checkpoint.lower()), 
        amodel=amodel
    ).cuda()
    model.load_ckpt(ckpt=os.path.join(
        DATA_ROOT_DIR,
        'laion-clap-pretrained/laion_clap',
        checkpoint
    ))
    return model


def load_audio(file_path, target_sr=44100, duration=30.0, start=0.0):
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
    return data


@torch.no_grad()
def compute_clap_each(audio_file, model):
    try:
        data = load_audio(audio_file, target_sr=48000, duration=10)
        print(audio_file, 'loaded')
    
    except Exception as e:
        print(audio_file, 'unsuccessful due to', e)
        return None
    
    audio_data = data.reshape(1, -1)

    audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float().cuda()
    audio_embed = model.get_audio_embedding_from_data(x=audio_data_tensor, use_tensor=True)
    audio_embed = audio_embed.squeeze(0).cpu()
    return audio_embed


@torch.no_grad()
def compute_embeddings_batch(batch, audio_files, model):
    batch_results = []
    for i in batch:
        if i >= len(audio_files):
            break
        audio_file = audio_files[i]
        audio_embed = compute_clap_each(audio_file, model)
        batch_results.append((i, audio_file, audio_embed))
    return batch_results


@torch.no_grad()
def precompute_clap_for_dataset(
    dataset_file, 
    embedding_output_file, 
    checkpoint='630k-audioset-fusion-best.pt'
):
    contents, audio_files = load_dataset_file(dataset_file)

    model = load_clap_model(checkpoint)

    if os.path.exists(embedding_output_file):
        print('loading already computed embedding file from', embedding_output_file)
        with open(embedding_output_file, 'rb') as f:
            saved_data = pickle.load(f)
            curr_audio_indices = saved_data['audio_indices']
            curr_audio_files = saved_data['audio_files']
            curr_audio_embeds = saved_data['audio_embeds']

    else:
        curr_audio_indices = []
        curr_audio_files = []
        curr_audio_embeds = []

    print('computing embeddings for {}'.format(dataset_file))
    start_index = len(curr_audio_files)
    remaining_indices = list(range(start_index, len(audio_files)))

    batch_size = 128
    batches = [
        list(range(i, min(i + batch_size, len(audio_files)))) \
            for i in range(start_index, len(audio_files), batch_size)
    ]

    with multiprocessing.Pool(processes=4) as pool:
        for i, batch in enumerate(batches):
            batch_results = pool.map(
                partial(compute_embeddings_batch, model=model, audio_files=audio_files), 
                [batch]
            )

            for result in batch_results[0]:
                curr_audio_indices.append(result[0])
                curr_audio_files.append(result[1])
                curr_audio_embeds.append(result[2])

            with open(embedding_output_file, 'wb') as f:
                pickle.dump({
                    'audio_indices': curr_audio_indices,
                    'audio_files': curr_audio_files, 
                    'audio_embeds': curr_audio_embeds
                }, f)

            print(f"Saved progress for batch {i+1}/{len(batches)}: \
                audio_indices {len(curr_audio_indices)}, \
                audio_files {len(curr_audio_files)}, \
                audio_embeds {len(curr_audio_embeds)}*{curr_audio_embeds[0].shape}")
    
    return curr_audio_indices, curr_audio_files, curr_audio_embeds


def build_faiss_index(embeddings):
    d = embeddings[0].size(0)
    index = faiss.IndexFlatL2(d)
    np_embeddings = np.vstack([emb.numpy() for emb in embeddings])
    index.add(np_embeddings)
    return index


def build_faiss_index_dataset(
    dataset_file, 
    embedding_output_file, 
    faiss_output_file, 
    checkpoint='630k-audioset-fusion-best.pt',
    only_precompute_clap=False
):
    audio_indices, audio_files, audio_embeds = precompute_clap_for_dataset(dataset_file, embedding_output_file, checkpoint)
    
    if only_precompute_clap:
        return 

    valid_indices, valid_files, valid_embeds = [], [], []
    for audio_index, audio_file, audio_embed in zip(audio_indices, audio_files, audio_embeds):
        if audio_embed is not None:
            valid_indices.append(audio_index)
            valid_files.append(audio_file)
            valid_embeds.append(audio_embed)

    print('building faiss index')
    faiss_index = build_faiss_index(valid_embeds)

    print('saving faiss index')
    faiss.write_index(faiss_index, faiss_output_file)
    with open(faiss_output_file + '.filenames', 'wb') as f:
        pickle.dump({'audio_indices': valid_indices, 'audio_files': valid_files}, f)


# ==================== Generate interleaved dataset files ====================

def build_interleaved_dataset(dataset_file, interleaved_output_file, embedding_output_file, faiss_output_file, mode='knn', n_samples=3):
    contents, audio_files = load_dataset_file(dataset_file)

    dataset_dic = {
        "dataset_path": contents["dataset_path"],
        "split": contents["split"],
        "split_path": contents["split_path"],
        "flamingo_task": contents["flamingo_task"],
        "total_num": 0,
        "interleaved_data": {},   
    }

    # interleaved_data is 
    # {
    #     id: {
    #         "generation_index_in_split": index of sample in the train or val or test.json,
    #         "fewshot_indices_in_train": list(indices) of few shot samples in train.json
    #     }
    # }

    if mode == 'knn':
        model = load_clap_model(checkpoint='630k-audioset-fusion-best.pt')

        print('loading already computed embedding file from', embedding_output_file)
        with open(embedding_output_file, 'rb') as f:
            precomputed_data = pickle.load(f)
            precomputed_audio_indices = precomputed_data['audio_indices']
            precomputed_audio_files = precomputed_data['audio_files']
            precomputed_audio_embeds = precomputed_data['audio_embeds']

        faiss_index = faiss.read_index(faiss_output_file)
        with open(faiss_output_file+'.filenames', 'rb') as f:
            _data = pickle.load(f)
        faiss_index_audio_indices = _data['audio_indices']
        faiss_index_audio_files = _data['audio_files']

    print('looking for few shot samples and building interleaved_{} data'.format(mode))
    for i in tqdm(range(contents["total_num"])):
        if mode == 'random':
            few_shot_indices = list(np.random.choice(
                list(set(list(range(contents["total_num"]))) - set([i])),
                size=n_samples-1,
                replace=False
            ))
            few_shot_indices = list(map(int, few_shot_indices))

        elif mode == 'knn':
            if audio_files[i] in precomputed_audio_files:
                idx = precomputed_audio_files.index(audio_files[i])
                query_embedding_np = precomputed_audio_embeds[idx]
                if query_embedding_np is not None:
                    query_embedding_np = query_embedding_np.numpy().reshape(1, -1)
                else:
                    continue

            else:
                query_embedding_np = compute_clap_each(audio_files[i], model)
                if query_embedding_np is not None:
                    query_embedding_np = query_embedding_np.numpy().reshape(1, -1)      
                else:
                    continue       

            distances, knn_indices = faiss_index.search(query_embedding_np, n_samples+50)
            distances = distances[0]
            knn_indices = knn_indices[0]

            knn_filenames = [faiss_index_audio_files[idx] for idx in knn_indices]
            combined = list(zip(knn_indices, knn_filenames))
            unique_indices = defaultdict(list)
            for idx, filename in combined:
                unique_indices[filename].append(idx)

            cleared_knn_indices = [random.choice(unique_indices[filename]) for filename in unique_indices if filename != audio_files[i]]

            if dataset_file.endswith('train.json'):
                cleared_knn_indices = [knn_i for knn_i in cleared_knn_indices if faiss_index_audio_indices[knn_i] != i]
            cleared_knn_indices = cleared_knn_indices[:n_samples-1]
            np.random.shuffle(cleared_knn_indices)
            
            few_shot_indices = [faiss_index_audio_indices[knn_i] for knn_i in cleared_knn_indices]

        dataset_dic["interleaved_data"][dataset_dic["total_num"]] = {
            "generation_index_in_split": i,
            "fewshot_indices_in_train": few_shot_indices
        }
        dataset_dic["total_num"] += 1
    
    with open(interleaved_output_file, 'w') as json_file:
        json.dump(dataset_dic, json_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_name', type=str, help='dataset name')
    parser.add_argument('-f', '--flamingo_task', type=str, help='flamingo task')
    parser.add_argument('--interleave', action="store_true", help='prepare the interleave dataset')
    args = parser.parse_args()

    global DATA_ROOT_DIR
    DATA_ROOT_DIR = "YOUR_DATA_ROOT_DIR"
    dataset_root = os.path.join(DATA_ROOT_DIR, "datasets")
    output_root = os.path.join(DATA_ROOT_DIR, "audio-flamingo-data/dataset_files")
    os.makedirs(output_root, exist_ok=True)

    dataset_name = args.dataset_name  # "Clotho-v2", "AudioSet", "Clotho-AQA", "WavText5K", "FSD50k", ...
    flamingo_task = args.flamingo_task  # AQA, AudioCaptioning, EventClassification, SceneClassification, Tagging, ...

    # must be train first otherwise there's no train.embedding for query
    for split in ["train", "val", "test"]:
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
        
        if args.interleave:
            faiss_output_file = dataset_file.replace('{}.json'.format(split), "train_faiss_index.index")
            embedding_output_file = dataset_file.replace('.json', ".embedding")

            if split == 'train':
                if (not os.path.exists(faiss_output_file)) or (not os.path.exists(faiss_output_file + '.filenames')):
                    build_faiss_index_dataset(
                        dataset_file, embedding_output_file, faiss_output_file, 
                        only_precompute_clap=False
                    )
                else:
                    print('{} exists; exiting'.format(faiss_output_file))
            else:
                build_faiss_index_dataset(
                    dataset_file, embedding_output_file, 
                    faiss_output_file=None, 
                    only_precompute_clap=True
                )
                print('precomputing embedding for {} subset finished'.format(split))

            mode = 'knn'
            interleaved_output_file = '/'.join(
                dataset_file.split('/')[:-1] + \
                ['interleaved_{}-'.format(mode) + dataset_file.split('/')[-1]]
            )
            if not os.path.exists(interleaved_output_file):
                build_interleaved_dataset(
                    dataset_file=dataset_file, 
                    interleaved_output_file=interleaved_output_file, 
                    embedding_output_file=embedding_output_file, 
                    faiss_output_file=faiss_output_file, 
                    mode=mode, 
                    n_samples=8
                )
            else:
                print('{} exists; exiting'.format(interleaved_output_file))
        


