# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.



import argparse
import functools
import glob
import os
import random
import string
import json
import sys 
sys.path.append('../')
from tqdm import tqdm
import yaml
from collections import defaultdict
import io
import warnings
import subprocess
import pickle

import numpy as np
import torch

from data.data import get_audiotext_dataloader
from src.factory import create_model_and_transforms
from train.train_utils import Dict2Class, get_autocast, get_cast_dtype

def inference_this(
    args, data_config, clap_config, model_config, test_dataset_name, tmp_file,
    temperature=1.0, num_beams=3, ckpt=-1, end_batch_idx=-2, verbose=False,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable the tokenizer parallelism warning
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config, 
        use_local_files=args.offline,
        gradient_checkpointing=args.gradient_checkpointing,
        freeze_lm_embeddings=args.freeze_lm_embeddings,
    )

    device_id = 0
    model = model.to(device_id)
    model.eval()

    if ckpt == -1:
        checkpoint_list = glob.glob(f"{args.expdir}/{args.run_name}/checkpoint_*.pt")
        resume_from_checkpoint = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
    else:
        resume_from_checkpoint = f"{args.expdir}/{args.run_name}/checkpoint_{ckpt}.pt"
    checkpoint = torch.load(resume_from_checkpoint, map_location="cpu")
    msd = checkpoint["model_state_dict"]
    msd = {k.replace("module.", ""): v for k, v in msd.items()}
    x,y = model.load_state_dict(msd, False)
    print(x)
    print(y)
    
    autocast = get_autocast(
        args.precision, cache_enabled=(not args.fsdp)
    )
    cast_dtype = get_cast_dtype(args.precision)

    # model = model.to(dtype=cast_dtype)

    if test_dataset_name in data_config["valid_dataset_config"]:
        data_config["valid_dataset_config"] = {test_dataset_name: data_config["valid_dataset_config"][test_dataset_name]}
    else:
        data_config["valid_dataset_config"] = {test_dataset_name: True}
    
    all_test_AudioTextDataInfo = get_audiotext_dataloader(data_config, clap_config, tokenizer, args.batch_size, split='test')
    
    assert test_dataset_name in list(all_test_AudioTextDataInfo.keys()), "{} not a test set".format(test_dataset_name)
    dataloader = all_test_AudioTextDataInfo[test_dataset_name].dataloader

    deduplicate_tasks = ["Clotho-v2-AudioCaptioning", "audiocaps-AudioCaptioning", "MACS-AudioCaptioning", "LP-MusicCaps-MSD-AudioCaptioning", "LP-MusicCaps-MC-AudioCaptioning"]
    if any([test_dataset_name.startswith(x) for x in deduplicate_tasks]):
        deduplicate = True 
    else:
        deduplicate = False

    if os.path.exists(tmp_file):
        with open(tmp_file, 'rb') as pickle_file:
            tmp_data = pickle.load(pickle_file)
        results_dic = tmp_data['results_dic']
        results = tmp_data['results']
        finished_batches = tmp_data['finished_batches']
        print('reading tmp data from {}: {} batches already computed'.format(tmp_file, finished_batches+1))
    
    else:
        tmp_data = {}
        results_dic = {}  # for deduplicate
        results = []  # for non-deduplicate
        finished_batches = -1
        print('no tmp data found; will store tmp data to {}'.format(tmp_file))

    # print(len(dataloader))
    # print('---------------------')
    for batch_idx, batch in tqdm(enumerate(dataloader)):
        if end_batch_idx > 0 and batch_idx == end_batch_idx:
            break
        
        if batch_idx <= finished_batches:
            continue

        audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
        audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
        input_ids = batch["input_ids"].to(device_id, non_blocking=True)
        filenames = batch["filenames"]
        # print(input_ids)

        media_token_id = tokenizer.encode("<audio>")[-1]
        sep_token_id = tokenizer.sep_token_id

        for idx in range(input_ids.shape[0]):
            filename = filenames[idx]
            if type(filename) is list:
                # interleaved data
                filename = filename[-1]

            input_id = input_ids[idx]
            for sep_location in range(len(input_id)-1, -1, -1):
                # find last <SEP>
                if input_id[sep_location] == sep_token_id:
                    break
            # print(tokenizer.decode(input_id))
            prompt = input_id[:sep_location+1]

            prompt_decoded = tokenizer.decode(prompt).replace(tokenizer.sep_token, '')
            ground_truth_decoded = tokenizer.decode(input_id).split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '')
            
            if not (deduplicate and (filename, prompt_decoded) in results_dic):
                # print(prompt)
                # print(prompt_decoded)
                output = model.generate(
                    audio_x=audio_clips[idx].unsqueeze(0),
                    audio_x_mask=audio_embed_mask[idx].unsqueeze(0),
                    lang_x=prompt.unsqueeze(0),
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=256,
                    temperature=temperature,
                )[0]
                output_decoded = tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '')
                # print(ground_truth_decoded)
                # print('------')
                # print(output_decoded)

            if deduplicate:
                if (filename, prompt_decoded) in results_dic:
                    results_dic[(filename, prompt_decoded)]['ground_truth'].append(ground_truth_decoded)
            
                else:
                    results_dic[(filename, prompt_decoded)] = {
                        'ground_truth': [ground_truth_decoded], 
                        'output': output_decoded
                    }
            else:
                results.append((filename, prompt_decoded, ground_truth_decoded, output_decoded))
                

        tmp_data['results_dic'] = results_dic
        tmp_data['results'] = results
        tmp_data['finished_batches'] = batch_idx
        with open(tmp_file, 'wb') as pickle_file:
            pickle.dump(tmp_data, pickle_file)

    if deduplicate:
        for (filename, prompt) in results_dic:
            ground_truth = '|'.join(results_dic[(filename, prompt)]['ground_truth'])
            output = results_dic[(filename, prompt)]['output']
            results.append((filename, prompt, ground_truth, output))

    # if verbose:
    #     for filename, prompt, ground_truth, output in results:
    #         print('-'*30)
    #         print('filename:', filename)
    #         print('prompt:', prompt)
    #         print('ground_truth:', ground_truth)
    #         print('output:', output)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../config/config.yaml', help='yaml config path')
    parser.add_argument('-t', '--task', type=str, help='which task to inference')
    parser.add_argument('-temp', '--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('-nb', '--num_beams', type=int, default=1, help='num beams for beam search')
    parser.add_argument('--ckpt', type=int, default=-1, help='checkpoint idx, -1 means latest')
    parsed_args = parser.parse_args()

    print(parsed_args)

    test_dataset_name = parsed_args.task

    output_file = os.path.join(
        '../outputs/', 
        parsed_args.task.replace('/', '-'), 
        '{}-ckpt{}.log'.format(
            parsed_args.config.split('/')[-1][:-5], 
            parsed_args.ckpt
        )
    )
    tmp_file = output_file.replace('.log', '.tmp.pickle')
    print('output file:', output_file)

    print('no previous log file; generating samples')

    config = yaml.load(open(parsed_args.config), Loader=yaml.FullLoader)
    # print(config)
    # print('----------------------')
    data_config = config['data_config']
    model_config = config['model_config']
    print(model_config)
    clap_config = config['clap_config']
    args = Dict2Class(config['train_config'])

    results = inference_this(
        args, data_config, clap_config, model_config, test_dataset_name, 
        temperature=float(parsed_args.temperature),
        num_beams=int(parsed_args.num_beams),
        ckpt=parsed_args.ckpt,
        verbose=True,
        tmp_file=tmp_file,
    )

if __name__ == "__main__":
    main()