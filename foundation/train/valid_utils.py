# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/mlfoundations/open_flamingo under the MIT license.
#   LICENSE is in incl_licenses directory.

import argparse
import functools
import os
import random
from tqdm import tqdm
import sys 
sys.path.append('../')
import yaml
import time

import numpy as np
import torch
from data.data import get_audiotext_dataloader


@torch.no_grad()
def validation_losses(model, data_config, clap_config, tokenizer, batch_size, autocast, cast_dtype, device_id, verbose=True):

    model.eval()

    @torch.no_grad()
    def get_val_loss(validloader):

        loss_sum = 0.0
        for idx, batch in tqdm(enumerate(validloader)):

            audio_clips = batch["audio_clips"].to(device_id, dtype=cast_dtype, non_blocking=True)
            audio_embed_mask = batch["audio_embed_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)
            input_ids = batch["input_ids"].to(device_id, dtype=cast_dtype, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device_id, dtype=cast_dtype, non_blocking=True)

            labels = input_ids.clone()
            labels[labels == tokenizer.pad_token_id] = -100
            labels[:, :2] = -100
            labels[labels == tokenizer.encode("<audio>")[-1]] = -100

            sep_locations = labels == tokenizer.sep_token_id
            eoc_locations = labels == endofchunk_token_id

            for i in range(labels.shape[0]):
                shouldmask = True
                for j in range(labels.shape[1]):
                    if shouldmask and (labels[i][j] != tokenizer.eos_token_id):
                        masked_value = -100
                    else:
                        masked_value = labels[i][j]

                    if labels[i][j] == tokenizer.sep_token_id:
                        shouldmask = False
                    elif labels[i][j] == endofchunk_token_id:
                        shouldmask = True
                    
                    labels[i][j] = masked_value
                
                if labels[i][-1] not in [-100, tokenizer.eos_token_id, tokenizer.pad_token_id, endofchunk_token_id]:
                    for j in range(labels.shape[1]-1, -1, -1):
                        if labels[i][j] not in [-100, tokenizer.eos_token_id, endofchunk_token_id]:
                            labels[i][j] = -100
                        else:
                            break

            labels = labels.to(device_id)

            with autocast():
                output = model(
                    audio_x=audio_clips,
                    audio_x_mask=audio_embed_mask,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                valid_loss_no_multiplier = output.loss.item()
                loss_sum += valid_loss_no_multiplier

        return loss_sum / ((idx+1) * batch_size)

    media_token_id = tokenizer("<audio>", add_special_tokens=False)["input_ids"][-1]
    assert media_token_id == tokenizer.encode("<audio>")[-1]
    endofchunk_token_id = tokenizer("<|endofchunk|>", add_special_tokens=False)["input_ids"][-1]

    valid_losses = {}
    all_valid_AudioTextDataInfo = get_audiotext_dataloader(data_config, clap_config, tokenizer, batch_size, split='val')
    for valid_dataset_name in all_valid_AudioTextDataInfo:
        if verbose:
            print('computing validation loss on {}'.format(valid_dataset_name))

        validloader = all_valid_AudioTextDataInfo[valid_dataset_name].dataloader 
        valid_losses[valid_dataset_name] = get_val_loss(validloader)

        if verbose:
            print('validation loss on {} is {:.3f}'.format(valid_dataset_name, valid_losses[valid_dataset_name]))
    
    model.train() 

    return valid_losses
