# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

import argparse
import csv
import itertools
import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.data.builder import DATASETS
from llava.eval.mmmu_utils.eval_utils import parse_choice
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger
from huggingface_hub import snapshot_download
from peft import PeftModel
import torch


def load_existing_ids(output_file):
    if not os.path.exists(output_file):
        return set(), []
    try:
        with open(output_file, "r") as f:
            lines = f.readlines()
            outputs = [json.loads(line) for line in lines]
            processed_ids = {item["id"] for item in outputs}
            return processed_ids, outputs
    except Exception as e:
        print(f"Error loading existing outputs: {e}")
        return set(), []

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def main() -> None:
    
    args = parser.parse_args()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", "-m", type=str, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--think-mode", type=str2bool, default=False)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()
    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # Load stage 3 model with line 56
    model_path = snapshot_download(args.model_base)
    model_think = os.path.join(model_path, 'stage35')

    model = llava.load(model_path, devices=devices)
    if args.think_mode:
        model = PeftModel.from_pretrained(
            model,
            model_think,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    model = model.to("cuda")
    # Set up generation config
    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    # Load data and chunk it
    json_file = DATASETS[args.task]["data_path"]
    instances = io.load(json_file)
    instances = instances[dist.rank() :: dist.size()]

    output_path = os.path.join(args.output_dir, f"outputs_{args.task}.jsonl")
    processed_ids, outputs = load_existing_ids(output_path)
    BATCH_SIZE = 8
    count = len(outputs)
    # Run inference
    new_outputs = []

    for i in tqdm(range(0, len(instances), BATCH_SIZE), disable=not dist.is_main()):
        batch_keys = instances[i:i + BATCH_SIZE]

        batch_sounds = []
        batch_prompts = []
        batch_ids = []
        batch_gt_answers = []
        for key in batch_keys:
            sound_path = instances[key]["sound"]
            if sound_path in processed_ids:
                continue

            try:
                sound = llava.Sound(sound_path)
            except Exception as e:
                logger.warning(f"Failed to load sound for {sound_path}: {e}")
                continue
            conversations = instances[key]["conversations"]
            question = conversations[0]["value"]
            prompt = '<sound>\n' + question
            batch_sounds.append(sound)
            batch_prompts.append(prompt)
            batch_ids.append(sound_path)
            batch_gt_answers.append(conversations[1]["value"])

        if not batch_sounds:
            continue

        # try:
        responses = model.generate_content_batch(
            [[s, p] for s, p in zip(batch_sounds, batch_prompts)],
            generation_config=generation_config
        )
        
        if isinstance(responses, str):
            responses = [responses]

        for idx, response in enumerate(responses):
            key = batch_keys[idx]
            output = {
                "id": batch_ids[idx],
                "question": batch_prompts[idx],
                "gt_answer": batch_gt_answers[idx],
                "pred": response
            }
            new_outputs.append(output)
            count += 1

        if count % 20 == 0:
            # Gather and save outputs
            if dist.size() > 1:
                outputs_new = dist.gather(new_outputs, dst=0)
                if dist.is_main():
                    outputs_new = list(itertools.chain(*outputs_new))
                    final_outputs = outputs + outputs_new
                    io.save(os.path.join(args.output_dir, f"outputs_{args.task}.jsonl"), final_outputs)
            else:
                final_outputs = outputs + new_outputs
                io.save(os.path.join(args.output_dir, f"outputs_{args.task}.jsonl"), final_outputs)
    if dist.size() > 1:
        new_outputs = dist.gather(new_outputs, dst=0)
        if not dist.is_main():
            return
        new_outputs = list(itertools.chain(*new_outputs))
        final_outputs = outputs + new_outputs
    io.save(os.path.join(args.output_dir, "outputs_"+str(args.task)+".jsonl"), final_outputs)

if __name__ == "__main__":
    main()
