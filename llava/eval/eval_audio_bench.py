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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="auto")
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # Load stage 3 model with line 56
    model = llava.load(args.model_base, model_base=None, devices=devices)
    # Uncomment line 58-63 to load stage 3.5 model on top of stage 3 for thinking mode and long audio mode
    # model = PeftModel.from_pretrained(
    #         model,
    #         args.model_path,
    #         device_map="auto",
    #         torch_dtype=torch.float16,
    #     )
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

    count = len(outputs)
    # Run inference
    new_outputs = []
    for instance in tqdm(instances, disable=not dist.is_main()):
        uuid = instance["id"]
        sound_path = instance["sound"]

        if sound_path in processed_ids:
            continue  # Skip if already processed
        sound = llava.Sound(sound_path)
        conversations = instance["conversations"]
        question = conversations[0]["value"]

        response = model.generate_content([sound, question], generation_config=generation_config)
       
        print("response", response)

        output = {"id": sound_path, "question": question, "gt_answer": conversations[1]["value"],  "pred": response}
        new_outputs.append(output)
        count = count +1
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
