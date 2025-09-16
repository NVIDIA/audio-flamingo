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
from peft import PeftModel


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
    parser.add_argument("--model-base", "-m", type=str, required=True)
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--generation-config", type=None)
    parser.add_argument("--think-mode", action="store_true")
    parser.add_argument("--output-dir", type=str, default="./")
    args = parser.parse_args()

    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # Load data
    json_file = 'static/mmar.json'
    json_data = io.load(json_file)

    # Load model
    from huggingface_hub import snapshot_download
    model_path = snapshot_download(args.model_base)
    model_think = os.path.join(model_path, 'stage35')

    model = llava.load(model_path, device_map=None)
    if args.think_mode:
        model = PeftModel.from_pretrained(
            model,
            model_think,
            # device_map="auto",
            torch_dtype=torch.float16,
        )
    model = model.to("cuda")
    model.config.pad_token_id = model.tokenizer.eos_token_id

    generation_config = model.default_generation_config
    # generation_config= {"max_new_tokens": 256}
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    data_dir = json_data["split_path"]
    instances = json_data["data"]
    items = list(instances.items())
    instances = items[dist.rank()::dist.size()]
    instances = dict(instances)

    output_path = os.path.join(args.output_dir, f"outputs.jsonl")
    processed_ids, outputs = load_existing_ids(output_path)
    count = len(outputs)

    # Batch inference
    BATCH_SIZE = args.batch_size
    keys = list(instances.keys())
    new_outputs = []

    for i in tqdm(range(0, len(keys), BATCH_SIZE), disable=not dist.is_main()):
        batch_keys = keys[i:i + BATCH_SIZE]

        batch_sounds = []
        batch_prompts = []
        batch_ids = []

        for key in batch_keys:
            sound_path = os.path.join(data_dir, instances[key]["name"])
            if sound_path in processed_ids:
                continue

            try:
                sound = llava.Sound(sound_path)
            except Exception as e:
                logger.warning(f"Failed to load sound for {sound_path}: {e}")
                continue
            if args.think_mode:
                prompt = '<sound>\n' + instances[key]["prompt"] + "\nPlease think and reason about the input music before you respond."
            else:
                prompt = '<sound>\n' + instances[key]["prompt"]
            batch_sounds.append(sound)
            batch_prompts.append(prompt)
            batch_ids.append(sound_path)

        if not batch_sounds:
            continue

        with torch.no_grad():
            responses = model.generate_content_batch_decode(
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
                "gt_answer": instances[key]["output"],
                "pred": response
            }
            new_outputs.append(output)
            count += 1

        # # Periodic save
        if count % 2 == 0:
            if dist.size() > 1:
                outputs_new = dist.gather(new_outputs, dst=0)
                if dist.is_main():
                    outputs_new = list(itertools.chain(*outputs_new))
                    final_outputs = outputs + outputs_new
                    io.save(output_path, final_outputs)
            else:
                final_outputs = outputs + new_outputs
                io.save(output_path, final_outputs)

    # Final save
    if dist.size() > 1:
        new_outputs = dist.gather(new_outputs, dst=0)
        if not dist.is_main():
            return
        new_outputs = list(itertools.chain(*new_outputs))
        final_outputs = outputs + new_outputs
    else:
        final_outputs = outputs + new_outputs

    io.save(output_path, final_outputs)


if __name__ == "__main__":
    main()
