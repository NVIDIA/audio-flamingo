#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
import itertools
import json
import os
import random
import string
import socket, contextlib, time
from typing import Tuple, Sequence

# ---- BEFORE importing torch.distributed, set NCCL safety (even if we use gloo) ----
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import torch
import torch.distributed as tdist
from tqdm import tqdm

import llava
from llava.utils import io
from llava.utils.logging import logger
from huggingface_hub import snapshot_download
from peft import PeftModel

# -------------------- utils --------------------

sound_tag_re = re.compile(r"<sound(?:-\d+)?>")

def random_string(length=12):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in ("yes", "true", "t", "1"): return True
    if v in ("no", "false", "f", "0"): return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

def _norm(p: str) -> str:
    try:
        return os.path.abspath(p)
    except Exception:
        return p

def load_existing_ids(output_file):
    """Reads existing JSONL results and returns (processed_ids, outputs_list)."""
    if not os.path.exists(output_file):
        return set(), []
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            outputs = [json.loads(line) for line in f if line.strip()]
        processed_ids = {item.get("id") for item in outputs if "id" in item}
        return set(_norm(p) for p in processed_ids if p is not None), outputs
    except Exception as e:
        print(f"Error loading existing outputs: {e}")
        return set(), []

def append_jsonl(path: str, records):
    if not records:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _safe_find_free_port(start: int = 29500, span: int = 2000, retries: int = 200) -> int:
    import random
    for _ in range(retries):
        port = random.randint(start, start + span)
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError("Could not find a free port after retries")

def _autodetect_dist_env() -> Tuple[int, int, int, int, str, str]:
    e = os.environ
    world_size = int(e.get("WORLD_SIZE") or e.get("SLURM_NTASKS") or "1")
    rank = int(e.get("RANK") or e.get("SLURM_PROCID") or "0")
    local_rank = int(e.get("LOCAL_RANK") or e.get("SLURM_LOCALID") or "0")
    local_world_size = int(e.get("LOCAL_WORLD_SIZE") or e.get("SLURM_NTASKS_PER_NODE") or "1")
    master_addr = e.get("MASTER_ADDR") or "127.0.0.1"
    master_port = e.get("MASTER_PORT") or ""
    if world_size <= 1:
        world_size, rank, local_rank, local_world_size = 1, 0, 0, 1
    return world_size, rank, local_world_size, local_rank, master_addr, master_port

def _prepare_dist_env(output_dir: str):
    ws, r, lws, lr, addr, port = _autodetect_dist_env()
    os.makedirs(output_dir, exist_ok=True)
    port_file = os.path.join(output_dir, ".master_port")

    if ws > 1 and not port:
        if r == 0:
            chosen = _safe_find_free_port()
            with open(port_file, "w") as f:
                f.write(str(chosen))
            os.environ["MASTER_PORT"] = str(chosen)
        else:
            t0 = time.time()
            while not os.path.exists(port_file):
                time.sleep(0.05)
                if time.time() - t0 > 30:
                    raise RuntimeError("Timeout waiting for master port file")
            with open(port_file, "r") as f:
                os.environ["MASTER_PORT"] = f.read().strip()
    elif ws <= 1:
        os.environ.setdefault("MASTER_PORT", str(_safe_find_free_port()))
        r, lr, lws = 0, 0, 1

    os.environ.setdefault("MASTER_ADDR", addr or "127.0.0.1")
    os.environ.setdefault("WORLD_SIZE", str(ws))
    os.environ.setdefault("RANK", str(r))
    os.environ.setdefault("LOCAL_WORLD_SIZE", str(lws))
    os.environ.setdefault("LOCAL_RANK", str(lr))

    print(f"[DIST] RANK={os.environ['RANK']} WORLD_SIZE={os.environ['WORLD_SIZE']} "
          f"LOCAL_RANK={os.environ['LOCAL_RANK']}/{os.environ['LOCAL_WORLD_SIZE']} "
          f"MASTER={os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
          flush=True)

    return int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"]), int(os.environ["LOCAL_WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])

def _init_process_group_gloo():
    if not tdist.is_initialized():
        tdist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            world_size=int(os.environ["WORLD_SIZE"]),
            rank=int(os.environ["RANK"]),
        )


# -------------------- main --------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", "-m", type=str, required=True)
    parser.add_argument("--json-path", type=str, default="static/mmar.json",
                        help="Path to MMAR-style input JSON file.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--think-mode", type=str2bool, default=False)
    parser.add_argument("--output-dir", type=str, default="./")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Append after every N outer steps (rank 0).")
    args = parser.parse_args()

    # Dist env & init (GLOO)
    world_size, rank, local_world_size, local_rank = _prepare_dist_env(args.output_dir)
    _init_process_group_gloo()

    # Device mapping
    num_cuda = torch.cuda.device_count()
    if num_cuda == 0:
        raise RuntimeError("No CUDA devices visible.")
    torch.cuda.set_device(local_rank % num_cuda)
    devices: Sequence[int] = range(local_rank, num_cuda, local_world_size)
    if rank == 0:
        print("Distributed Setup Complete!", flush=True)

    # ---------- Load data ----------
    json_file = args.json_path
    json_data = io.load(json_file)
    data_dir = json_data["split_path"]
    instances = json_data["data"]
    keys = list(instances.keys())
    sharded_keys = keys[rank::world_size]

    # ---------- Load model ----------
    if not os.path.isdir(args.model_base):
        model_path = snapshot_download(args.model_base)
    else:
        model_path = args.model_base
    model_think = os.path.join(model_path, "stage35")

    model = llava.load(model_path, devices=devices)
    if args.think_mode:
        if os.path.exists(os.path.join(model_think, "non_lora_trainables.bin")):
            non_lora_trainables = torch.load(
                os.path.join(model_think, "non_lora_trainables.bin"),
                map_location="cpu",
            )
            non_lora_trainables = {
                    (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
                }
            model.load_state_dict(non_lora_trainables, strict=False)
        model = PeftModel.from_pretrained(
            model,
            model_think,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    model = model.to("cuda")

    generation_config = getattr(model, "default_generation_config", {})
    if args.generation_config:
        generation_config.update(**args.generation_config)

    # ---------- Outputs ----------
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "outputs.jsonl")

    if args.overwrite:
        processed_ids, outputs = set(), []
    else:
        processed_ids, outputs = load_existing_ids(output_path)

    if rank == 0:
        logger.info(f"Total items: {len(keys)} | This rank items: {len(sharded_keys)} | Already done: {len(processed_ids)}")

    BATCH_SIZE = args.batch_size
    new_outputs = []
    steps_since_save = 0

    # ---------- Inference loop ----------
    for i in tqdm(range(0, len(sharded_keys), BATCH_SIZE), disable=(rank != 0)):
        batch_keys = sharded_keys[i:i + BATCH_SIZE]

        batch_sounds, batch_prompts, batch_ids, batch_gt_answers = [], [], [], []

        for key in batch_keys:

            rec = instances[key]
            if isinstance(rec["name"], list):
                sound_path = [os.path.join(data_dir, item) for item in rec["name"]]
                nid = _norm("-".join(sound_path))
            else:
                sound_path = os.path.join(data_dir, rec["name"])
                nid = _norm(sound_path)

            if nid in processed_ids:
                continue

            if isinstance(sound_path, str):
                if os.path.isfile(sound_path):
                    try:
                        sound = llava.Sound(sound_path)
                    except Exception as e:
                        logger.warning(f"Failed to load sound for {sound_path}: {e}")
                        continue
                else:
                    # if text only question, in this case, the ID is arbitary
                    sound = None
            elif isinstance(sound_path, list):
                sound = []
                for sp in sound_path:
                    try:
                        sound.append(llava.Sound(sp))
                    except Exception as e:
                        logger.warning(f"Failed to load sound for {sp}: {e}")
                        continue

            if sound is not None:
                # Check if neither <sound> nor <sound-*> is present anywhere in the prompt
                if not sound_tag_re.search(rec["prompt"]):
                    prompt = "<sound>\n" + rec["prompt"]
                else:
                    prompt = rec["prompt"]
            else:
                prompt = rec["prompt"]

            batch_sounds.append(sound)
            batch_prompts.append(prompt)
            batch_ids.append(nid)
            batch_gt_answers.append(rec["output"])

        if not batch_sounds:
            steps_since_save += 1
            if steps_since_save >= args.save_every:
                if world_size > 1:
                    gathered = [None for _ in range(world_size)] if rank == 0 else None
                    tdist.gather_object(new_outputs, object_gather_list=gathered, dst=0)
                    if rank == 0:
                        merged = list(itertools.chain.from_iterable(g for g in gathered if g))
                        append_jsonl(output_path, merged)
                        new_outputs.clear()
                else:
                    append_jsonl(output_path, new_outputs)
                    new_outputs.clear()
                steps_since_save = 0
            continue

        responses = model.generate_content_batched(
            [[s, p] for s, p in zip(batch_sounds, batch_prompts)],
            generation_config=generation_config,
        )
        if isinstance(responses, str):
            responses = [responses]

        for idx, response in enumerate(responses):
            rec_out = {
                "id": batch_ids[idx],
                "question": batch_prompts[idx],
                "gt_answer": batch_gt_answers[idx],
                "pred": response,
            }
            new_outputs.append(rec_out)
            processed_ids.add(batch_ids[idx])

        steps_since_save += 1

        if steps_since_save >= args.save_every:
            if world_size > 1:
                gathered = [None for _ in range(world_size)] if rank == 0 else None
                tdist.gather_object(new_outputs, object_gather_list=gathered, dst=0)
                if rank == 0:
                    merged = list(itertools.chain.from_iterable(g for g in gathered if g))
                    append_jsonl(output_path, merged)
                new_outputs.clear()
            else:
                append_jsonl(output_path, new_outputs)
                new_outputs.clear()
            steps_since_save = 0

    # ---------- Final save ----------
    if world_size > 1:
        gathered = [None for _ in range(world_size)] if rank == 0 else None
        tdist.gather_object(new_outputs, object_gather_list=gathered, dst=0)
        if rank == 0:
            merged = list(itertools.chain.from_iterable(g for g in gathered if g))
            append_jsonl(output_path, merged)
    else:
        append_jsonl(output_path, new_outputs)

    tdist.barrier()
    if rank == 0:
        print(f"[DONE] Appended results to {output_path}", flush=True)


if __name__ == "__main__":
    import sys
    print(f"[BOOT] __file__={__file__} argv={sys.argv}", flush=True)
    main()