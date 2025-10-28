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
from typing import Tuple, Sequence, Any

# ---- BEFORE importing torch.distributed, set NCCL safety (even if we use gloo) ----
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_SHM_DISABLE", "1")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

import torch
import torch.distributed as tdist
from tqdm import tqdm

import llava
# Prevent llava.utils.distributed from re-initing later
from llava.utils import distributed as llava_dist
from llava.data.builder import DATASETS
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
    master_port = e.get("MASTER_PORT") or ""  # empty means “not set”
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
    # prevent llava_dist.init() from re-initializing to NCCL
    if hasattr(llava_dist, "init"):
        try:
            llava_dist.init = lambda *a, **k: None
        except Exception:
            pass

# ---------- schema helpers ----------

def _load_instances_and_datadir(obj: Any):
    """
    Supports:
      1) MMAR-style: {"split_path": "...", "data": {...}}
      2) Direct dict/list of rows (conversations-style or similar).
    Returns: (instances, data_dir)
    """
    data_dir = ""
    instances = obj
    if isinstance(obj, dict) and "split_path" in obj and "data" in obj:
        data_dir = obj.get("split_path") or ""
        instances = obj.get("data") or {}
    return instances, data_dir

def _extract_record_fields(row: Any, data_dir: str):
    """
    Normalizes fields from either schema into:
      - sound_path: str | list[str] | None
      - prompt: str
      - gt_answer: str
      - id_hint: str  (for stable dedup id; may be file path(s) or random)
    """
    if isinstance(row, dict) and "name" in row and "prompt" in row:
        name = row["name"]
        if isinstance(name, list):
            paths = [os.path.join(data_dir, p) for p in name]
            sound_path = paths
            id_hint = "-".join(paths)
        else:
            p = os.path.join(data_dir, name)
            sound_path = p
            id_hint = p
        prompt = row["prompt"]
        gt_answer = row.get("output", "")
        return sound_path, prompt, gt_answer, id_hint

    # conversations-style
    if isinstance(row, dict) and "conversations" in row:
        sound_path = row.get("sound", None)
        if isinstance(sound_path, list):
            id_hint = "-".join(sound_path)
        else:
            id_hint = sound_path if isinstance(sound_path, str) else random_string()
        convs = row["conversations"]
        question = (convs[0]["value"] if len(convs) > 0 else "").strip()
        gt_answer = (convs[1]["value"] if len(convs) > 1 else "")
        prompt = question
        return sound_path, prompt, gt_answer, id_hint

    # Fallback: treat as text-only row if possible
    prompt = row.get("prompt", "") if isinstance(row, dict) else ""
    gt_answer = row.get("output", "") if isinstance(row, dict) else ""
    return None, prompt, gt_answer, random_string()

# -------------------- main --------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", "-m", type=str, required=True)
    parser.add_argument("--task", type=str, default="Clotho-AQA-AQA")
    parser.add_argument("--infer-json", type=str, default=None,
                        help="If provided, load this JSON/JSONL; else fallback to DATASETS[task].")
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--think-mode", type=str2bool, default=False)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--output-dir", type=str, default="outputs/")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Append to output after every N steps (rank 0 only).")
    parser.add_argument("--batch-size", type=int, default=8)
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

    # Load model
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

    # ---- Data loading (supports both schemas) ----
    if args.infer_json is not None:
        raw = io.load(args.infer_json)
    else:
        json_file = DATASETS[args.task]["data_path"]
        raw = io.load(json_file)

    instances, data_dir = _load_instances_and_datadir(raw)

    if isinstance(instances, dict):
        all_keys = list(instances.keys())
        get_row = lambda k: instances[k]
        out_suffix = args.task
    else:
        all_keys = list(range(len(instances)))
        get_row = lambda k: instances[k]
        out_suffix = args.task if args.infer_json is None else os.path.splitext(os.path.basename(args.infer_json))[0]

    sharded_keys = all_keys[rank::world_size]

    # ---- Outputs ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"outputs_{out_suffix}.jsonl")

    if args.overwrite:
        processed_ids, outputs = set(), []
    else:
        processed_ids, outputs = load_existing_ids(output_path)

    BATCH_SIZE = args.batch_size
    new_outputs = []
    steps_since_save = 0

    if rank == 0:
        logger.info(f"Total items: {len(all_keys)} | This rank items: {len(sharded_keys)} | Already done: {len(processed_ids)}")

    # ---- Inference loop ----
    for i in tqdm(range(0, len(sharded_keys), BATCH_SIZE), disable=(rank != 0)):
        batch_keys = sharded_keys[i:i + BATCH_SIZE]
        batch_sounds, batch_prompts, batch_ids, batch_gt_answers = [], [], [], []

        for key in batch_keys:
            row = get_row(key)
            sound_path, prompt_in, gt_answer, id_hint = _extract_record_fields(row, data_dir)

            # Normalize ID
            if isinstance(id_hint, str):
                nid = _norm(id_hint)
            else:
                nid = _norm(random_string())
            if nid in processed_ids:
                continue

            # Build llava.Sound (single, list, or None)
            sound = None
            if isinstance(sound_path, str):
                if os.path.isfile(sound_path):
                    try:
                        sound = llava.Sound(sound_path)
                    except Exception as e:
                        logger.warning(f"Failed to load sound for {sound_path}: {e}")
                        # leave sound=None (text-only)
                else:
                    # text-only: keep sound=None
                    pass
            elif isinstance(sound_path, list):
                tmp_list = []
                for sp in sound_path:
                    try:
                        if os.path.isfile(sp):
                            tmp_list.append(llava.Sound(sp))
                        else:
                            logger.warning(f"Sound path does not exist: {sp}")
                    except Exception as e:
                        logger.warning(f"Failed to load sound for {sp}: {e}")
                # if none succeeded, keep sound as [] to signal multi-audio expected but missing
                sound = tmp_list if len(tmp_list) > 0 else []

            # Prompt construction with proper sound-tag handling
            prompt = prompt_in
            if sound is not None and (isinstance(sound, llava.Sound) or isinstance(sound, list)):
                # Only add <sound> if no <sound> or <sound-*> tag appears anywhere
                if not sound_tag_re.search(prompt):
                    if args.think_mode:
                        prompt = "<sound>\n" + prompt + "\nPlease think and reason about the input music before you respond."
                    else:
                        prompt = "<sound>\n" + prompt
            # If sound is None (text-only), leave prompt as-is (even if no tag)

            batch_sounds.append(sound)
            batch_prompts.append(prompt)
            batch_ids.append(nid)
            batch_gt_answers.append(gt_answer)

        # If nothing to do in this step, still handle periodic save
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

        # ---- Batched generation: input is exactly [[sound, prompt], ...] ----
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

        # ---- Periodic APPEND ----
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

    # ---- Final save & exit ----
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
