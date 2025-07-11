# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

import os
import os.path as osp
from itertools import chain
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from hydra.utils import instantiate
from torch.utils.data import ConcatDataset, Dataset
from transformers import PreTrainedTokenizer

from llava.data.datasets_mixture import DATASETS_LEGACY
from llava.train.args import DataArguments, TrainingArguments
from llava.utils import io
from llava.utils.logging import logger
import time
import numpy as np
__all__ = ["DATASETS", "MIXTURES", "register_datasets", "register_mixtures", "parse_mixture", "build_dataset"]


def load_dataset_yaml(name):
    fname = f"{name}.yaml" if not name.endswith(".yaml") else name

    # yaml under llava/data/registry/datasets
    repo_path = osp.join(osp.dirname(__file__), "registry", "datasets", fname)
    if osp.exists(repo_path):
        return repo_path

    # # yaml under <fs yaml path>
    abs_path = osp.expanduser(fname)
    if osp.exists(abs_path):
        return abs_path

    raise FileNotFoundError(f"Dataset '{name}' is not found in the {repo_path} or {abs_path}.")


def register_datasets(name: Optional[str] = None):
    if name is None:
        name = os.environ.get("VILA_DATASETS", "default")
        logger.info(f"Registering datasets from environment: '{name}'.")
    # return io.load(osp.join(osp.dirname(__file__), "registry", "datasets", f"{name}.yaml"))
    dataset_meta = {}
    for _name in name.split(","):
        yamlpath = load_dataset_yaml(_name)
        logger.info(f"Registering datasets from: '{yamlpath}'.")
        meta = io.load(yamlpath)
        dataset_meta.update(meta)
    return dataset_meta


def register_mixtures():
    return io.load(os.path.join(os.path.dirname(__file__), "registry", "mixtures.yaml"))


DATASETS = register_datasets()
MIXTURES = register_mixtures()


def parse_mixture(mixture: str) -> List[str]:
    names = mixture.split("+") if "+" in mixture else [mixture]
    while any(name in MIXTURES for name in names):
        names = list(chain(*[MIXTURES.get(name, [name]) for name in names]))
    return sorted(names)


class SubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, limit: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.limit = limit

    def __len__(self) -> int:
        return int(len(self.dataset) * self.limit)

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]

class RepeatedDataset(Dataset):
    def __init__(self, dataset: Dataset, times: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.times = times

    def __len__(self) -> int:
        return len(self.dataset) * self.times

    def __getitem__(self, index: int) -> Any:
        return self.dataset[index % len(self.dataset)]


def get_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    else:
        return 1


def build_dataset(
    mixture: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    logger.warning(f"Training VILA with mixture '{mixture}'.")
    datasets = []
    dataset_rng = np.random.default_rng(1234)
    for name in parse_mixture(mixture):        

        if "*" in name:
            name, times = name.split("*")
            times = int(times)
        else:
            times = 1
        limit_dataset = False
        if "#" in name:
            # we limit the max length of this dataset
            name, max_length_percent = name.split("#")
            limit_dataset = True
        if DATASETS is not None and name in DATASETS:
            if name in DATASETS_LEGACY:
                logger.warning(f"Dataset '{name}' exists in both new and legacy registries. Using the new one.")
            dataset = instantiate(DATASETS[name], _partial_=True)(
                tokenizer=tokenizer,
                data_args=data_args,
                global_batch_size=(
                    training_args.per_device_train_batch_size
                    # * torch.distributed.get_world_size()
                    * get_world_size()
                    * training_args.gradient_accumulation_steps
                ),
            )
        elif name in DATASETS_LEGACY:
            logger.warning(f"Dataset '{name}' is from the legacy registry. Please consider migrating it.")
            dataset = build_dataset_legacy(
                name,
                data_args=data_args,
                training_args=training_args,
                tokenizer=tokenizer,
            )
        else:
            raise ValueError(f"Dataset '{name}' is not found in the registries.")

        
        if limit_dataset:
            # we limit the max length of this dataset
            max_length = int(float(int(max_length_percent) / 100.) * len(dataset))
            dataset = SubsetDataset(dataset, float(int(max_length_percent) / 100.))

        if times > 1:
            dataset = RepeatedDataset(dataset, times)
        datasets.append(dataset)
    return ConcatDataset(datasets)


def build_dataset_legacy(
    name: str,
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
) -> Dataset:
    from llava.data.dataset import (
        LazySupervisedDataset,
        LazyWDSDataset,
    )

    dataset = DATASETS_LEGACY[name]
    dataset_type = dataset.dataset_type
    if dataset_type == "torch":
        dataset_cls = LazySupervisedDataset
    elif dataset_type == "wds":
        dataset_cls = LazyWDSDataset
    else:
        raise NotImplementedError(f"{dataset_type} is not supported.")

    data_args.meta_path = getattr(dataset, "meta_path", None)
    data_args.caption_choice = getattr(dataset, "caption_choice", None)
    data_args.caption_choice_2 = getattr(dataset, "caption_choice_2", None)
    data_args.start_idx = getattr(dataset, "start_idx", None)
    data_args.end_idx = getattr(dataset, "end_idx", None)

    return dataset_cls(
        tokenizer=tokenizer,
        data_path=dataset.data_path,
        image_folder=getattr(dataset, "image_path"),
        data_args=data_args,
        training_args=training_args,
    )
