# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/


import json
import os
import random
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import ConcatDataset, Dataset, DistributedSampler, RandomSampler, Sampler
from transformers import PreTrainedModel, Trainer
from transformers.modeling_utils import unwrap_model
from transformers.trainer import ALL_LAYERNORM_LAYERS  # ShardedDDPOption,
from transformers.trainer import get_parameter_names, has_length, is_sagemaker_mp_enabled, logger

from llava.train.sequence_parallel import get_pg_manager
from llava.trl.trainer import DPOTrainer
import numpy as np

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [
        lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class VILADistributedSampler(DistributedSampler):
    """This class is implemented by Jason Lu."""

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size=None,
        # NOTE: this is the total size but not per-worker
        sample_len_list=None,
        force_accumulation=True,
        sp_degree: int = 1,
        gradient_accumulation_steps: int = 1,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True  # always True
        self.sp_degree = max(1, sp_degree)
        self.bs_divisible_by_sp = batch_size % self.sp_degree == 0

        # Consider sequence parallelism
        if self.sp_degree > 1:  # Sequence Parallelism is enabled
            PROCESS_GROUP_MANAGER = get_pg_manager()
            self.dp_rank = PROCESS_GROUP_MANAGER.dp_rank
            self.dp_num_replicas = num_replicas // sp_degree
            self.corresponding_ranks = list(range(self.dp_rank * self.sp_degree, (self.dp_rank + 1) * self.sp_degree))
        else:
            self.dp_rank = rank
            self.dp_num_replicas = num_replicas

        self.batch_size = batch_size
        self.global_batch_size = batch_size * self.dp_num_replicas

        # NOTE: org_ is without drop last
        self.org_sample_len_list = self.per_replica_samples = sample_len_list
        assert sum(sample_len_list) == len(self.dataset)

        if self.drop_last:  # type: ignore[arg-type]
            self.per_replica_samples = [
                sample_len
                // (self.num_replicas * self.batch_size * gradient_accumulation_steps // self.sp_degree)
                * self.batch_size
                * gradient_accumulation_steps
                // self.sp_degree
                for sample_len in self.per_replica_samples
            ]
            self.num_samples = sum(self.per_replica_samples)
        else:
            raise NotImplementedError

        self.total_size = self.num_samples * self.num_replicas
        self.total_samples = [samples * self.num_replicas for samples in self.per_replica_samples]

        self.shuffle = shuffle
        self.seed = seed

        # whether to force accumulate
        self.force_accumulation = force_accumulation

    def __len__(self) -> int:
        return self.num_samples * self.sp_degree

    def __iter__(self):

        indices = list(range(len(self.dataset)))

        # 1. split the full indices first (note: without drop last at this moment)
        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(
                indices[sum(self.org_sample_len_list[:i]) : sum(self.org_sample_len_list[:i]) + self.total_samples[i]]
            )

        assert sum([len(indices) for indices in indices_list]) == self.total_size, (
            sum([len(indices) for indices in indices_list]),
            self.total_size,
        )

        if (
            self.sp_degree > 1 and self.bs_divisible_by_sp
        ):  # Sequence Parallelism is enabled, to ensure the same behavior as data parallelism
            dp_indices_dict = {}  # {rank: indices_list}
            all_indices_dict = {}  # {rank: all_indices}

            for i in self.corresponding_ranks:
                dp_indices_list = []
                for idx, indices in enumerate(indices_list):
                    dp_indices_list.append(
                        indices[i * self.per_replica_samples[idx] : (i + 1) * self.per_replica_samples[idx]]
                    )

                random.seed(self.seed + self.epoch)
                for indice in range(len(dp_indices_list)):
                    random.shuffle(dp_indices_list[indice])

                dp_indices_dict[i] = dp_indices_list.copy()

            for rank, dp_indices_list in dp_indices_dict.items():
                dp_indices_list = sorted(dp_indices_list, key=lambda x: -len(x))
                dp_all_indices = [-1] * self.num_samples
                indices_available = list(range(self.num_samples))

                for indice in dp_indices_list:

                    original_indices = range(len(indice))
                    transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]

                    mapped_indices = [indices_available[idx] for idx in transformed_indices]
                    # update indices_available
                    for idx in reversed(transformed_indices):
                        del indices_available[idx]
                    for i, idx in enumerate(mapped_indices):
                        dp_all_indices[idx] = indice[i]

                all_indices_dict[rank] = dp_all_indices

            # Interleaving Merge
            merged_indices = []
            interleaved_indices = []
            for item_idx in range(len(all_indices_dict[self.corresponding_ranks[0]])):
                for rank in self.corresponding_ranks:
                    interleaved_indices.append(all_indices_dict[rank][item_idx])
            merged_indices.append(interleaved_indices)

            all_indices = merged_indices[0]
        else:
            # let's first do subsample
            for idx, indices in enumerate(indices_list):
                indices_list[idx] = indices[
                    self.rank * self.per_replica_samples[idx] : (self.rank + 1) * self.per_replica_samples[idx]
                ]

            random.seed(self.seed + self.epoch)
            for indice in range(len(indices_list)):
                random.shuffle(indices_list[indice])

            indices_list = sorted(indices_list, key=lambda x: -len(x))
            all_indices = [-1] * self.num_samples
            indices_available = list(range(self.num_samples))

            for indice in indices_list:

                original_indices = range(len(indice))
                transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]

                mapped_indices = [indices_available[idx] for idx in transformed_indices]
                # update indices_available
                for idx in reversed(transformed_indices):
                    del indices_available[idx]
                for i, idx in enumerate(mapped_indices):
                    all_indices[idx] = indice[i]
        assert -1 not in all_indices
        return iter(all_indices)


class LongVILADistributedSampler(VILADistributedSampler):
    """This class is implemented by Yukang Chen."""

    def __iter__(self):
        def batch_shuffle(indices):
            batch_indices = list(range(indices[0] // self.batch_size, indices[-1] // self.batch_size + 1))
            random.shuffle(batch_indices)
            indices_shuffled = [
                batch_indices[i // self.batch_size] * self.batch_size + index % self.batch_size
                for i, index in enumerate(indices)
            ]
            return indices_shuffled

        indices = list(range(len(self.dataset)))

        # 1. split the full indices first (note: without drop last at this moment)
        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(
                indices[sum(self.org_sample_len_list[:i]) : sum(self.org_sample_len_list[:i]) + self.total_samples[i]]
            )

        assert sum([len(indices) for indices in indices_list]) == self.total_size, (
            sum([len(indices) for indices in indices_list]),
            self.total_size,
        )

        if self.sp_degree > 1:  # Sequence Parallelism is enabled, to ensure the same behavior as data parallelism
            dp_indices_dict = {}  # {rank: indices_list}
            all_indices_dict = {}  # {rank: all_indices}

            for i in self.corresponding_ranks:
                dp_indices_list = []
                for idx, indices in enumerate(indices_list):
                    dp_indices_list.append(
                        indices[i * self.per_replica_samples[idx] : (i + 1) * self.per_replica_samples[idx]]
                    )

                random.seed(self.seed + self.epoch)
                for indice in range(len(dp_indices_list)):
                    batch_shuffle(dp_indices_list[indice])

                dp_indices_dict[i] = dp_indices_list.copy()

            for rank, dp_indices_list in dp_indices_dict.items():
                dp_indices_list = sorted(dp_indices_list, key=lambda x: -len(x))
                dp_all_indices = [-1] * self.num_samples
                indices_available = list(range(self.num_samples))

                for indice in dp_indices_list:

                    original_indices = range(len(indice))
                    transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]

                    mapped_indices = [indices_available[idx] for idx in transformed_indices]
                    # update indices_available
                    for idx in reversed(transformed_indices):
                        del indices_available[idx]
                    for i, idx in enumerate(mapped_indices):
                        dp_all_indices[idx] = indice[i]

                all_indices_dict[rank] = dp_all_indices

            # Interleaving Merge
            merged_indices = []
            interleaved_indices = []
            for item_idx in range(len(all_indices_dict[self.corresponding_ranks[0]])):
                for rank in self.corresponding_ranks:
                    interleaved_indices.append(all_indices_dict[rank][item_idx])
            merged_indices.append(interleaved_indices)

            all_indices = merged_indices[0]
        else:
            # let's first do subsample
            for idx, indices in enumerate(indices_list):
                indices_list[idx] = indices[
                    self.rank * self.per_replica_samples[idx] : (self.rank + 1) * self.per_replica_samples[idx]
                ]

            random.seed(self.seed + self.epoch)
            for indice in range(len(indices_list)):
                batch_shuffle(indices_list[indice])

            indices_list = sorted(indices_list, key=lambda x: -len(x))
            all_indices = [-1] * self.num_samples
            indices_available = list(range(self.num_samples))
            for indice in indices_list:
                original_indices = range(len(indice))
                transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]
                mapped_indices = [indices_available[idx] for idx in transformed_indices]
                # update indices_available
                for idx in reversed(transformed_indices):
                    del indices_available[idx]
                for i, idx in enumerate(mapped_indices):
                    all_indices[idx] = indice[i]
        assert -1 not in all_indices
        return iter(all_indices)

def get_length_grouped_batches(
    lengths: List[int],
    batch_size: int,
    world_size: int,
    generator=None,
    merge: bool = True,
) -> List:

    N = len(lengths)
    M = world_size * batch_size
    if N < M:
        # fallback: just random permute everything
        idx = np.arange(N)
        if generator is not None:
            seed = generator.initial_seed()
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        rng.shuffle(idx)
        if merge:
            return idx.tolist()
        else:
            # one megabatch only
            out = [idx.tolist()]
            # pad to world_size empty lists if needed
            return [out + [[]] * (world_size - 1)]

    # 1) build RNG
    if generator is not None:
        seed = generator.initial_seed()
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()

    # 2) keys for lexsort: primary = -length, secondary = random
    lengths_arr = np.array(lengths, dtype=np.int64)
    key_length = -lengths_arr
    key_rand = rng.permutation(N)

    # 3) single global lexsort (last key is primary)
    sorted_idx = np.lexsort((key_rand, key_length))

    # 4) trim to full megabatches
    num_mb = len(sorted_idx) // M
    trimmed = sorted_idx[: num_mb * M]

    # 5) reshape to [num_mb, M]
    mb = trimmed.reshape(num_mb, M)

    # 6) optional shuffle of whole megabatches
    rng.shuffle(mb)

    # 7) split each row into [world_size, batch_size]
    mb = mb.reshape(num_mb, world_size, batch_size)

    if merge:
        # flatten in order megabatch → replica → sample
        return mb.reshape(-1).tolist()
    else:
        # build nested Python lists: [ [ [..], [..], … ],  … ]
        return [
            [mb[i, r].tolist() for r in range(world_size)]
            for i in range(num_mb)
        ]


# def get_length_grouped_batches(
#     lengths: List[int],
#     batch_size: int,
#     world_size: int,
#     generator=None,
#     merge: bool = True,
# ) -> List:
#     """
#     Create length-grouped megabatches.
    
#     First, a random permutation of indices is computed. Then we split
#     into megabatches of size (world_size * batch_size) and sort each
#     megabatch by descending length. Finally, each megabatch is split
#     into `world_size` chunks (one per replica).

#     If merge is True, a flat list is returned; if False, the nested
#     structure is kept.
#     """
#     indices = torch.randperm(len(lengths), generator=generator)
#     megabatch_size = world_size * batch_size
#     # Partition indices into megabatches
#     megabatches = [
#         indices[i : i + megabatch_size].tolist()
#         for i in range(0, len(lengths), megabatch_size)
#     ]
#     # Within each megabatch, sort indices in descending order of length.
#     sorted_megabatches = [
#         sorted(megabatch, key=lambda i: lengths[i], reverse=True)
#         for megabatch in megabatches
#     ]
#     # Split each sorted megabatch evenly among replicas.
#     split_megabatches = [
#         split_to_even_chunks(megabatch, lengths, world_size)
#         for megabatch in sorted_megabatches
#     ]
#     if merge:
#         # Flatten into a single list.
#         return [i for megabatch in split_megabatches for batch in megabatch for i in batch]
#     else:
#         # Return the nested structure: list of megabatches, each containing a list (of length world_size) of batches.
#         return split_megabatches

class LengthGroupedVILADistributedSampler(DistributedSampler):
    """
    A sampler that groups examples by (approximate) length and then
    distributes them across replicas following VILA’s accumulation logic.
    
    Parameters:
      - dataset: the dataset to sample from.
      - batch_size: batch size per replica.
      - lengths: a list of lengths (one per example in the dataset).
      - num_replicas: total number of distributed replicas (if not provided,
        will be inferred from torch.distributed).
      - rank: the rank of the current process.
      - shuffle: whether to shuffle groups.
      - seed: base random seed.
      - drop_last: whether to drop the tail of incomplete megabatches (set True).
      - sp_degree: sequence-parallel degree.
      - gradient_accumulation_steps: used for scaling the effective batch size.
      - group_by_modality: if True, you might call a different grouping function.
      - generator: optional torch.Generator for determinism.
      - force_accumulation: whether to force the VILA accumulation ordering.
    """
    def __init__(
        self,
        dataset,
        batch_size: int,
        lengths: List[int],
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        sp_degree: int = 1,
        gradient_accumulation_steps: int = 1,
        group_by_modality: bool = True,
        generator=None,
        force_accumulation: bool = True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank,
                         shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.dataset = dataset
        self.batch_size = batch_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality
        self.sp_degree = max(1, sp_degree)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.force_accumulation = force_accumulation
        self.seed = seed
        self.epoch = 0  # This should be updated externally at each epoch.

        self.world_size = self.num_replicas  # from DistributedSampler

        self.bs_divisible_by_sp = (batch_size % self.sp_degree == 0)
        if self.sp_degree > 1:
            # Get sequence parallelism group info.
            PROCESS_GROUP_MANAGER = get_pg_manager()  # Must be implemented.
            self.dp_rank = PROCESS_GROUP_MANAGER.dp_rank
            self.dp_num_replicas = self.num_replicas // self.sp_degree
            self.corresponding_ranks = list(range(self.dp_rank * self.sp_degree, (self.dp_rank + 1) * self.sp_degree))
        else:
            self.dp_rank = self.rank
            self.dp_num_replicas = self.num_replicas

        # Compute the number of full megabatches (each of size world_size * batch_size).
        megabatch_size = self.world_size * self.batch_size
        num_full_megabatches = len(self.dataset) // megabatch_size
        # For each full megabatch, each replica gets batch_size examples.
        self.num_samples = num_full_megabatches * self.batch_size

    def __len__(self) -> int:
        # When using sequence parallelism, the effective number may be scaled.
        return self.num_samples * (self.sp_degree if self.sp_degree > 1 else 1)

    def __iter__(self):
        # Get the nested list of length-grouped batches.
        # Each element in "megabatches" is a list of length world_size, one per replica.
        megabatches = get_length_grouped_batches(
            self.lengths,
            self.batch_size,
            self.world_size,
            generator=self.generator,
            merge=False,
        )
        # For each megabatch, select the batch corresponding to this replica.
        indices_list = []
        for megabatch in megabatches:
            if self.rank < len(megabatch):
                indices_list.append(megabatch[self.rank])
        total_samples = sum(len(lst) for lst in indices_list)

        if self.sp_degree > 1 and self.bs_divisible_by_sp:
            # --- Sequence Parallelism branch ---
            # For each of the corresponding sequence-parallel ranks, split each batch.
            dp_indices_dict = {}
            all_indices_dict = {}
            for r in self.corresponding_ranks:
                dp_indices_list = []
                for lst in indices_list:
                    # Split each list into sp_degree equal parts.
                    part_size = len(lst) // self.sp_degree
                    dp_indices_list.append(lst[r * part_size : (r + 1) * part_size])
                random.seed(self.seed + self.epoch)
                for sublist in dp_indices_list:
                    random.shuffle(sublist)
                dp_indices_dict[r] = dp_indices_list.copy()
            # Now, for each sequence-parallel rank, remap the indices.
            for r, dp_list in dp_indices_dict.items():
                # Sort the sublists by descending length.
                dp_list = sorted(dp_list, key=lambda x: -len(x))
                num_samples_r = sum(len(x) for x in dp_list)
                dp_all_indices = [-1] * num_samples_r
                indices_available = list(range(num_samples_r))
                for sublist in dp_list:
                    n = len(sublist)
                    transformed_indices = [i * len(indices_available) // n for i in range(n)]
                    mapped_indices = [indices_available[j] for j in transformed_indices]
                    for j in sorted(transformed_indices, reverse=True):
                        del indices_available[j]
                    for i, pos in enumerate(mapped_indices):
                        dp_all_indices[pos] = sublist[i]
                all_indices_dict[r] = dp_all_indices
            # Interleave the indices from all sequence-parallel ranks.
            merged_indices = []
            # Assumes each dp_all_indices list is of the same length.
            interleaved_length = len(next(iter(all_indices_dict.values())))
            for i in range(interleaved_length):
                for r in self.corresponding_ranks:
                    merged_indices.append(all_indices_dict[r][i])
            final_indices = merged_indices
        else:
            # --- Non-sequence-parallel branch ---
            random.seed(self.seed + self.epoch)
            for sublist in indices_list:
                random.shuffle(sublist)
            # Sort the groups by descending length.
            indices_list = sorted(indices_list, key=lambda x: -len(x))
            dp_all_indices = [-1] * total_samples
            indices_available = list(range(total_samples))
            for sublist in indices_list:
                n = len(sublist)
                transformed_indices = [i * len(indices_available) // n for i in range(n)]
                mapped_indices = [indices_available[j] for j in transformed_indices]
                for j in sorted(transformed_indices, reverse=True):
                    del indices_available[j]
                for i, pos in enumerate(mapped_indices):
                    dp_all_indices[pos] = sublist[i]
            final_indices = dp_all_indices

        assert -1 not in final_indices, "Some indices were not assigned properly."
        return iter(final_indices)


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class VILADPOTrainer(DPOTrainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Always using Jason's sampler.
        sample_len_list = self.args.sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        num_replicas = self.args.world_size
        rank = self.args.process_index
        return VILADistributedSampler(
            self.train_dataset,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            batch_size=self.args.train_batch_size,
            sample_len_list=sample_len_list,
            sp_degree=self.args.seq_parallel_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.eval_dataset is None or not has_length(self.eval_dataset):
            return None

        # Always using Jason's sampler.
        sample_len_list = self.args.eval_sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return VILADistributedSampler(
            eval_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.eval_batch_size,
            sample_len_list=sample_len_list,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if 0:  # self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            # TODO(ligeng): fix save_model for multi-node training on large models (e.g., Llama-70b)
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)


class LLaVATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = True

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        print('AF3 sampler')
        sample_len_list = self.args.sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        num_replicas = self.args.world_size
        rank = self.args.process_index
        longvila_sampler = self.args.longvila_sampler

        if self.args.group_by_modality_length:
            sampler = LengthGroupedVILADistributedSampler
            if not isinstance(self.train_dataset, ConcatDataset):
                lengths = self.train_dataset.modality_lengths
            else:
                lengths = []
                for d in self.train_dataset.datasets:
                    lengths += d.modality_lengths

            return sampler(
                self.train_dataset,
                lengths=lengths,
                num_replicas=num_replicas,
                rank=rank,
                seed=seed,
                batch_size=self.args.train_batch_size,
                sp_degree=self.args.seq_parallel_size,
                gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                group_by_modality=True
            )
        else:
            sampler = LongVILADistributedSampler if longvila_sampler else VILADistributedSampler
            return sampler(
            self.train_dataset,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            batch_size=self.args.train_batch_size,
            sample_len_list=sample_len_list,
            sp_degree=self.args.seq_parallel_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            )


    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.eval_dataset is None or not has_length(self.eval_dataset):
            return None

        sample_len_list = self.args.eval_sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return VILADistributedSampler(
            eval_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.eval_batch_size,
            sample_len_list=sample_len_list,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

    def _inner_training_loop(self, batch_size: Optional[int] = None, *args, **kwargs):
        # NOTE(zhijianl): In the latest transformers, if the batch size in the training arguments differs from
        # the one in the training state, the batch size from the state is used by default. This can be
        # problematic when resuming with different batch sizes or gradient accumulation steps. To prevent this,
        # we enforce using the batch size specified in the training arguments.
        batch_size = self.args.train_batch_size
        return super()._inner_training_loop(batch_size, *args, **kwargs)

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            elif self.args.vision_tower_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "vision_tower" in name]
                # projector_lora_A_parameters = [name for name in projector_parameters if "lora_A" in name]
                # projector_lora_B_parameters = [name for name in projector_parameters if "lora_B" in name]
                # other_lora_A_parameters = [name for name in opt_model.named_parameters() if "lora_A" in name and name not in projector_parameters]
                # other_lora_B_parameters = [name for name in opt_model.named_parameters() if "lora_B" in name and name not in projector_parameters]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_tower_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_tower_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if 0:  # self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            # TODO(ligeng): fix save_model for multi-node training on large models (e.g., Llama-70b)
            state_dict = self.model.state_dict()

        if self.args.lora_enable:
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                non_lora_state_dict,
                os.path.join(output_dir, "non_lora_trainables.bin"),
            )
            # config
            self.model._name_or_path = output_dir
            self.model.architectures = [self.model.__class__.__name__]
            self.model.config.save_pretrained(output_dir)

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)

        if self.args.debug_e2e and self.control.should_training_stop:

            # Only save log history if the current process is rank 0
            if dist.get_rank() == 0:
                with open(f"{self.args.output_dir}/log_history.json", "w") as f:
                    json.dump(self.state.log_history, f, indent=4)

        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
