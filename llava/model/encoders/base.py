# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

from torch import nn

__all__ = ["BaseEncoder"]


class BaseEncoder(nn.Module):
    def __init__(self, parent: nn.Module) -> None:
        super().__init__()
        self._parent = [parent]

    @property
    def parent(self) -> nn.Module:
        return self._parent[0]
