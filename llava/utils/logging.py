# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

import typing

if typing.TYPE_CHECKING:
    from loguru import Logger
else:
    Logger = None

__all__ = ["logger"]


def __get_logger() -> Logger:
    from loguru import logger

    return logger


logger = __get_logger()
