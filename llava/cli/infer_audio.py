# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

import argparse
import importlib.util
import json
import os

from pydantic import BaseModel
from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video, Sound
from llava.model.configuration_llava import JsonSchemaResponseFormat, ResponseFormat
from peft import PeftModel
import torch

def get_schema_from_python_path(path: str) -> str:
    schema_path = os.path.abspath(path)
    spec = importlib.util.spec_from_file_location("schema_module", schema_path)
    schema_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(schema_module)

    # Get the Main class from the loaded module
    Main = schema_module.Main
    assert issubclass(
        Main, BaseModel
    ), f"The provided python file {path} does not contain a class Main that describes a JSON schema"
    return Main.schema_json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-base", "-m", type=str, required=True)
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--text", type=str)
    parser.add_argument("--media", type=str, nargs="+")
    parser.add_argument("--json-mode", action="store_true")
    parser.add_argument("--think-mode", action="store_true")
    parser.add_argument("--json-schema", type=str, default=None)
    args = parser.parse_args()

    # Convert json mode to response format
    if not args.json_mode:
        response_format = None
    elif args.json_schema is None:
        response_format = ResponseFormat(type="json_object")
    else:
        schema_str = get_schema_from_python_path(args.json_schema)
        print(schema_str)
        response_format = ResponseFormat(type="json_schema", json_schema=JsonSchemaResponseFormat(schema=schema_str))

    # Load model
    from huggingface_hub import snapshot_download

    # ---------------------------------
    # SINGLE-TURN MODEL SETUP
    # ---------------------------------
    model_path = snapshot_download(args.model_base)
    model_think = os.path.join(model_path, 'stage35')

    model = llava.load(model_path, device_map=None)
    if args.think_mode:
        model = PeftModel.from_pretrained(
            model,
            model_think,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    model = model.to("cuda")
    # Set conversation mode
    clib.default_conversation = clib.conv_templates[args.conv_mode].copy()

    # Prepare multi-modal prompt
    prompt = []
    if args.media is not None:
        for media in args.media or []:
            if any(media.endswith(ext) for ext in [".wav",".mp3", ".flac"]):
                media = Sound(media)
            else:
                raise ValueError(f"Unsupported media type: {media}")
            prompt.append(media)
    if args.text is not None:
        prompt.append(args.text)

    # Generate response
    response = model.generate_content(prompt, response_format=response_format)
    print(colored(response, "cyan", attrs=["bold"]))


if __name__ == "__main__":
    main()
