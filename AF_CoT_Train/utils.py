# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import time

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from transformers import AutoModelForCausalLM, AutoTokenizer
from qwen_omni_utils import process_mm_info
import torch


def inference_qwen2_5_omni(model, processor, audio_path, prompt, sys_prompt="You are a sound understanding and classification model."):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {"role": "user", "content": [
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": prompt},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(**inputs, use_audio_in_video=True, return_audio=False, thinker_max_new_tokens=256, thinker_do_sample=False)

    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text[0].split("assistant\n")[-1]


def inference_qwen3(model, tokenizer, prompt, record_time=False):
    # 30s for 8B models
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    if record_time:
        time_start = time.time()

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=4096  # 32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    if record_time:
        time_end = time.time()
        print("generation time: {:.1f} seconds".format(time_end - time_start))

    return {
        "thinking content:": thinking_content,
        "content:": content
    }


def load_models(model_name="Qwen/Qwen3-8B", omni_model_name="Qwen/Qwen2.5-Omni-7B"):
    # "Qwen/Qwen3-14B" will OOM with omni model
    model_omni = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        omni_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="flash_attention_2",
    )
    model_omni.disable_talker()
    processor_omni = Qwen2_5OmniProcessor.from_pretrained(omni_model_name)

    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if "FP8" not in model_name else "auto",
        device_map="cuda",
    )

    return (model, tokenizer, model_omni, processor_omni)
