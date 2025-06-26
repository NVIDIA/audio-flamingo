# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import string
import yaml
from copy import deepcopy

import torch
from transformers import AutoTokenizer, set_seed 
set_seed(0)

from data import AudioTextDataProcessor
from src.factory import create_model_and_transforms


def prepare_tokenizer(model_config):
    tokenizer_path = model_config['tokenizer_path']
    cache_dir = model_config['cache_dir']
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=False,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<audio>", "<|endofchunk|>"]}
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    if text_tokenizer.sep_token is None:
        text_tokenizer.add_special_tokens({"sep_token": "<SEP>"})
    return text_tokenizer


def prepare_model(model_config, clap_config, checkpoint_path, device_id=0):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable the tokenizer parallelism warning
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=False,
        gradient_checkpointing=False,
        freeze_lm_embeddings=False,
    )
    model.eval()
    model = model.to(device_id)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict, False)

    return model


def inference(model, tokenizer, item, processed_item, inference_kwargs, device_id=0):
    filename, audio_clips, audio_embed_mask, input_ids, attention_mask = processed_item
    audio_clips = audio_clips.to(device_id, dtype=None, non_blocking=True)
    audio_embed_mask = audio_embed_mask.to(device_id, dtype=None, non_blocking=True)
    input_ids = input_ids.to(device_id, dtype=None, non_blocking=True).squeeze()

    media_token_id = tokenizer.encode("<audio>")[-1]
    eoc_token_id = tokenizer.encode("<|endofchunk|>")[-1]
    sep_token_id = tokenizer.sep_token_id
    eos_token_id = tokenizer.eos_token_id
    
    outputs = model.generate(
        audio_x=audio_clips.unsqueeze(0),
        audio_x_mask=audio_embed_mask.unsqueeze(0),
        lang_x=input_ids.unsqueeze(0),
        eos_token_id=eos_token_id,
        max_new_tokens=128,
        **inference_kwargs,
    )

    outputs_decoded = [
        tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '') for output in outputs
    ]

    return outputs_decoded


def main(config_file, data_root, checkpoint_path, items, inference_kwargs, is_dialogue=False, do_dialogue_last=False):
    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    clap_config = config['clap_config']
    model_config = config['model_config']

    text_tokenizer = prepare_tokenizer(model_config)
    DataProcessor = AudioTextDataProcessor(
        data_root=data_root,
        clap_config=clap_config,
        tokenizer=text_tokenizer,
        max_tokens=512,
    )

    print("===== checkpoint_path: {} =====".format(checkpoint_path))
    model = prepare_model(
        model_config=model_config, 
        clap_config=clap_config, 
        checkpoint_path=checkpoint_path
    )

    for item in items:
        print('----- File: {} -----'.format(item['name']))

        if is_dialogue:
            staged_item = deepcopy(item)
            if do_dialogue_last:
                if "assistant" in staged_item['dialogue'][-1]:
                    del staged_item['dialogue'][-1]["assistant"]

                processed_item = DataProcessor.process(staged_item)
                outputs = inference(
                    model, text_tokenizer, staged_item, processed_item,
                    inference_kwargs,
                )

                print('Prompt:', item['dialogue'][-1]['user'])
                print('Audio Flamingo:', outputs)

            else:
                
                staged_item['dialogue'] = []
                for each_round in item['dialogue'] :
                    staged_item['dialogue'].append({'user': each_round['user']})

                    processed_item = DataProcessor.process(staged_item)
                    outputs = inference(
                        model, text_tokenizer, staged_item, processed_item,
                        inference_kwargs,
                    )[0]

                    staged_item['dialogue'][-1]['assistant'] = outputs

                    print('Prompt:', each_round['user'])
                    print('Audio Flamingo:', outputs)

        else:
            processed_item = DataProcessor.process(item)
            outputs = inference(
                model, text_tokenizer, item, processed_item,
                inference_kwargs,
            )

            print('Prompt:', item['prompt'])
            print('Audio Flamingo:', outputs)
    
    return outputs


if __name__ == "__main__":
    data_root = 'YOUR_DATA_ROOT_DIR/datasets'

    # ---------- foundation model ---------- #

    config_file = 'configs/foundation.yaml'
    checkpoint_path = "YOUR_CHECKPOINT_ROOT_DIR/foundation.pt"

    inference_kwargs = {
        "do_sample": True,
        "top_k": 30,
        "top_p": 0.95,
        "num_return_sequences": 1
    }

    items = [
        # captioning
        {'name': 'audiocaps/audio/test/50531.flac', 'prefix': 'The task is audio captioning.', 'prompt': 'Describe the sound in a sentence.'},
        {'name': 'audiocaps/audio/test/50455.flac', 'prefix': 'The task is audio captioning.', 'prompt': 'Describe the sound in a sentence.'},
        {'name': 'audiocaps/audio/test/50109.flac', 'prefix': 'The task is audio captioning.', 'prompt': 'Describe the sound in a sentence.'},
        # question answering
        {'name': 'Clotho-AQA/audio_files/river_mouth3.wav', 'prefix': 'The task is audio question answering.', 'prompt': 'Please answer this question: Are there waves?\nOPTIONS:\nyes.\nno.'},
        {'name': 'Clotho-AQA/audio_files/fdv_orage_26082011.wav', 'prefix': 'The task is audio question answering.', 'prompt': 'Please answer this question: Is outside sunny?\nOPTIONS:\nyes.\nno.'},
        {'name': 'Clotho-AQA/audio_files/Creaking pier.wav', 'prefix': 'The task is audio question answering.', 'prompt': 'Please answer this question: What type of animal is making the light sound in the background?'},
        {'name': 'Clotho-AQA/audio_files/quick walk.wav', 'prefix': 'The task is audio question answering.', 'prompt': 'Please answer this question: what activity is the person doing?'},
        # classification
        {'name': 'CochlScene/Test/Park/Park_user0807_14976869_001.wav', 'prefix': 'The task is scene classification.', 'prompt': 'classify this sound.\nOPTIONS:\n - bus.\n - cafe.\n - car.\n - crowdedindoor.\n - elevator.\n - kitchen.\n - park.\n - residentialarea.\n - restaurant.\n - restroom.\n - street.\n - subway.\n - subwaystation.'},
        {'name': 'CochlScene/Test/Cafe/Cafe_user0802_14892039_004.wav', 'prefix': 'The task is scene classification.', 'prompt': 'classify this sound.\nOPTIONS:\n - bus.\n - cafe.\n - car.\n - crowdedindoor.\n - elevator.\n - kitchen.\n - park.\n - residentialarea.\n - restaurant.\n - restroom.\n - street.\n - subway.\n - subwaystation.'},
        {'name': 'CochlScene/Test/Bus/Bus_user0419_14867701_002.wav', 'prefix': 'The task is scene classification.', 'prompt': 'classify this sound.\nOPTIONS:\n - bus.\n - cafe.\n - car.\n - crowdedindoor.\n - elevator.\n - kitchen.\n - park.\n - residentialarea.\n - restaurant.\n - restroom.\n - street.\n - subway.\n - subwaystation.'},
        {'name': 'FSD50k/44khz/eval/37199.wav', 'prefix': 'The task is event classification.', 'prompt': 'describe this sound in the order from specific to general.'},
        {'name': 'FSD50k/44khz/eval/210276.wav', 'prefix': 'The task is event classification.', 'prompt': 'describe this sound in the order from specific to general.'},
        {'name': 'FSD50k/44khz/eval/53447.wav', 'prefix': 'The task is event classification.', 'prompt': 'describe this sound in the order from specific to general.'}
    ]

    main(config_file, data_root, checkpoint_path, items, inference_kwargs, is_dialogue=False)

    # ---------- chat model ---------- #

    config_file = 'configs/chat.yaml'
    checkpoint_path = "YOUR_CHECKPOINT_ROOT_DIR/chat.pt"
    
    inference_kwargs = {
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 1  # must be 1 output for dialogues
    }

    items = [
        {
            'name': "audioset/eval_segments/22khz/Y0bRUkLsttto.wav",
            'prefix': "The task is dialog.",
            'dialogue': [
                {"user": "What genre does this music belong to?"}, 
                {"user": "Can you describe the vocals in this track?"}
            ]
        },
        {
            'name': "audioset/eval_segments/22khz/YXyktNsq4SZU.wav",
            'prefix': "The task is dialog.",
            'dialogue': [
                {"user": "Can you briefly explain what you hear in the audio?"}, 
                {"user": "Is it just one car or are there multiple cars?"},
                {"user": "What can you tell me about the car's engine?"},
                {"user": "Does it sound like they are racing?"}
            ]
        }
    ]

    main(config_file, data_root, checkpoint_path, items, inference_kwargs, is_dialogue=True, do_dialogue_last=False)

