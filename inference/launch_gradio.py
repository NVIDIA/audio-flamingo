# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

import os
import sys
import argparse
import random
import pickle
import string
import yaml

try:
    import gradio as gr
except ImportError:
    os.system('pip install gradio')
    import gradio as gr

import librosa
from pydub import AudioSegment
import soundfile as sf

import numpy as np
import torch
import laion_clap

from inference_examples import prepare_tokenizer, prepare_model, inference
from data import AudioTextDataProcessor


def load_laionclap(checkpoint='630k-audioset-fusion-best.pt'):
    if checkpoint in ['630k-audioset-best.pt', '630k-best.pt', '630k-audioset-fusion-best.pt', '630k-fusion-best.pt']:
        amodel = 'HTSAT-tiny'
    elif checkpoint in ['music_audioset_epoch_15_esc_90.14.pt', 'music_speech_epoch_15_esc_89.25.pt', 'music_speech_audioset_epoch_15_esc_89.98.pt']:
        amodel = 'HTSAT-base'
    else:
        raise NotImplementedError
    
    model = laion_clap.CLAP_Module(
        enable_fusion=('fusion' in checkpoint.lower()), 
        amodel=amodel
    ).cuda()
    model.load_ckpt(ckpt=os.path.join(
        DATA_ROOT_DIR,
        'audio-flamingo-data/laion-clap-pretrained/laion_clap',
        checkpoint
    ))
    model.eval()
    return model


def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def load_audio(file_path, target_sr=44100, duration=33.25, start=0.0):
    if file_path.endswith('.mp3'):
        audio = AudioSegment.from_file(file_path)
        if len(audio) > (start + duration) * 1000:
            audio = audio[start * 1000:(start + duration) * 1000]

        if audio.frame_rate != target_sr:
            audio = audio.set_frame_rate(target_sr)

        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        data = np.array(audio.get_array_of_samples())
        if audio.sample_width == 2:
            data = data.astype(np.float32) / np.iinfo(np.int16).max
        elif audio.sample_width == 4:
            data = data.astype(np.float32) / np.iinfo(np.int32).max
        else:
            raise ValueError("Unsupported bit depth: {}".format(audio.sample_width))

    else:
        with sf.SoundFile(file_path) as audio:
            original_sr = audio.samplerate
            channels = audio.channels

            max_frames = int((start + duration) * original_sr)

            audio.seek(int(start * original_sr))
            frames_to_read = min(max_frames, len(audio))
            data = audio.read(frames_to_read)

            if data.max() > 1 or data.min() < -1:
                data = data / max(abs(data.max()), abs(data.min()))
        
        if original_sr != target_sr:
            if channels == 1:
                data = librosa.resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
            else:
                data = librosa.resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
        else:
            if channels != 1:
                data = data.T[0]
    
    if data.min() >= 0:
        data = 2 * data / abs(data.max()) - 1.0
    else:
        data = data / max(abs(data.max()), abs(data.min()))
    return data


@torch.no_grad()
def compute_laionclap_text_audio_sim(audio_file, laionclap_model, outputs):
    try:
        data = load_audio(audio_file, target_sr=48000)
    
    except Exception as e:
        print(audio_file, 'unsuccessful due to', e)
        return None
    
    # compute audio embedding
    audio_data = data.reshape(1, -1)
    audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float().cuda()
    audio_embed = laionclap_model.get_audio_embedding_from_data(x=audio_data_tensor, use_tensor=True)

    # compute text embedding
    text_embed = laionclap_model.get_text_embedding(outputs, use_tensor=True)

    # compute cosine similarities
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cos_similarity = cos(audio_embed.repeat(text_embed.shape[0], 1), text_embed)
    return cos_similarity.squeeze().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default='configs/chat.yaml')
    parser.add_argument("--checkpoint_path", type=str)
    args = parser.parse_args()

    global DATA_ROOT_DIR
    DATA_ROOT_DIR = "YOUR_DATA_ROOT_DIR"

    config_file = args.config_file
    checkpoint_path = args.checkpoint_path

    inference_kwargs = {
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 10
    }

    config = yaml.load(open(config_file), Loader=yaml.FullLoader)
    clap_config = config['clap_config']
    model_config = config['model_config']

    text_tokenizer = prepare_tokenizer(model_config)
    DataProcessor = AudioTextDataProcessor(
        data_root='./',
        clap_config=clap_config,
        tokenizer=text_tokenizer,
        max_tokens=512,
    )

    laionclap_model = load_laionclap('630k-audioset-fusion-best.pt')

    model = prepare_model(
        model_config=model_config, 
        clap_config=clap_config, 
        checkpoint_path=checkpoint_path
    )


    def inference_item(name, prompt):
        item = {
            'name': str(name), 
            'prefix': 'The task is dialog.', 
            'prompt': str(prompt)
        }
        processed_item = DataProcessor.process(item)

        outputs = inference(
            model, text_tokenizer, item, processed_item,
            inference_kwargs,
        )

        laionclap_scores = compute_laionclap_text_audio_sim(
            item["name"],
            laionclap_model,
            outputs
        )

        outputs_joint = [(output, score) for (output, score) in zip(outputs, laionclap_scores)]
        outputs_joint.sort(key=lambda x: -x[1])

        return outputs_joint[0][0] + ' (Laion-CLAP score = {:.2f})'.format(outputs_joint[0][1])


    with gr.Blocks() as ui:
        name = gr.Textbox(label="Audio file path",)
        prompt = gr.Textbox(
            label="Instruction",
            value='What do you hear in this audio?'
        )
        output_text = gr.Textbox(label="Output")

        with gr.Row():
            play_audio_button = gr.Button("Play Audio")
        audio_output = gr.Audio(label="Playback")
        play_audio_button.click(fn=lambda x: x, inputs=name, outputs=audio_output)

        inference_button = gr.Button("Inference")
        inference_button.click(
            fn=inference_item, 
            inputs=[name, prompt],
            outputs=output_text
        )

    ui.launch(server_name="0.0.0.0", server_port=7777)
