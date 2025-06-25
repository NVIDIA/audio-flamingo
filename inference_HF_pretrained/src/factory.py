# Copyright (c) 2025 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/mlfoundations/open_flamingo under the MIT license.
#   LICENSE is in incl_licenses directory.

import sys
from copy import deepcopy
from typing import Optional
from contextlib import suppress

import torch
import torchaudio
import numpy as np
from torch import nn
import torchvision.transforms
import torchaudio.transforms as T

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from my_laion_clap.CLAP.src.laion_clap.clap_module.htsat import create_htsat_model

# Attempt to import local modules with fallback
try:
    from .flamingo import Flamingo
    from .flamingo_lm import FlamingoLMMixin
    from .utils import extend_instance
except ImportError:
    from flamingo import Flamingo
    from flamingo_lm import FlamingoLMMixin
    from utils import extend_instance

# Ensure correct module path inclusion
sys.path.append("../")

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def int16_to_float32_torch(x):
    return (x / 32767.0).type(torch.float32)

def float32_to_int16_torch(x):
    x = torch.clamp(x, min=-1., max=1.)
    return (x * 32767.).type(torch.int16)

class CLAPAudioCfp:
    model_type: str = "HTSAT"
    model_name: str = "large"
    sample_rate: int = 16000
    audio_length: int = 1024
    window_size: int = 1024
    hop_size: int = 160
    fmin: int = 50
    fmax: int = 8000
    class_num: int = 527
    mel_bins: int = 64
    clip_samples: int = 160000


class CLAP(nn.Module):
    def __init__(self, clap_config):
        super(CLAP, self).__init__()

        self.clap_config = clap_config

        self.method = clap_config["method"]
        device_id = f'cuda:{torch.cuda.current_device()}'

        if ('finetune' in clap_config) and clap_config['finetune']:
            self.finetune = True 
            print('Finetuning CLAP encoder as well!')
        else:
            self.finetune = False 

        audio_cfg = CLAPAudioCfp()
        enable_fusion = True
        fusion_type = "aff_2d"
        self.nvclap = create_htsat_model(audio_cfg, enable_fusion, fusion_type)
        clap_state_dict = torch.load(clap_config["checkpoint"], map_location = 'cpu')
        clap_state_dict_copy = clap_state_dict['state_dict'].copy()
        for key in list(clap_state_dict['state_dict'].keys()):
            if 'audio' in key:
                clap_state_dict_copy[key.replace('module.audio_branch.','')] = clap_state_dict_copy[key]
                del clap_state_dict_copy[key]
            else:
                del clap_state_dict_copy[key]
        self.nvclap.load_state_dict(clap_state_dict_copy, strict = False)
        self.nvclap = self.nvclap.to(device_id)
        
        for param in self.nvclap.parameters():
            param.requires_grad = self.finetune

        if self.finetune:
            self.nvclap.train()
        else:
            self.nvclap.eval()

        print('loaded NVCLAP model: {}'.format(clap_config["checkpoint"]))
                
    def get_mel(self, audio_data):

        # mel shape: (n_mels, T)
        mel_tf = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            win_length=1024,
            hop_length=160,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,
            onesided=True,
            n_mels=64,
            f_min=50,
            f_max=8000
        ).to(audio_data.device)
        
        mel = mel_tf(audio_data)

        # we use log mel spectrogram as input
        mel = torchaudio.transforms.AmplitudeToDB(top_db=None)(mel)

        return mel.T  # (T, n_mels)

    def get_audio_features(self, sample, audio_data, max_len, data_truncating, data_filling, require_grad=False):

        grad_fn = suppress if require_grad else torch.no_grad
        with grad_fn():
            if len(audio_data) > max_len:
                if data_truncating == "rand_trunc":
                    longer = torch.tensor([True])
                elif data_truncating == "fusion":
                    # fusion
                    mel = self.get_mel(audio_data)
                    # split to three parts
                    chunk_frames = max_len // 160 + 1  # the +1 related to how the spectrogram is computed
                    total_frames = mel.shape[0]
                    if chunk_frames == total_frames:
                        # there is a corner case where the audio length is
                        # larger than max_len but smaller than max_len+hop_size.
                        # In this case, we just use the whole audio.
                        mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                        sample["mel_fusion"] = mel_fusion
                        longer = torch.tensor([False])
                    else:
                        ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
                        if len(ranges[1]) == 0:
                            # if the audio is too short, we just use the first chunk
                            ranges[1] = [0]
                        if len(ranges[2]) == 0:
                            # if the audio is too short, we just use the first chunk
                            ranges[2] = [0]
                        # randomly choose index for each part
                        idx_front = np.random.choice(ranges[0])
                        idx_middle = np.random.choice(ranges[1])
                        idx_back = np.random.choice(ranges[2])
                        # select mel
                        mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
                        mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
                        mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]

                        # shrink the mel
                        mel_shrink = torchvision.transforms.Resize(size=[chunk_frames, 64])(mel[None])[0]
                        # logging.info(f"mel_shrink.shape: {mel_shrink.shape}")

                        # stack
                        mel_fusion = torch.stack([mel_shrink, mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
                        sample["mel_fusion"] = mel_fusion
                        longer = torch.tensor([True])
                else:
                    raise NotImplementedError(
                        f"data_truncating {data_truncating} not implemented"
                    )
                # random crop to max_len (for compatibility)
                overflow = len(audio_data) - max_len
                idx = np.random.randint(0, overflow + 1)
                audio_data = audio_data[idx: idx + max_len]

            else:  # padding if too short
                if len(audio_data) < max_len:  # do nothing if equal
                    if data_filling == "repeatpad":
                        n_repeat = int(max_len / len(audio_data))
                        audio_data = audio_data.repeat(n_repeat)
                        # audio_data = audio_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        # audio_data = F.interpolate(audio_data,size=max_len,mode="bicubic")[0,0,0]
                        audio_data = F.pad(
                            audio_data,
                            (0, max_len - len(audio_data)),
                            mode="constant",
                            value=0,
                        )
                    elif data_filling == "pad":
                        audio_data = F.pad(
                            audio_data,
                            (0, max_len - len(audio_data)),
                            mode="constant",
                            value=0,
                        )
                    elif data_filling == "repeat":
                        n_repeat = int(max_len / len(audio_data))
                        audio_data = audio_data.repeat(n_repeat + 1)[:max_len]
                    else:
                        raise NotImplementedError(
                            f"data_filling {data_filling} not implemented"
                        )
                if data_truncating == 'fusion':
                    mel = self.get_mel(audio_data)
                    mel_fusion = torch.stack([mel, mel, mel, mel], dim=0)
                    sample["mel_fusion"] = mel_fusion
                longer = torch.tensor([False])

        sample["longer"] = longer
        sample["waveform"] = audio_data

        return sample


    def load_audio(self, clips):

        # waveform, sr = torchaudio.load(filename)
        # waveform = torchaudio.functional.resample(waveform, orig_freq=self.clap_config['sampling_rate'], new_freq=16000)
        processed_clips = []
        for clip in clips:
            audio_data = int16_to_float32_torch(float32_to_int16_torch(clip))
            sample = self.get_audio_features({}, audio_data, 160000, "fusion", "repeatpad")
            processed_clips.append(sample)

        waveforms = {}
        waveforms["mel_fusion"] = torch.stack([item["mel_fusion"] for item in processed_clips], dim=0)
        waveforms["longer"] = torch.stack([item["longer"] for item in processed_clips], dim=0)
        waveforms["waveform"] = torch.stack([item["waveform"] for item in processed_clips], dim=0)

        return waveforms


    def forward(self, audio_clips):
        
        # It will handle various segments, 1 audio will have various segments [B X n_segments X time]
        # expand batch dimension during inference
        if len(audio_clips.shape) == 2:
            audio_clips = audio_clips.unsqueeze(0)
        assert len(audio_clips.shape) == 3

        audio_embeds = []
        for audio_clip in audio_clips:
            audio = self.load_audio(audio_clip)
            audio_embed = self.nvclap(audio) #.reshape(-1, self.clap_config["audio_embed_dim"])
            audio_embeds.append(audio_embed)

        audio_embeds = torch.stack(audio_embeds, dim=0)
        # audio_embeds.requires_grad = self.finetune

        return audio_embeds
    

def create_model_and_transforms(
    clap_config: dict,
    lang_encoder_path: str,
    tokenizer_path: str,
    audio_transformer_kwargs: dict,
    cross_attn_every_n_layers: int = 1,
    use_local_files: bool = False,
    decoder_layers_attr_name: str = None,
    freeze_lm_embeddings: bool = False,
    unfreeze_full_lm: bool = False,
    cache_dir: Optional[str] = None,
    **flamingo_kwargs,
):
    clap = CLAP(clap_config)

    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<audio>", "<|endofchunk|>", "<|PAD_TOKEN|>"]}
    )

    text_tokenizer.pad_token = None
    text_tokenizer.pad_token_id = None

    text_tokenizer.pad_token = "<|PAD_TOKEN|>"
    text_tokenizer.pad_token_id = text_tokenizer.encode("<|PAD_TOKEN|>")[-1]

    if text_tokenizer.sep_token is None:
        text_tokenizer.add_special_tokens({"sep_token": "<SEP>"})

    lang_encoder = AutoModelForCausalLM.from_pretrained(
        lang_encoder_path,
        local_files_only=use_local_files,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    extend_instance(lang_encoder, FlamingoLMMixin)

    if decoder_layers_attr_name is None:
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))
    
    if ('finetune' in clap_config) and clap_config['finetune']:
        unfreeze_clap = True 
    else:
        unfreeze_clap = False 

    model = Flamingo(
        clap,
        unfreeze_clap,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<audio>")[-1],
        text_tokenizer.sep_token_id,
        clap_embed_dim = clap_config["audio_embed_dim"],
        audio_transformer_kwargs=audio_transformer_kwargs, 
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )

    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    model.audio_transformer_clap.requires_grad_(True)
    
    model.lang_encoder.gated_cross_attn_layers_sound.requires_grad_(True)

    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
    
    if unfreeze_full_lm:
        model.lang_encoder.requires_grad_(True)

    if unfreeze_clap:
        model.clap.requires_grad_(True)


    print("Flamingo model initialized with {:,} trainable parameters (audio transformer has {:,}, LM has {:,})".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.audio_transformer_clap.parameters() if p.requires_grad),
        sum(p.numel() for p in model.lang_encoder.parameters() if p.requires_grad),
    ))

    return model, text_tokenizer


def _infer_decoder_layers_attr_name(model):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "gptneoxforcausallm": "gpt_neox.layers",
    "mpt": "transformer.blocks",
    "mosaicgpt": "transformer.blocks",
    "qwen": "model.layers",
}


if __name__ == '__main__':
    import torch
    torch.set_printoptions(profile="full")  # only in debug mode
    import sys 
    sys.path.append('../')
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='../configs/config.yaml', help='yaml config path')
    args = parser.parse_args()

    config = yaml.load(open(args.config), Loader=yaml.FullLoader)

    data_config = config['data_config']
    model_config = config["model_config"]
    clap_config = config["clap_config"]

    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=False,
        gradient_checkpointing=True,
        freeze_lm_embeddings=True
    )
    model = model.cuda()

    from data.data import AudioTextData, DataCollator
    from torch.utils.data import DataLoader

    batch_size = 8
    trainset = AudioTextData(
        **data_config, clap_config=clap_config, tokenizer=tokenizer,
        epoch=1, force_reblend=True
    )
    data_collator = DataCollator(tokenizer)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=data_collator, num_workers=4)

    for step, batch in enumerate(trainloader):
        audio_clips = batch["audio_clips"].cuda()
        audio_embed_mask = batch["audio_embed_mask"].cuda()
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()

        print('batch {}:'.format(step+1), audio_clips.shape, audio_embed_mask.shape, input_ids.shape, attention_mask.shape)

        labels = input_ids.clone()

        labels[labels == tokenizer.pad_token_id] = -100
        labels[:, :2] = -100
        labels[labels == tokenizer.encode("<audio>")[-1]] = -100

        sep_locations = labels == tokenizer.sep_token_id
        endofchunk_token_id = tokenizer.encode("<|endofchunk|>")[-1]
        eoc_locations = labels == endofchunk_token_id

        if not all(sep_locations.sum(dim=1) == eoc_locations.sum(dim=1)):
            print("Warning: sep loc {} but eoc loc {}".format(sep_locations.sum(dim=1), eoc_locations.sum(dim=1)))
            
            for input_id in labels:
                input_id[input_id==-100] = tokenizer.encode("-")[-1]
                print(input_id, '\n', tokenizer.decode(input_id))

        for i in range(labels.shape[0]):
            shouldmask = True
            for j in range(labels.shape[1]):
                if shouldmask and (labels[i][j] != tokenizer.eos_token_id):
                    masked_value = -100
                else:
                    masked_value = labels[i][j]

                if labels[i][j] == tokenizer.sep_token_id:
                    shouldmask = False
                elif labels[i][j] == endofchunk_token_id:
                    shouldmask = True
                
                labels[i][j] = masked_value

            if labels[i][-1] not in [-100, tokenizer.eos_token_id, tokenizer.pad_token_id, endofchunk_token_id]:
                debug_masked_labels_in_the_end = []
                for j in range(labels.shape[1]-1, -1, -1):
                    if labels[i][j] not in [-100, tokenizer.eos_token_id, endofchunk_token_id]:
                        debug_masked_labels_in_the_end.insert(0, deepcopy(labels[i][j].item()))
                        labels[i][j] = -100
                    else:
                        break
                        
                print('hit max_token and masking ids from the end:', \
                    tokenizer.decode(torch.LongTensor(debug_masked_labels_in_the_end).to(labels.device))
                )

        if step == 50:
            break
