# Copyright (c) 2024 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/mlfoundations/open_flamingo under the MIT license.
#   LICENSE is in incl_licenses directory.

import sys 
sys.path.append('../')

from typing import Optional
from copy import deepcopy

from transformers import AutoModelForCausalLM, AutoTokenizer
from my_laion_clap.CLAP.src.laion_clap.hook import CLAP_Module
from my_ms_clap.src.CLAPWrapper import CLAPWrapper

import torch
from torch import nn

try:
    from .flamingo import Flamingo
    from .flamingo_lm import FlamingoLMMixin
    from .utils import extend_instance
except:
    from flamingo import Flamingo
    from flamingo_lm import FlamingoLMMixin
    from utils import extend_instance


class CLAP(nn.Module):
    def __init__(self, clap_config):
        super(CLAP, self).__init__()
        self.method = clap_config["method"]
        device_id = f'cuda:{torch.cuda.current_device()}'

        if ('finetune' in clap_config) and clap_config['finetune']:
            self.finetune = True 
            print('Finetuning CLAP encoder as well!')
        else:
            self.finetune = False 

        if self.method == 'laion-clap':
            # https://github.com/LAION-AI/CLAP
            if clap_config["model_name"] in ['630k-audioset-best', '630k-best', '630k-audioset-fusion-best', '630k-fusion-best']:
                amodel = 'HTSAT-tiny'
            elif clap_config["model_name"] in ['music_speech_audioset_epoch_15_esc_89.98']:
                amodel = 'HTSAT-base'
            else:
                raise NotImplementedError
        
            enable_fusion = 'fusion' in clap_config["model_name"].lower()
            self.laion_clap = CLAP_Module(enable_fusion=enable_fusion, amodel=amodel, device=device_id)
            self.laion_clap.load_ckpt(ckpt=clap_config["checkpoint"])
            
            
            for param in self.laion_clap.parameters():
                param.requires_grad = self.finetune

            if self.finetune:
                self.laion_clap.train()
            else:
                self.laion_clap.eval()

            print('loaded laion-clap model: {}'.format(clap_config["checkpoint"]))
    
        elif self.method == 'microsoft-clap':
            # https://github.com/microsoft/CLAP
            self.ms_clap = CLAPWrapper(
                clap_config["checkpoint"], 
                config_root=clap_config["config_root"],
                version=clap_config['model_name'], 
                use_cuda=True
            )
            
            if clap_config['model_name'] in ['2022', '2023']:
                for param in self.ms_clap.clap.parameters():
                    param.requires_grad = self.finetune
                if self.finetune:
                    self.ms_clap.clap.train()
                else:
                    self.ms_clap.clap.eval()
            else:
                for param in self.ms_clap.clapcap.parameters():
                    param.requires_grad = self.finetune
                if self.finetune:
                    self.ms_clap.clapcap.train()
                else:
                    self.ms_clap.clapcap.eval()

            print('loaded microsoft-clap model: {}'.format(clap_config["checkpoint"]))
        
        else:
            raise NotImplementedError

    def forward(self, audio_clips):
        
        if len(audio_clips.shape) == 2:
            audio_clips = audio_clips.unsqueeze(0)
        assert len(audio_clips.shape) == 3

        audio_embeds = []
        for x in audio_clips:
            if self.method == 'laion-clap':
                audio_embed = self.laion_clap.get_audio_embedding_from_data(x=x, use_tensor=True)
            elif self.method == 'microsoft-clap':
                audio_embed = self.ms_clap.get_audio_embeddings_from_clips(x)
                
            audio_embeds.append(audio_embed)

        audio_embeds = torch.stack(audio_embeds, dim=0)
        audio_embeds.requires_grad = self.finetune

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
        {"additional_special_tokens": ["<audio>", "<|endofchunk|>"]}
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
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
        audio_embed_dim=clap_config["audio_embed_dim"],
        audio_transformer_kwargs=audio_transformer_kwargs, 
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        **flamingo_kwargs,
    )

    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    model.audio_transformer.requires_grad_(True)
    model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
    if not freeze_lm_embeddings:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
    
    if unfreeze_full_lm:
        model.lang_encoder.requires_grad_(True)
    
    if unfreeze_clap:
        model.clap.requires_grad_(True)

    print("Flamingo model initialized with {:,} trainable parameters (audio transformer has {:,}, LM has {:,})".format(
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        sum(p.numel() for p in model.audio_transformer.parameters() if p.requires_grad),
        sum(p.numel() for p in model.lang_encoder.parameters() if p.requires_grad)
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
}
