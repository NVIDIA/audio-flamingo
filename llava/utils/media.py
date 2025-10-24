# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

import re
import glob
import os
import tempfile
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import PIL
import PIL.Image
import requests
from transformers import PretrainedConfig
from pydub import AudioSegment

from llava.constants import MEDIA_TOKENS
from llava.media import Sound
from llava.utils import make_list
from llava.utils.logging import logger
import torch
import whisper
import soundfile as sf
from librosa import resample as librosa_resample
from transformers import AutoFeatureExtractor
import math
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler, UniformClipSampler
import kaldiio
wav_processor = AutoFeatureExtractor.from_pretrained('Qwen/Qwen2-Audio-7B')

SOUND_TAG_RE = re.compile(r"<sound(?:-(\d+))?>", flags=re.IGNORECASE)

__all__ = ["extract_media"]

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def _get_num_windows(T, sr):

    window_length  = int(30.0 * sr)
    window_overlap = int(0.0 * sr)
    max_num_window = 20
    num_windows = 1
    if T <= window_length:
        num_windows = 1
        full_length = window_length
    elif T >= (max_num_window * window_length - (max_num_window - 1) * window_overlap):
        num_windows = max_num_window
        full_length = (max_num_window * window_length - (max_num_window - 1) * window_overlap)
    else:
        num_windows = 1 + int(np.ceil((T - window_length) / float(window_length - window_overlap)))
        full_length = num_windows * window_length - (num_windows - 1) * window_overlap
    
    return num_windows, full_length

def _load_audio(file_path, target_sr=16000, duration=30.0, start=0.0):
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
                data = librosa_resample(data.flatten(), orig_sr=original_sr, target_sr=target_sr)
            else:
                data = librosa_resample(data.T, orig_sr=original_sr, target_sr=target_sr)[0]
        else:
            if channels != 1:
                data = data.T[0]
    
    if data.min() >= 0:
        data = 2 * data / abs(data.max()) - 1.0
    else:
        data = data / max(abs(data.max()), abs(data.min()))
    
    assert len(data.shape) == 1, data.shape
    return data


def _load_sound_mask(sound_file, sample_rate=16000, window_length=30.0, window_overlap=0.0, max_num_window=20, audio_start = 0.0):
    if sound_file is None:
        return None
    window_length  = int(window_length * sample_rate)
    window_overlap = int(window_overlap * sample_rate)
    max_num_window = int(max_num_window)
    duration = max_num_window * (window_length - window_overlap) + window_overlap

    sound_outputs = []
    audio_feature_masks = []
    audio_embed_masks = []

    try:
        audio_data = _load_audio(sound_file, sample_rate, duration, audio_start) # already cuts to max duration
        T = len(audio_data)
        audio_data = audio_data.reshape(1, -1)
        num_windows, full_length = _get_num_windows(T, sample_rate)

        audio_data_tensor = torch.from_numpy(int16_to_float32(float32_to_int16(audio_data))).float()
        for i in range(num_windows):
            audio_embed_mask = torch.zeros(750)
            start = i * (window_length - window_overlap)
            audio_data_tensor_this = audio_data_tensor[:, start:start+window_length]
            orig_length = audio_data_tensor_this.shape[1]
            audio_data_tensor_this = wav_processor(audio_data_tensor_this.cpu().numpy(), sampling_rate=sample_rate, return_tensors="pt") #.squeeze(0) text="dummy", audios=audio_data_tensor_this, return_tensors="pt") #
            sound_outputs.append(audio_data_tensor_this["input_features"])
            # calculate the mask for the input melspec to Whisper
            melspec_frames_this_window = int(math.ceil(orig_length / 160))
            feature_attention_mask = torch.zeros(3000, dtype=torch.int32)
            feature_attention_mask[:melspec_frames_this_window] = 1
            audio_feature_masks.append(feature_attention_mask.unsqueeze(0))
            # calculate the mask for the output embedding for use in AF3
            conv_lengths = (melspec_frames_this_window - 1) // 2 + 1
            output_embedding_lengths = (conv_lengths - 2) // 2 + 1
            audio_embed_mask[:output_embedding_lengths] = 1
            audio_embed_masks.append(audio_embed_mask)
    except:
        print("Error loading sound file: ", sound_file)
        sound_outputs.append(torch.zeros(1,128,3000))
        audio_feature_masks.append(torch.zeros(1, 3000, dtype=torch.int32))
        audio_embed_masks.append(torch.zeros(750))
    sound_outputs = torch.stack(sound_outputs, dim=0)
    audio_feature_masks = torch.stack(audio_feature_masks, dim=0)
    audio_embed_masks = torch.stack(audio_embed_masks, dim=0)
    return sound_outputs.numpy().tolist(), audio_feature_masks ,audio_embed_masks


def _extract_sound_mask(sound: Sound, config: PretrainedConfig):
    frames, audio_feature_masks, audio_embed_masks = _load_sound_mask(sound.path)
    return frames, audio_feature_masks, audio_embed_masks

# def extract_media(
#     messages: List[Dict[str, Any]],
#     config: Optional[PretrainedConfig] = None,
#     draft: bool = False,
# ) -> Dict[str, List[Any]]:
#     media = defaultdict(list)
#     media_meta = defaultdict(list)
#     for message in messages:
#         text = ""
#         for part in make_list(message["value"]):
#             if isinstance(part, list) and not isinstance(part[0], Sound):
#                 part = part[0]
#             else:
#                 part = part
#             if isinstance(part, str):
#                 for token in MEDIA_TOKENS.values():
#                     if token in part:
#                         # logger.warning(f"Media token '{token}' found in text: '{part}'. Removed.")
#                         part = part.replace(token, "").strip()                        
#                 text += part
#             if isinstance(part, Sound):
#                 if draft:
#                     media["sound"].append(part)
#                 else:
#                     sound, audio_feature_masks,audio_embed_masks = _extract_sound_mask(part, config)
#                     media["sound"].append(sound)
#                     media_meta["sound_feature_masks"].append(audio_feature_masks)
#                     media_meta["sound_embed_masks"].append(audio_embed_masks)
#                 text += MEDIA_TOKENS["sound"] * len(sound)
#             if isinstance(part, list):
#                 for item in part:
#                     if isinstance(item, Sound):
#                         sound, audio_feature_masks,audio_embed_masks = _extract_sound_mask(part, config)
#                         media["sound"].append(sound)
#                         media_meta["sound_feature_masks"].append(audio_feature_masks)
#                         media_meta["sound_embed_masks"].append(audio_embed_masks)
#             if part is None:
#                 media["sound"].append(None)
#                 media_meta["sound_feature_masks"].append(None)
#                 media_meta["sound_embed_masks"].append(None)
   
#         message["value"] = text
#     return media, media_meta

def extract_media(
    messages: List[Dict[str, Any]],
    config: Optional[PretrainedConfig] = None,
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)
    media_meta = defaultdict(list)

    def _wrap_sound_runs(s: str) -> str:
        tok = MEDIA_TOKENS["sound"]
        if not tok:
            return s
        etok = re.escape(tok)
        pattern = re.compile(f'(?:{etok})+')  # any-length run of <sound>

        out = []
        last = 0
        n = len(s)
        for m in pattern.finditer(s):
            i, j = m.span()
            # copy preceding text
            chunk = s[last:i]
            out.append(chunk)

            if i > 0 and s[i-1] != '\n':
                if out and out[-1]:
                    out[-1] = out[-1].rstrip()  # remove spaces before the newline
                out.append('\n')

            # emit the run itself (no internal newlines)
            out.append(m.group(0))

            # newline AFTER the run (not at end) — we’ll normalize spaces around newlines later
            if j < n and s[j:j+1] != '\n':
                out.append('\n')

            last = j

        out.append(s[last:])
        s2 = ''.join(out)

        # --- NEW: global cleanup so there are NO spaces touching a newline ---
        # " \n" -> "\n" and "\n " -> "\n"
        s2 = re.sub(r'[ \t]*\n[ \t]*', '\n', s2)

        # Never end with a newline
        s2 = s2.rstrip('\n')

        return s2

    for message in messages:
        parts = make_list(message["value"])

        has_placeholder = any(
            isinstance(p, str) and SOUND_TAG_RE.search(p) is not None
            for p in parts
        )

        local_sound_lengths: Dict[int, int] = {}
        next_local_idx = 1

        def _register_sound(snd_obj):
            nonlocal next_local_idx
            if draft:
                snd, sfm, sem = snd_obj, None, None
            else:
                snd, sfm, sem = _extract_sound_mask(snd_obj, config)

            media["sound"].append(snd)
            media_meta["sound_feature_masks"].append(sfm)
            media_meta["sound_embed_masks"].append(sem)

            idx = next_local_idx
            next_local_idx += 1
            try:
                local_sound_lengths[idx] = len(snd)  # number of windows for this sound occurrence
            except Exception:
                local_sound_lengths[idx] = 1
            return idx

        # Collect sounds (and mirror None) but don’t write text yet
        for part in parts:
            if isinstance(part, list) and part and not isinstance(part[0], Sound):
                part = part[0]

            if isinstance(part, Sound):
                _register_sound(part)
            elif isinstance(part, list):
                for item in part:
                    if isinstance(item, Sound):
                        _register_sound(item)
            elif part is None:
                media["sound"].append(None)
                media_meta["sound_feature_masks"].append(None)
                media_meta["sound_embed_masks"].append(None)

        text_out: List[str] = []

        def _strip_literal_media_tokens(s: str) -> str:
            # used only in legacy mode (no placeholders)
            for tok in MEDIA_TOKENS.values():
                if tok in s:
                    s = s.replace(tok, "")
            return s

        if has_placeholder:
            # PLACEHOLDER MODE: replace <sound> / <sound-k> with repeated tokens (no stripping)
            consumed = set()
            next_implicit = 1

            def _consume_next_unconsumed():
                nonlocal next_implicit
                while next_implicit in consumed or next_implicit not in local_sound_lengths:
                    next_implicit += 1
                    if next_implicit >= next_local_idx:
                        return None
                consumed.add(next_implicit)
                ret = next_implicit
                next_implicit += 1
                return ret

            def _replace_placeholders(s: str) -> str:
                def _sub_fn(m: re.Match) -> str:
                    g = m.group(1)
                    if g is not None:
                        k = int(g)
                        n = local_sound_lengths.get(k, 0)
                        if n <= 0:
                            return ""
                        consumed.add(k)
                        return MEDIA_TOKENS["sound"] * n
                    else:
                        k = _consume_next_unconsumed()
                        if k is None:
                            return ""
                        n = local_sound_lengths.get(k, 0)
                        return MEDIA_TOKENS["sound"] * n
                return SOUND_TAG_RE.sub(_sub_fn, s)

            for part in parts:
                if isinstance(part, list) and part and not isinstance(part[0], Sound):
                    part = part[0]
                if isinstance(part, str):
                    text_out.append(_replace_placeholders(part))
                # sounds/None/lists contribute via placeholders only in this mode

        else:
            # LEGACY MODE: strip literals in strings; inject tokens when encountering Sound objects
            i_sound = 0
            for part in parts:
                if isinstance(part, list) and part and not isinstance(part[0], Sound):
                    part = part[0]

                if isinstance(part, str):
                    text_out.append(_strip_literal_media_tokens(part))
                elif isinstance(part, Sound):
                    i_sound += 1
                    nrep = local_sound_lengths.get(i_sound, 0)
                    if nrep > 0:
                        text_out.append(MEDIA_TOKENS["sound"] * nrep)
                elif isinstance(part, list):
                    for item in part:
                        if isinstance(item, Sound):
                            i_sound += 1
                            nrep = local_sound_lengths.get(i_sound, 0)
                            if nrep > 0:
                                text_out.append(MEDIA_TOKENS["sound"] * nrep)
                # None adds no text

        message["value"] = _wrap_sound_runs("".join(text_out))

    return media, media_meta