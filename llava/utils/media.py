# Copyright (c) 2025 NVIDIA CORPORATION.
# Licensed under the MIT license.

# Adapted from https://github.com/NVlabs/VILA/tree/main under the Apache 2.0 license.
# LICENSE is in incl_licenses directory.

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
from llava.media import Image, Video, Speech, Sound
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

__all__ = ["extract_media"]

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)


def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)


def _extract_image(image: Union[Image, PIL.Image.Image]) -> PIL.Image.Image:
    if isinstance(image, Image):
        if image.path.startswith("http://") or image.path.startswith("https://"):
            image = PIL.Image.open(requests.get(image.path, stream=True).raw)
        else:
            image = PIL.Image.open(image.path)
    return image


def _load_video_bytesio(video_bytesio: BytesIO, *, num_frames: int) -> List[PIL.Image.Image]:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as temp_video:
        temp_video.write(video_bytesio.read())
        temp_video_name = temp_video.name
        return _load_video(temp_video_name, num_frames=num_frames)


def _load_video(video_path: str, *, num_frames: int) -> List[PIL.Image.Image]:
    # Load video frames from a directory
    if os.path.isdir(video_path):
        frame_paths = sorted(glob.glob(os.path.join(video_path, "*")))
        indices = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        return [PIL.Image.open(frame_paths[index]) for index in indices]

    # Load video frames from a video file
    vidcap = cv2.VideoCapture(video_path)

    # Find the last frame as frame count might not be accurate
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    while frame_count > 0:
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
        if vidcap.grab():
            break
        frame_count -= 1
    else:
        raise ValueError(f"Video '{video_path}' has no frames.")

    # Extract frames uniformly
    indices = np.round(np.linspace(0, frame_count - 1, num_frames)).astype(int)
    frames = {}
    for index in indices:
        if index in frames:
            continue
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
        success, frame = vidcap.read()
        if not success:
            logger.warning(f"Failed to read frame {index} from video '{video_path}'. Skipped.")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames[index] = PIL.Image.fromarray(frame)
    return [frames[index] for index in indices if index in frames]


def _extract_video(video: Video, config: PretrainedConfig) -> List[PIL.Image.Image]:
    num_frames = config.num_video_frames
    if getattr(config, "fps") != 0:
        logger.warning("Extracting frames from video with specified FPS is not supported yet. Ignored.")
    if isinstance(video.path, BytesIO):
        frames = _load_video_bytesio(video.path, num_frames=num_frames)
    else:
        frames = _load_video(video.path, num_frames=num_frames)
    return frames

def _load_speech(speech_path: str):
    # Load video frames from a directory
    if speech_path is None:
        return None
    speech_outputs = []

    speech = whisper.load_audio(speech_path)
    speech = whisper.pad_or_trim(speech)
    mel = whisper.log_mel_spectrogram(speech)
    speech_outputs.append(mel.unsqueeze(0))
    speech_frames = torch.stack(speech_outputs, dim=0)
    return speech_frames.numpy().tolist()

def _extract_speech(speech: Speech, config: PretrainedConfig):
    frames = _load_speech(speech.path)
    return frames
       
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

def extract_media(
    messages: List[Dict[str, Any]],
    config: Optional[PretrainedConfig] = None,
    draft: bool = False,
) -> Dict[str, List[Any]]:
    media = defaultdict(list)
    media_meta = defaultdict(list)
    for message in messages:
        text = ""
        for part in make_list(message["value"]):
            if isinstance(part, list):
                part = part[0]
            else:
                part = part
            if isinstance(part, str):
                for token in MEDIA_TOKENS.values():
                    if token in part:
                        logger.warning(f"Media token '{token}' found in text: '{part}'. Removed.")
                        part = part.replace(token, "").strip()
                text += part
            if isinstance(part, (Image, PIL.Image.Image)):
                if draft:
                    media["image"].append(part)
                else:
                    media["image"].append(_extract_image(part))
                text += MEDIA_TOKENS["image"]
            if isinstance(part, Video):
                if draft:
                    media["video"].append(part)
                else:
                    media["video"].append(_extract_video(part, config))
                text += MEDIA_TOKENS["video"]
            if isinstance(part, Speech):
                if draft:
                    media["speech"].append(part)
                else:
                    media["speech"].append(_extract_speech(part, config))
                text += MEDIA_TOKENS["speech"]
            if isinstance(part, Sound):
                if draft:
                    media["sound"].append(part)
                else:
                    sound, audio_feature_masks,audio_embed_masks = _extract_sound_mask(part, config)
                    media["sound"].append(sound)
                    media_meta["sound_feature_masks"].append(audio_feature_masks)
                    media_meta["sound_embed_masks"].append(audio_embed_masks)
                text += MEDIA_TOKENS["sound"] * len(sound)
   
        message["value"] = text
    return media, media_meta
