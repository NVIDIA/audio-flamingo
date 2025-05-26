#!/usr/bin/env python3
import asyncio
import logging
import os
import yaml
import json
import numpy as np
import torch
from collections import deque
from typing import Optional, List, Tuple
import time

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, AutoSubscribe
from livekit.plugins import deepgram
from dotenv import load_dotenv
from safetensors.torch import load_file
from huggingface_hub import snapshot_download

# Import Audio Flamingo components
from .src.factory import create_model_and_transforms
from .src.flamingo import Flamingo
from audio_flamingo_2.train.train_utils import Dict2Class


logger = logging.getLogger(__name__)

# Set specific loggers to reduce noise
logging.getLogger("livekit").setLevel(logging.INFO)
logging.getLogger("deepgram").setLevel(logging.INFO)
logging.getLogger("numba").setLevel(logging.INFO)
logging.getLogger("transformers").setLevel(logging.INFO)

# Load environment variables
load_dotenv()
logger.info("Environment variables loaded")

# Audio configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.1  # 100ms chunks for real-time processing
PROCESS_INTERVAL = 5.0  # Process every 5 second

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

class AudioFlamingoProcessor:
    def __init__(self, model, tokenizer, clap_config, device_id=0):
        self.model: Flamingo = model
        self.tokenizer = tokenizer
        self.clap_config = clap_config
        self.device_id = device_id
        
        # Get window parameters
        self.window_length = int(float(clap_config["window_length"]) * SAMPLE_RATE)
        self.window_overlap = int(float(clap_config["window_overlap"]) * SAMPLE_RATE)
        self.max_num_window = int(clap_config["max_num_window"])
        logger.info(f"window_length: {self.window_length}, window_overlap: {self.window_overlap}, max_num_window: {self.max_num_window}")
        
        # Audio buffer for sliding window
        max_buffer_size = self.window_length * self.max_num_window
        self.audio_buffer = deque(maxlen=max_buffer_size)
        
        # Cache for recent detections to avoid duplicates
        self.recent_detections = deque(maxlen=5)
        self.last_detection_time = {}
        
        # Use float32 to avoid overflow issues
        self.cast_dtype = torch.float32
        
        logger.info(f"Initialized AudioFlamingoProcessor:")
        logger.info(f"  - Window length: {self.window_length} samples ({self.window_length/SAMPLE_RATE:.2f}s)")
        logger.info(f"  - Window overlap: {self.window_overlap} samples")
        logger.info(f"  - Max windows: {self.max_num_window}")
    
    def add_audio_chunk(self, audio_data: np.ndarray):
        """Add audio chunk to buffer"""
        chunk_size = len(audio_data.flatten())
        self.audio_buffer.extend(audio_data.flatten())
        # logger.info(f"Added {chunk_size} samples to buffer. Buffer size: {len(self.audio_buffer)}/{self.audio_buffer.maxlen}")
        # if too much in buffer, drop the oldest chunk

    def prepare_audio_windows(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare audio windows from buffer"""
        if len(self.audio_buffer) < self.window_length:
            logger.info(f"Buffer too small: {len(self.audio_buffer)} < {self.window_length}")
            return None, None
        
        buffer_array = np.array(self.audio_buffer)
        
        # Calculate number of windows
        num_windows = min(
            1 + max(0, (len(buffer_array) - self.window_length) // (self.window_length - self.window_overlap)),
            self.max_num_window
        )
        
        # Extract windows
        audio_clips = []
        for i in range(num_windows):
            start = i * (self.window_length - self.window_overlap)
            end = start + self.window_length
            
            if end <= len(buffer_array):
                window = buffer_array[start:end]
                # Normalize and convert
                window = int16_to_float32(float32_to_int16(window))
                audio_clips.append(torch.from_numpy(window).float().unsqueeze(0))
        
        if not audio_clips:
            return None, None
        
        audio_clips = torch.cat(audio_clips)
        audio_embed_mask = torch.ones(len(audio_clips))
        
        logger.info(f"Prepared {len(audio_clips)} windows from buffer of {len(buffer_array)} samples")
        
        return audio_clips, audio_embed_mask
    
    async def process_audio(self) -> Optional[str]:
        """Process current audio buffer and return detected sounds"""
        logger.info("Starting audio processing...")
        audio_clips, audio_embed_mask = self.prepare_audio_windows()
        if audio_clips is None:
            logger.info("No audio clips prepared, skipping processing")
            return None
        
        try:
            # Move to device with float32 to avoid overflow
            logger.info(f"Moving audio clips to device {self.device_id}")
            audio_clips = audio_clips.to(self.device_id, dtype=self.cast_dtype, non_blocking=True)
            audio_embed_mask = audio_embed_mask.to(self.device_id, dtype=self.cast_dtype, non_blocking=True)
            
            # Simple prompt for sound detection
            prompt = "What sounds do you hear?"
            sample = f"<audio>{prompt}{self.tokenizer.sep_token}"
            logger.info(f"Using prompt: {sample}")
            
            text = self.tokenizer(
                sample,
                max_length=512,
                padding="longest",
                truncation="only_first",
                return_tensors="pt"
            )
            
            input_ids = text["input_ids"].to(self.device_id, non_blocking=True)
            attention_mask = text.get("attention_mask", torch.ones_like(input_ids)).to(self.device_id, non_blocking=True)
            
            # Set pad_token_id if not set
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Generate response using the inference Flamingo's generate method
            logger.info("Generating model response...")
            with torch.no_grad():
                with torch.autocast(
                    device_type='cuda' if torch.cuda.is_available() else 'cpu',
                    enabled=False  # Disable autocast to avoid overflow issues
                ):
                    output = self.model.generate(
                        audio_x=audio_clips.unsqueeze(0),
                        audio_x_mask=audio_embed_mask.unsqueeze(0),
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        max_new_tokens=64,
                        temperature=0.0,  # Use temperature=0.0 as recommended for benchmarking
                        do_sample=False,  # Use greedy decoding for more consistent results
                    )
            
            # Check if output is empty
            logger.info(f"Generated output shape: {output.shape if output is not None else 'None'}")
            if output is None or len(output) == 0 or len(output[0]) == 0:
                logger.warning("Model generated empty output")
                return "silence"
            
            # Decode output
            logger.info("Decoding model output...")
            
            # Get only the newly generated tokens (skip the input)
            input_length = input_ids.shape[1]
            if output.shape[1] > input_length:
                new_tokens = output[0][input_length:]
                output_decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            else:
                # Fallback: decode the full output and split
                full_decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
                if self.tokenizer.sep_token in full_decoded:
                    output_decoded = full_decoded.split(self.tokenizer.sep_token)[-1]
                else:
                    output_decoded = full_decoded
            
            output_decoded = output_decoded.strip()
            
            logger.info(f"Model output: '{output_decoded}'")
            return output_decoded if output_decoded else "silence"
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def should_announce(self, detection: str) -> bool:
        """Check if we should announce this detection"""
        logger.info(f"Checking if should announce: '{detection}'")
        if not detection or len(detection) < 5:
            logger.info("Detection too short, not announcing")
            return False
        
        # Check for common non-informative responses
        non_informative = ["i don't hear", "no sound", "silence", "nothing", "quiet"]
        if any(phrase in detection.lower() for phrase in non_informative):
            logger.info(f"Detection is non-informative: '{detection}'")
            return False
        
        # Check if recently detected
        current_time = time.time()
        detection_lower = detection.lower().strip()
        
        # Avoid exact duplicates in recent history
        for recent in self.recent_detections:
            if detection_lower == recent.lower():
                return False
        
        # Check cooldown for similar detections
        if detection_lower in self.last_detection_time:
            if current_time - self.last_detection_time[detection_lower] < 3.0:
                return False
        
        # Update tracking
        self.recent_detections.append(detection)
        self.last_detection_time[detection_lower] = current_time
        
        logger.info(f"Will announce detection: '{detection}'")
        return True

class AudioFlamingoAgent:
    def __init__(self, ctx: JobContext):
        logger.info(f"AudioFlamingoAgent initialized for room: {ctx.room.name if ctx.room else 'No room'}")
        self.ctx = ctx
        self.processor: Optional[AudioFlamingoProcessor] = None
        self.tts = deepgram.TTS(
            model="aura-asteria-en",
            sample_rate=SAMPLE_RATE  # Set to 16000 Hz to match our audio source
        )
        self.audio_source = rtc.AudioSource(SAMPLE_RATE, 1)
        self.processing = False
        
    async def initialize_model(self):
        """Initialize Audio Flamingo model"""
        logger.info("Initializing Audio Flamingo model...")
        
        try:
            # Download model from HuggingFace
            model_dir = "./download/audio_flamingo_model"
            if not os.path.exists(os.path.join(model_dir, "configs")):
                logger.info("Downloading Audio Flamingo model from HuggingFace...")
                snapshot_download(repo_id="nvidia/audio-flamingo-2", local_dir=model_dir)
            
            # Download CLAP model if needed
            clap_model_dir = "./download/clap_model"
            if not os.path.exists(clap_model_dir):
                logger.info("Downloading CLAP model...")
                snapshot_download(repo_id="laion/clap-htsat-fused", local_dir=clap_model_dir)
            
            # Download Qwen model if needed
            qwen_model_dir = "./download/qwen_model"
            if not os.path.exists(qwen_model_dir):
                logger.info("Downloading Qwen model...")
                snapshot_download(repo_id="Qwen/Qwen2.5-3B", local_dir=qwen_model_dir)
            logger.info(f"Qwen model downloaded to: {qwen_model_dir}")
            
            # Load configuration - use local config since HF model doesn't include it
            config_path = "./inference_HF_pretrained/configs/inference.yaml"
            if not os.path.exists(config_path):
                config_path = "./audio_flamingo_2/configs/inference.yaml"
            
            if not os.path.exists(config_path):
                logger.error(f"Config file not found at {config_path}")
                raise FileNotFoundError(f"Config file not found at {config_path}")
            
            logger.info(f"Loading config from: {config_path}")
            config = yaml.load(open(config_path), Loader=yaml.FullLoader)
            
            data_config = config['data_config']
            model_config = config['model_config']
            clap_config = config['clap_config']
            args = Dict2Class(config['train_config'])
            
            # Update model config to use local paths
            model_config['lang_encoder_path'] = qwen_model_dir
            model_config['tokenizer_path'] = qwen_model_dir
            
            # Update CLAP config to use local model
            clap_config['model_name'] = clap_model_dir
            
            # Fix checkpoint format if needed
            original_checkpoint = os.path.join(clap_model_dir, "pytorch_model.bin")
            fixed_checkpoint = os.path.join(clap_model_dir, "pytorch_model_fixed.bin")
            
            if not os.path.exists(fixed_checkpoint) and os.path.exists(original_checkpoint):
                logger.info("Fixing CLAP checkpoint format...")
                original_state_dict = torch.load(original_checkpoint, map_location='cpu')
                fixed_state_dict = {'state_dict': original_state_dict}
                torch.save(fixed_state_dict, fixed_checkpoint)
                logger.info("CLAP checkpoint fixed!")
            
            clap_config['checkpoint'] = fixed_checkpoint
            
            # Create model
            logger.info(f"Creating model with config: {model_config}")
            model, tokenizer = create_model_and_transforms(
                **model_config,
                clap_config=clap_config,
                use_local_files=True,  # Force local files
                gradient_checkpointing=args.gradient_checkpointing,
                freeze_lm_embeddings=args.freeze_lm_embeddings,
            )
            logger.info(f"Model created: {model}")
            
            # Load checkpoint
            assert torch.cuda.is_available(), "CUDA is not available"
            device_id = 0 if torch.cuda.is_available() else "cpu"
            model = model.to(device_id)
            model.eval()
            
            # Load state dict
            metadata_path = os.path.join(model_dir, "safe_ckpt/metadata.json")
            if not os.path.exists(metadata_path):
                logger.error(f"Metadata file not found at {metadata_path}")
                raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
                
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            state_dict = {}
            for chunk_name in metadata:
                chunk_path = os.path.join(model_dir, f"safe_ckpt/{chunk_name}.safetensors")
                if not os.path.exists(chunk_path):
                    logger.error(f"Checkpoint chunk not found at {chunk_path}")
                    raise FileNotFoundError(f"Checkpoint chunk not found at {chunk_path}")
                chunk_tensors = load_file(chunk_path)
                state_dict.update(chunk_tensors)
            
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Model loaded: {model}")
            
            # Create processor
            self.processor = AudioFlamingoProcessor(model, tokenizer, clap_config, device_id)
            
            logger.info("Audio Flamingo model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    async def process_audio_stream(self, audio_stream: rtc.AudioStream):
        """Process incoming audio stream"""
        logger.info("Starting audio stream processing")
        
        last_process_time = time.time()
        frame_count = 0
        
        try:
            async for event in audio_stream:
                frame = event.frame
                frame_count += 1
                if isinstance(event, rtc.AudioFrameEvent):

                    
                    if frame_count % 100 == 0:  # Log every 100 frames
                        logger.info(f"Received {frame_count} audio frames")
                    
                    # Convert audio frame to numpy array
                    audio_data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Add to buffer (only if processor is ready)
                    if self.processor:
                        self.processor.add_audio_chunk(audio_data)
                        
                        # Process periodically
                        current_time = time.time()
                        if frame_count % 100 == 0:  # Log every 100 frames
                            logger.info(f"current_time: {current_time}, last_process_time: {last_process_time}, PROCESS_INTERVAL: {PROCESS_INTERVAL}, self.processing: {self.processing}, self.processor.audio_buffer len: {len(self.processor.audio_buffer)}")
                        if current_time - last_process_time >= PROCESS_INTERVAL and not self.processing:
                            last_process_time = current_time
                            logger.info("Processing audio")
                            asyncio.create_task(self.process_and_announce())
                    else:
                        if frame_count % 100 == 0:
                            logger.info("Processor not ready, dropping audio frame")
                        # Test mode - just announce that we're receiving audio
                        current_time = time.time()
                        if current_time - last_process_time >= PROCESS_INTERVAL:
                            last_process_time = current_time
                            logger.info("TEST MODE: Received audio, announcing test message")
                            asyncio.create_task(self.announce_detection("test audio detected"))
        
        except Exception as e:
            logger.error(f"Error in audio stream processing: {e}")
    
    async def process_and_announce(self):
        """Process audio and announce detections"""
        if self.processing:
            return
        
        self.processing = True
        try:
            logger.info("Processing audio calling")
            detection = await self.processor.process_audio()
            logger.info(f"Detection: {detection}")
            
            if detection and self.processor.should_announce(detection):
                logger.info(f"Detected: {detection}")
                await self.announce_detection(detection)
                
        except Exception as e:
            logger.error(f"Error in process_and_announce: {e}")
        finally:
            logger.info("process_and_announce finished")
            self.processing = False
    
    async def announce_detection(self, detection: str):
        """Announce detected sound via TTS"""
        try:
            # Format the announcement
            announcement = f"I hear {detection}"
            logger.info(f"Announcing: {announcement}")
            
            # Generate TTS and stream it
            tts_stream = self.tts.stream()
            tts_stream.push_text(announcement)
            tts_stream.flush()
            
            logger.info(f"flushed tts stream")
            
            frame_count = 0
            # Add a timeout to prevent hanging
            try:
                async with asyncio.timeout(5.0):  # 5 second timeout
                    async for audio in tts_stream:
                        # Ensure the audio frame matches our audio source format
                        frame = audio.frame
                        frame_count += 1
                        if frame_count <= 2:  # Only log first few frames
                            logger.info(f"frame {frame_count}: {frame}")
                        if frame.sample_rate != SAMPLE_RATE or frame.num_channels != 1:
                            logger.warning(f"TTS audio format mismatch: {frame.sample_rate}Hz, {frame.num_channels}ch vs expected {SAMPLE_RATE}Hz, 1ch")
                            # Skip this frame or convert it
                            continue
                        await self.audio_source.capture_frame(frame)
            except asyncio.TimeoutError:
                logger.warning(f"TTS stream timed out after {frame_count} frames")
            
            # Close the stream after we're done
            await tts_stream.aclose()
            logger.info(f"TTS completed, sent {frame_count} frames")
            
        except Exception as e:
            logger.error(f"Error announcing detection: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    async def run(self):
        """Main agent loop"""
        
        await self.ctx.wait_for_participant()
        
        logger.info("waited for participant, Audio Flamingo agent starting...")
        
        # Initialize model
        await self.initialize_model()
        
        # Room is already connected via entrypoint
        logger.info(f"Agent running in room: {self.ctx.room.name}")
        
        # Publish TTS audio track
        tts_track = rtc.LocalAudioTrack.create_audio_track("tts-output", self.audio_source)
        await self.ctx.room.local_participant.publish_track(tts_track)
        logger.info("Published TTS audio track")
        
        # Assert exactly one remote participant
        remote_participants = list(self.ctx.room.remote_participants.values())
        assert len(remote_participants) == 1, f"Expected 1 remote participant, got {len(remote_participants)}"
        
        participant = remote_participants[0]
        logger.info(f"Processing participant: {participant.identity}")
        
        # Find and subscribe to the audio track
        audio_track_found = False
        for track_sid, publication in participant.track_publications.items():
            logger.info(f"Track: {track_sid}, kind: {publication.kind}, subscribed: {publication.subscribed}")
            if publication.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Found audio track from {participant.identity}, subscribing...")
                if not publication.subscribed:
                    publication.set_subscribed(True)
                    logger.info(f"Subscribed to audio track")
                
                await asyncio.sleep(0.5)
                logger.info(f"publication: {publication}")
                assert publication.track is not None, f"Track is None for participant {participant.identity}"
                logger.info(f"Processing audio track from {participant.identity}")
                audio_stream = rtc.AudioStream(publication.track)
                asyncio.create_task(self.process_audio_stream(audio_stream))
                audio_track_found = True
                break
        
        assert audio_track_found, f"No audio track found for participant {participant.identity}"
        
        # Keep agent running
        logger.info("Audio Flamingo agent ready and listening...")
        await asyncio.Event().wait()

async def entrypoint(ctx: JobContext):
    """LiveKit agent entrypoint"""
    logger.info(f"Agent job started for room: {ctx.room.name if ctx.room else 'Unknown'}")
    logger.info(f"Job context: {ctx}")
    
    # Wait for the room to be ready
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    logger.info(f"Connected to room: {ctx.room.name}")
    
    
    agent = AudioFlamingoAgent(ctx)
    await agent.run()

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            job_memory_warn_mb=30000,
            # Configure automatic room dispatch
            # agent_name="audio-flamingo-agent",
        )
    ) 