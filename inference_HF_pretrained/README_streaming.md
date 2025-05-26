# Streaming Audio Flamingo Agent

This is a real-time streaming implementation of Audio Flamingo 2 using LiveKit for WebRTC communication.

## Overview

The agent listens to audio streams in real-time, processes them through Audio Flamingo 2 in a sliding window fashion, and announces detected sounds via text-to-speech.

## Key Features

- **Sliding Window Processing**: Processes audio in overlapping windows to capture sounds effectively
- **Real-time Detection**: Processes audio every 1 second for near real-time detection
- **Duplicate Filtering**: Avoids announcing the same sound multiple times
- **TTS Announcements**: Uses Deepgram TTS to announce detected sounds

## Setup

1. **Install Dependencies**:
```bash
pip install -r requirements_streaming.txt
```

2. **Environment Variables**:
Create a `.env` file with:
```
LIVEKIT_URL=your_livekit_server_url
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret
DEEPGRAM_API_KEY=your_deepgram_key
HUGGING_FACE_HUB_TOKEN=your_hf_token
```

3. **Download Model**:
The model will be automatically downloaded from HuggingFace on first run. It requires ~15GB of disk space.

## Running the Agent

```bash
python streaming_audio_flamingo_agent.py dev
```

For production deployment:
```bash
python streaming_audio_flamingo_agent.py start
```

## Architecture

### AudioFlamingoProcessor
- Maintains a sliding audio buffer
- Extracts overlapping windows for processing
- Manages detection filtering and deduplication

### AudioFlamingoAgent
- Handles LiveKit connection and audio streaming
- Manages TTS announcements
- Coordinates processing pipeline

## Testing

1. Start the agent
2. Join the same LiveKit room with audio. You can use this - https://github.com/livekit-examples/voice-assistant-frontend
3. Make various sounds (clapping, talking, music, etc.)
4. Agent will announce detected sounds via TTS 