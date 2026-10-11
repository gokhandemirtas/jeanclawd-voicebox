# JeanClawd Voicebox

Local FastAPI server with two endpoints:

- `POST /transcribe` — multipart field `audio`; Whisper `large-v3` transcription.
- `POST /synthesize` — JSON text plus Qwen speaker/emotion controls; returns a WAV stream.

The default configuration is appropriate for a 24 GB NVIDIA GPU: Whisper
`large-v3` in `float16` and Qwen3-TTS `0.6B-CustomVoice` share the GPU comfortably.
Models download from Hugging Face on first request. The service falls back to
CPU when CUDA is not available.

## Run

```bash
uv sync
uv run jeanclawd-voicebox
```

For temporary local autoplay of generated speech, run:

```bash
uv run jeanclawd-voicebox --debug
```

Debug playback looks for `ffplay`, `mpv`, `paplay`, or `aplay` and runs
asynchronously so it does not hold up the HTTP response.

Logs are written to `logs/voicebox.log` by default. Set
`VOICEBOX_LOG_FILE=/path/to/voicebox.log` to change the location. Failed
requests include a request ID in the response and the log contains the full
traceback for that ID.

The host also needs `ffmpeg` for the audio formats accepted by faster-whisper.

## Examples

```bash
curl -F audio=@speech.m4a http://localhost:8000/transcribe

curl -X POST http://localhost:8000/synthesize \
  -H 'content-type: application/json' \
  -d '{"text":"Welcome back!", "emotion":"excited"}' \
  --output welcome.wav
```

TTS emotions are natural-language instructions passed to Qwen. The default
English speakers are `Ryan` and `Aiden`; use the `speaker` and `instruction`
fields to select a voice and further control delivery.

## Configuration

```bash
VOICEBOX_DEVICE=auto                 # auto, cuda, or cpu
VOICEBOX_WHISPER_MODEL=large-v3      # use distil-large-v3 for lower latency
VOICEBOX_WHISPER_COMPUTE_TYPE=float16
VOICEBOX_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
VOICEBOX_BEAM_SIZE=5
VOICEBOX_PORT=8000
```

The API schema is available at `/docs`; the application intentionally exposes
only the two inference routes above.
