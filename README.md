# JeanClawd Voicebox

Local FastAPI server with two endpoints:

- `POST /transcribe` — multipart field `audio`; Whisper `large-v3` transcription.
- `POST /synthesize` — JSON text plus Qwen speaker/emotion controls; returns a WAV stream.

The default configuration is appropriate for a 24 GB NVIDIA GPU: Whisper
`large-v3` in `float16` and Qwen3-TTS `1.7B-CustomVoice` share the GPU comfortably.
Models download from Hugging Face on first request. CUDA is mandatory; the
server refuses to start if PyTorch cannot access an NVIDIA GPU.

## Run

```bash
uv sync
uv run jeanclawd-voicebox
```

Verify CUDA first:

```bash
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

The server always loads Whisper and Qwen on `cuda:0`; there is no CPU fallback.

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
  -d '{"text":"Welcome back!", "emotion":"excited", "stream":false}' \
  --output welcome.wav
```

Streaming is the default. The response is signed 16-bit little-endian PCM;
the client can start playback as soon as the first sentence arrives:

```bash
curl -N -X POST http://localhost:8000/synthesize \
  -H 'content-type: application/json' \
  -d '{"text":"The first sentence starts quickly. The next sentence follows as another audio segment.","emotion":"calm"}' \
  | ffplay -f s16le -ar 24000 -i - -nodisp -autoexit
```

Use `stream:false` when a complete WAV file is preferred. Qwen's current Python
package does not expose incremental decoded audio tokens, so streaming is
sentence/segment-level: the first segment plays before later segments finish.

For per-sentence emotional control, send ordered `segments` instead of `text`:

```json
{
  "segments": [
    {"text": "I have wonderful news!", "emotion": "excited"},
    {"text": "Please take a deep breath.", "emotion": "calm"},
    {"text": "We need to fix this immediately.", "emotion": "angry", "instruction": "Speak urgently and firmly."}
  ]
}
```

The `emotion` field accepts the six built-in values (`neutral`, `happy`, `sad`,
`angry`, `excited`, and `calm`). Add an `instruction` to a segment for a more
specific delivery.

TTS emotions are natural-language instructions passed to Qwen. The default
English speakers are `Ryan` and `Aiden`; use the `speaker` and `instruction`
fields to select a voice and further control delivery.

## Configuration

```bash
VOICEBOX_WHISPER_MODEL=large-v3      # use distil-large-v3 for lower latency
VOICEBOX_WHISPER_COMPUTE_TYPE=float16
VOICEBOX_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
VOICEBOX_BEAM_SIZE=5
VOICEBOX_PORT=8000
```

The API schema is available at `/docs`; the application intentionally exposes
only the two inference routes above.
