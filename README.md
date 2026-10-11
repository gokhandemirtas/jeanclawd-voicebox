# JeanClawd Voicebox

Local FastAPI server with two endpoints:

- `POST /transcribe` — multipart field `audio`; Whisper `large-v3` transcription.
- `POST /synthesize` — JSON text plus model-independent speaker/emotion controls; returns a WAV stream.

When CUDA is available, the service prefers Qwen3-TTS `1.7B-CustomVoice`.
Without a usable CUDA device, or if Qwen fails to load, it falls back to
Supertonic-3, which runs through ONNX Runtime on CPU. Models download on first
use.

## Run

```bash
uv sync
uv run jeanclawd-voicebox
```

Verify CUDA first:

```bash
.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Whisper uses CUDA when available and otherwise falls back to CPU settings.
Supertonic provides the CPU TTS fallback with ten built-in voices (`M1`–`M5`
and `F1`–`F5`). Set `VOICEBOX_SUPERTONIC_VOICE` to choose its default voice.

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

TTS emotions are translated per backend. Qwen receives natural-language
instructions; Supertonic uses its expression tags and speed control. The
default Qwen English speakers are `Ryan` and `Aiden`. When Supertonic is active,
use an `M1`–`M5` or `F1`–`F5` speaker value, or configure
`VOICEBOX_SUPERTONIC_VOICE`. The `instruction` field is applied by Qwen and is
ignored by Supertonic because it has no free-form instruction parameter.

## Configuration

```bash
VOICEBOX_WHISPER_MODEL=large-v3      # use distil-large-v3 for lower latency
VOICEBOX_WHISPER_COMPUTE_TYPE=float16
VOICEBOX_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
VOICEBOX_SUPERTONIC_VOICE=M1
VOICEBOX_BEAM_SIZE=5
VOICEBOX_PORT=8000
```

The API schema is available at `/docs`; the application intentionally exposes
only the two inference routes above.
