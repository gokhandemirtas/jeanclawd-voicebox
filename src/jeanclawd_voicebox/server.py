"""Two-endpoint local STT/TTS service."""

from __future__ import annotations

import io
import argparse
import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Annotated

import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("jeanclawd_voicebox")


def _configure_logging() -> None:
    log_path = Path(os.getenv("VOICEBOX_LOG_FILE", "logs/voicebox.log"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(handler, logging.FileHandler) for handler in logger.handlers):
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(threadName)s]: %(message)s"
        )
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = True


_configure_logging()


def _device() -> str:
    requested = os.getenv("VOICEBOX_DEVICE", "auto").lower()
    if requested in {"cuda", "cpu"}:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _device()
WHISPER_MODEL = os.getenv("VOICEBOX_WHISPER_MODEL", "large-v3")
TTS_MODEL = os.getenv(
    "VOICEBOX_TTS_MODEL",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
)
_whisper = None
_tts = None
_model_lock = Lock()
_inference_lock = Lock()
_debug_play_audio = False


def _get_whisper():
    global _whisper
    if _whisper is None:
        with _model_lock:
            if _whisper is None:
                from faster_whisper import WhisperModel

                compute_type = os.getenv(
                    "VOICEBOX_WHISPER_COMPUTE_TYPE",
                    "float16" if DEVICE == "cuda" else "int8",
                )
                _whisper = WhisperModel(
                    WHISPER_MODEL,
                    device=DEVICE,
                    compute_type=compute_type,
                    cpu_threads=int(os.getenv("VOICEBOX_CPU_THREADS", "4")),
                )
    return _whisper


def _get_tts():
    global _tts
    if _tts is None:
        with _model_lock:
            if _tts is None:
                from qwen_tts import Qwen3TTSModel

                dtype = torch.bfloat16 if DEVICE == "cuda" else torch.float32
                device_map = "cuda:0" if DEVICE == "cuda" else "cpu"
                _tts = Qwen3TTSModel.from_pretrained(
                    TTS_MODEL,
                    device_map=device_map,
                    dtype=dtype,
                )
    return _tts


class Emotion(str, Enum):
    neutral = "neutral"
    happy = "happy"
    sad = "sad"
    angry = "angry"
    excited = "excited"
    calm = "calm"


class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    emotion: Emotion = Emotion.neutral
    speaker: str = Field(default="Ryan", min_length=1, max_length=64)
    instruction: str | None = Field(default=None, max_length=500)
    language: str = Field(default="English", min_length=2, max_length=32)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        # Normalize line endings so multiline JSON/client requests are stable
        # across TTS backends while retaining words and punctuation.
        normalized = " ".join(line.strip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip())
        if not normalized:
            raise ValueError("text must contain at least one non-whitespace character")
        return normalized


def _emotion_instruction(emotion: Emotion) -> str:
    return {
        Emotion.neutral: "Speak naturally and clearly.",
        Emotion.happy: "Speak in a warm, happy, upbeat tone.",
        Emotion.sad: "Speak gently with a sincere, sad, reflective tone.",
        Emotion.angry: "Speak with controlled anger and firm intensity.",
        Emotion.excited: "Speak with energetic excitement and bright enthusiasm.",
        Emotion.calm: "Speak calmly, warmly, and meditatively.",
    }[emotion]


def _qwen_language(language: str) -> str:
    names = {
        "en": "English",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "de": "German",
        "fr": "French",
        "ru": "Russian",
        "pt": "Portuguese",
        "es": "Spanish",
        "it": "Italian",
    }
    return names.get(language.lower(), language)


def _autoplay_debug_audio(audio: bytes) -> None:
    """Play generated audio asynchronously when the temporary debug flag is on."""
    if not _debug_play_audio:
        return

    player = shutil.which("ffplay") or shutil.which("mpv") or shutil.which("paplay") or shutil.which("aplay")
    if player is None:
        logger.warning("debug playback requested, but no ffplay/mpv/paplay/aplay executable was found")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file.write(audio)
        audio_path = audio_file.name

    if os.path.basename(player) == "ffplay":
        command = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]
    elif os.path.basename(player) == "mpv":
        command = [player, "--no-video", "--really-quiet", audio_path]
    else:
        command = [player, audio_path]

    def play_and_cleanup() -> None:
        try:
            result = subprocess.run(command, check=False, capture_output=True, text=True)
            if result.returncode != 0:
                error = (result.stderr or result.stdout or "unknown player error").strip()
                logger.warning(
                    "debug playback failed: player=%s exit_code=%s error=%s",
                    os.path.basename(player),
                    result.returncode,
                    error[-1000:],
                )
            else:
                logger.info("debug playback finished: player=%s", os.path.basename(player))
        finally:
            try:
                os.unlink(audio_path)
            except FileNotFoundError:
                pass

    Thread(target=play_and_cleanup, daemon=True).start()
    logger.info("debug playback started with %s", os.path.basename(player))


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    global _whisper, _tts
    _whisper = None
    _tts = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="JeanClawd Voicebox",
    version="0.1.0",
    description="Local Whisper STT and expressive Qwen3-TTS",
    lifespan=lifespan,
)


@app.post("/transcribe")
async def transcribe(
    audio: Annotated[UploadFile, File(description="Audio file accepted by ffmpeg/faster-whisper")],
):
    """Transcribe an uploaded audio file with Whisper large-v3."""
    suffix = os.path.splitext(audio.filename or "audio.bin")[1] or ".bin"
    data = await audio.read()
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded audio file is empty")

    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    logger.info("[%s] started transcription: %s", request_id, audio.filename or "audio")
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as source:
            source.write(data)
            source.flush()
            with _inference_lock:
                segments, info = _get_whisper().transcribe(
                    source.name,
                    beam_size=int(os.getenv("VOICEBOX_BEAM_SIZE", "5")),
                    vad_filter=True,
                    condition_on_previous_text=True,
                )
                segment_list = list(segments)
        text = " ".join(segment.text.strip() for segment in segment_list).strip()
        logger.info("[%s] started -> transcribed in %.1f seconds", request_id, time.perf_counter() - started)
        return {
            "request_id": request_id,
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability,
            "duration_seconds": info.duration,
            "segments": [
                {"start": s.start, "end": s.end, "text": s.text.strip()}
                for s in segment_list
            ],
        }
    except Exception as exc:
        logger.exception("[%s] transcription failed", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed; request_id={request_id}. See the server log.",
        ) from exc


@app.post("/synthesize", response_class=StreamingResponse)
async def synthesize(request: SynthesisRequest):
    """Generate a WAV file with Qwen's natural-language style control."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    logger.info(
        "[%s] started synthesis: emotion=%s characters=%d",
        request_id,
        request.emotion.value,
        len(request.text),
    )
    instruction = _emotion_instruction(request.emotion)
    if request.instruction:
        instruction = f"{instruction} {request.instruction}"
    try:
        with _inference_lock:
            model = _get_tts()
            with torch.inference_mode():
                wavs, sample_rate = model.generate_custom_voice(
                    text=request.text,
                    language=_qwen_language(request.language),
                    speaker=request.speaker,
                    instruct=instruction,
                )

        audio = wavs[0]
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        output = io.BytesIO()
        sf.write(output, audio, int(sample_rate), format="WAV", subtype="PCM_16")
        output.seek(0)
        _autoplay_debug_audio(output.getvalue())
        logger.info("[%s] started -> generated in %.1f seconds", request_id, time.perf_counter() - started)
        return StreamingResponse(
            output,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "inline; filename=voicebox.wav",
                "X-Request-ID": request_id,
            },
        )
    except Exception as exc:
        logger.exception("[%s] synthesis failed", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed; request_id={request_id}. See the server log.",
            headers={"X-Request-ID": request_id},
        ) from exc


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="JeanClawd Voicebox STT/TTS server")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="temporarily autoplay generated TTS audio using a local audio player",
    )
    args = parser.parse_args()
    global _debug_play_audio
    _debug_play_audio = args.debug
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    uvicorn.run(
        "jeanclawd_voicebox.server:app",
        host=os.getenv("VOICEBOX_HOST", "0.0.0.0"),
        port=int(os.getenv("VOICEBOX_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
