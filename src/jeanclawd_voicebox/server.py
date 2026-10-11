"""Two-endpoint local STT/TTS service."""

from __future__ import annotations

import io
import argparse
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Annotated, Any

import soundfile as sf
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator, model_validator

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
    if not any(type(handler) is logging.StreamHandler for handler in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


_configure_logging()


def _device() -> str:
    """Choose the preferred device without making CUDA a startup requirement."""
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE = _device()
WHISPER_MODEL = os.getenv("VOICEBOX_WHISPER_MODEL", "large-v3")
TTS_MODEL = os.getenv(
    "VOICEBOX_TTS_MODEL",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
)
QWEN_SAMPLE_RATE = 24_000
SUPERTONIC_VOICE = os.getenv("VOICEBOX_SUPERTONIC_VOICE", "M1")
_whisper = None
_tts = None
_tts_backend = None
_model_lock = Lock()
_inference_lock = Lock()
_debug_play_audio = False


class Emotion(str, Enum):
    neutral = "neutral"
    happy = "happy"
    sad = "sad"
    angry = "angry"
    excited = "excited"
    calm = "calm"


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


class TTSBackend(str, Enum):
    qwen = "qwen"
    supertonic = "supertonic"


@dataclass(frozen=True)
class TTSRuntime:
    backend: TTSBackend
    model: Any
    sample_rate: int


# These are the model-facing emotion names. The reverse maps are built below
# so integrations can translate either direction without hard-coding a model.
# Supertonic has expression tags and speed control rather than Qwen's
# natural-language instruction parameter. These labels describe the closest
# available expression profile used by the adapter.
MODEL_EMOTION_MAP: dict[TTSBackend, dict[Emotion, str]] = {
    TTSBackend.qwen: {
        Emotion.neutral: "neutral",
        Emotion.happy: "happy",
        Emotion.sad: "sad",
        Emotion.angry: "angry",
        Emotion.excited: "excited",
        Emotion.calm: "calm",
    },
    TTSBackend.supertonic: {
        Emotion.neutral: "neutral",
        Emotion.happy: "laugh",
        Emotion.sad: "sigh",
        Emotion.angry: "intense",
        Emotion.excited: "laugh",
        Emotion.calm: "gentle",
    },
}
REVERSE_MODEL_EMOTION_MAP: dict[TTSBackend, dict[str, Emotion]] = {
    backend: {model_emotion: emotion for emotion, model_emotion in mapping.items()}
    for backend, mapping in MODEL_EMOTION_MAP.items()
}

# Supertonic supports inline expression tags and speed control. The tags are
# intentionally conservative because unsupported tags can become audible text.
SUPERTONIC_EMOTION_CONTROLS: dict[Emotion, tuple[str, float]] = {
    Emotion.neutral: ("", 1.05),
    Emotion.happy: ("<laugh>", 1.10),
    Emotion.sad: ("<sigh>", 0.90),
    Emotion.angry: ("", 1.15),
    Emotion.excited: ("<laugh>", 1.20),
    Emotion.calm: ("<sigh>", 0.85),
}


def _get_tts() -> TTSRuntime:
    global _tts, _tts_backend
    if _tts is None:
        with _model_lock:
            if _tts is None:
                if DEVICE == "cuda":
                    try:
                        from qwen_tts import Qwen3TTSModel

                        _tts = Qwen3TTSModel.from_pretrained(
                            TTS_MODEL,
                            device_map="cuda:0",
                            dtype=torch.bfloat16,
                        )
                        _tts_backend = TTSRuntime(TTSBackend.qwen, _tts, QWEN_SAMPLE_RATE)
                        logger.info("TTS backend selected: qwen device=cuda")
                    except Exception:
                        logger.exception("Qwen could not be loaded; falling back to Chatterbox")

                if _tts_backend is None:
                    try:
                        from supertonic import TTS
                    except ImportError as exc:
                        raise RuntimeError(
                            "Supertonic is required because Qwen is unavailable, but it is not installed. "
                            "Install the supertonic package in the runtime environment used by Voicebox."
                        ) from exc

                    _tts = TTS(auto_download=True)
                    _tts_backend = TTSRuntime(
                        TTSBackend.supertonic,
                        _tts,
                        44_100,
                    )
                    logger.info("TTS backend selected: supertonic")
    if _tts_backend is None:
        raise RuntimeError("No TTS backend is available")
    return _tts_backend


class SynthesisSegment(BaseModel):
    """One independently styled piece of a synthesis request."""

    text: str = Field(min_length=1, max_length=10_000)
    emotion: Emotion = Emotion.neutral
    instruction: str | None = Field(default=None, max_length=500)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        normalized = " ".join(
            line.strip()
            for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            if line.strip()
        )
        if not normalized:
            raise ValueError("segment text must contain at least one non-whitespace character")
        return normalized


class SynthesisRequest(BaseModel):
    text: str | None = Field(default=None, max_length=10_000)
    emotion: Emotion = Emotion.neutral
    speaker: str = Field(default="Ryan", min_length=1, max_length=64)
    instruction: str | None = Field(default=None, max_length=500)
    language: str = Field(default="English", min_length=2, max_length=32)
    stream: bool = Field(default=True, description="Stream signed 16-bit PCM segments")
    segments: list[SynthesisSegment] | None = Field(default=None, max_length=100)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        # Normalize line endings so multiline JSON/client requests are stable
        # across TTS backends while retaining words and punctuation.
        normalized = " ".join(line.strip() for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip())
        if not normalized:
            raise ValueError("text must contain at least one non-whitespace character")
        return normalized

    @model_validator(mode="after")
    def require_text_or_segments(self):
        if self.text is None and not self.segments:
            raise ValueError("provide text or at least one synthesis segment")
        if self.text is not None and self.segments:
            raise ValueError("use either text or segments, not both")
        return self


def _emotion_instruction(emotion: Emotion) -> str:
    return {
        Emotion.neutral: "Speak naturally and clearly.",
        Emotion.happy: "Speak in a warm, happy, upbeat tone.",
        Emotion.sad: "Speak gently with a sincere, sad, reflective tone.",
        Emotion.angry: "Speak with controlled anger and firm intensity.",
        Emotion.excited: "Speak with energetic excitement and bright enthusiasm.",
        Emotion.calm: "Speak calmly, warmly, and meditatively.",
    }[emotion]


def _model_emotion(backend: TTSBackend, emotion: Emotion) -> str:
    """Return the valid model-facing emotion label for a canonical emotion."""
    return MODEL_EMOTION_MAP[backend][emotion]


def _canonical_emotion(backend: TTSBackend, model_emotion: str) -> Emotion:
    """Translate a model-facing emotion label back to the public emotion enum."""
    try:
        return REVERSE_MODEL_EMOTION_MAP[backend][model_emotion]
    except KeyError as exc:
        raise ValueError(f"unknown {backend.value} emotion: {model_emotion}") from exc


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


def _supertonic_language(language: str) -> str:
    """Convert the public language field to Supertonic's ISO language code."""
    return {
        "english": "en",
        "chinese": "zh",
        "japanese": "ja",
        "korean": "ko",
        "german": "de",
        "french": "fr",
        "russian": "ru",
        "portuguese": "pt",
        "spanish": "es",
        "italian": "it",
    }.get(language.lower(), language.lower() if len(language) == 2 else "na")


def _split_text_for_streaming(text: str, max_chars: int = 240) -> list[str]:
    """Make playable segments without splitting ordinary sentences."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for sentence in sentences or [text]:
        if current and len(current) + len(sentence) + 1 > max_chars:
            chunks.append(current)
            current = ""
        if len(sentence) <= max_chars:
            current = f"{current} {sentence}".strip()
            continue
        words = sentence.split()
        for word in words:
            if current and len(current) + len(word) + 1 > max_chars:
                chunks.append(current)
                current = ""
            current = f"{current} {word}".strip()
    if current:
        chunks.append(current)
    return chunks


def _synthesis_segments(request: SynthesisRequest) -> list[SynthesisSegment]:
    if request.segments is not None:
        return request.segments
    return [SynthesisSegment(text=request.text or "", emotion=request.emotion, instruction=request.instruction)]


def _segment_instruction(segment: SynthesisSegment, request: SynthesisRequest) -> str:
    instruction = _emotion_instruction(segment.emotion)
    if segment.instruction:
        instruction = f"{instruction} {segment.instruction}"
    if request.instruction and request.segments:
        instruction = f"{instruction} {request.instruction}"
    return instruction


def _audio_array(audio: Any) -> np.ndarray:
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio)


def _generate_audio(
    runtime: TTSRuntime,
    text: str,
    segment: SynthesisSegment,
    request: SynthesisRequest,
    *,
    non_streaming_mode: bool,
) -> tuple[np.ndarray, int]:
    if runtime.backend is TTSBackend.qwen:
        wavs, sample_rate = runtime.model.generate_custom_voice(
            text=text,
            language=_qwen_language(request.language),
            speaker=request.speaker,
            instruct=_segment_instruction(segment, request),
            non_streaming_mode=non_streaming_mode,
        )
        return _audio_array(wavs[0]), int(sample_rate)

    expression_tag, speed = SUPERTONIC_EMOTION_CONTROLS[segment.emotion]
    expressive_text = f"{expression_tag} {text}".strip() if expression_tag else text
    voice_name = request.speaker if request.speaker in {f"M{i}" for i in range(1, 6)} | {f"F{i}" for i in range(1, 6)} else SUPERTONIC_VOICE
    style = runtime.model.get_voice_style(voice_name=voice_name)
    logger.debug(
        "supertonic emotion=%s mapped_to=%s expression=%s speed=%.2f voice=%s",
        segment.emotion.value,
        _model_emotion(TTSBackend.supertonic, segment.emotion),
        expression_tag or "none",
        speed,
        voice_name,
    )
    wav, _ = runtime.model.synthesize(
        expressive_text,
        voice_style=style,
        lang=_supertonic_language(request.language),
        speed=speed,
    )
    return _audio_array(wav), runtime.sample_rate


def _pcm16(audio: np.ndarray) -> bytes:
    samples = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (samples * 32767.0).astype("<i2").tobytes()


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    sf.write(
        output,
        np.frombuffer(pcm, dtype="<i2"),
        sample_rate,
        format="WAV",
        subtype="PCM_16",
    )
    return output.getvalue()


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
        command = [player, "-nodisp", "-autoexit", "-loglevel", "quiet", "-volume", "100", audio_path]
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


def _start_debug_pcm_playback(sample_rate: int) -> subprocess.Popen[bytes] | None:
    """Start ffplay for raw PCM so debug streaming can play each segment immediately."""
    player = shutil.which("ffplay")
    if player is None:
        logger.warning("streaming debug playback requires ffplay, but it was not found")
        return None

    command = [
        player,
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-i",
        "-",
        "-nodisp",
        "-autoexit",
        "-loglevel",
        "quiet",
        "-volume",
        "100",
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    logger.info("streaming debug playback started with ffplay")
    return process


@asynccontextmanager
async def lifespan(_: FastAPI):
    if DEVICE == "cuda":
        logger.info("CUDA available: %s; Qwen will be preferred", torch.cuda.get_device_name(0))
    else:
        logger.info("CUDA unavailable; Chatterbox CPU fallback will be used for TTS")
    yield
    global _whisper, _tts, _tts_backend
    _whisper = None
    _tts = None
    _tts_backend = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="JeanClawd Voicebox",
    version="0.1.0",
    description="Local Whisper STT with automatic Qwen or Chatterbox TTS",
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
    logger.info(
        "[%s] request received, processing transcription: %s",
        request_id,
        audio.filename or "audio",
    )
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
    """Generate audio as streamed PCM, or a complete WAV when stream=false."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    segments = _synthesis_segments(request)
    character_count = sum(len(segment.text) for segment in segments)
    logger.info(
        "[%s] request received, processing synthesis: emotion=%s characters=%d",
        request_id,
        ",".join(segment.emotion.value for segment in segments),
        character_count,
    )

    try:
        # Load before returning StreamingResponse so download/model errors still
        # produce a useful HTTP 500 rather than a broken audio stream.
        from starlette.concurrency import run_in_threadpool

        runtime = await run_in_threadpool(_get_tts)
    except Exception as exc:
        logger.exception("[%s] synthesis failed", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed; request_id={request_id}. See the server log.",
            headers={"X-Request-ID": request_id},
        ) from exc

    if not request.stream:
        try:
            pcm_parts: list[bytes] = []
            sample_rate = runtime.sample_rate
            with _inference_lock, torch.inference_mode():
                for segment in segments:
                    audio, sample_rate = _generate_audio(
                        runtime,
                        segment.text,
                        segment,
                        request,
                        non_streaming_mode=True,
                    )
                    pcm_parts.append(_pcm16(audio))
            wav = _wav_bytes(b"".join(pcm_parts), int(sample_rate))
            _autoplay_debug_audio(wav)
            logger.info("[%s] started -> generated in %.1f seconds", request_id, time.perf_counter() - started)
            return StreamingResponse(
                io.BytesIO(wav),
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

    def pcm_stream():
        sample_rate = runtime.sample_rate
        debug_player = None
        try:
            with _inference_lock:
                stream_index = 0
                for segment in segments:
                    for text_chunk in _split_text_for_streaming(segment.text):
                        with torch.inference_mode():
                            audio, sample_rate = _generate_audio(
                                runtime,
                                text_chunk,
                                segment,
                                request,
                                non_streaming_mode=False,
                            )
                        pcm = _pcm16(audio)
                        stream_index += 1
                        logger.info(
                            "[%s] streamed segment=%d emotion=%s characters=%d",
                            request_id,
                            stream_index,
                            segment.emotion.value,
                            len(text_chunk),
                        )
                        if _debug_play_audio and debug_player is None:
                            debug_player = _start_debug_pcm_playback(sample_rate)
                        if debug_player is not None and debug_player.stdin is not None:
                            try:
                                debug_player.stdin.write(pcm)
                                debug_player.stdin.flush()
                            except BrokenPipeError:
                                logger.warning("streaming debug playback stopped before segment=%d", stream_index)
                                debug_player.stdin.close()
                                debug_player.wait()
                                debug_player = None
                        yield pcm
            logger.info("[%s] started -> generated in %.1f seconds", request_id, time.perf_counter() - started)
        except Exception:
            logger.exception("[%s] synthesis stream failed", request_id)
        finally:
            if debug_player is not None:
                if debug_player.stdin is not None:
                    debug_player.stdin.close()
                debug_player.wait()
                logger.info("streaming debug playback finished: player=ffplay")

    return StreamingResponse(
        pcm_stream(),
        media_type=f"audio/L16;rate={runtime.sample_rate};channels=1",
        headers={
            "X-Request-ID": request_id,
            "X-Audio-Sample-Rate": str(runtime.sample_rate),
            "X-Audio-Format": "s16le",
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
