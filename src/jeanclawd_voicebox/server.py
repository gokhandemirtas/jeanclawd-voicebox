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
from enum import Enum
from pathlib import Path
from threading import Lock, Thread
from typing import Annotated

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
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is mandatory for JeanClawd Voicebox, but PyTorch cannot access an NVIDIA GPU. "
            "Check the NVIDIA driver, CUDA-enabled PyTorch installation, and CUDA_VISIBLE_DEVICES."
        )
    return "cuda"


DEVICE = _device()
WHISPER_MODEL = os.getenv("VOICEBOX_WHISPER_MODEL", "large-v3")
TTS_MODEL = os.getenv(
    "VOICEBOX_TTS_MODEL",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
)
QWEN_SAMPLE_RATE = 24_000
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

                dtype = torch.bfloat16
                device_map = "cuda:0"
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


def _start_debug_pcm_playback() -> subprocess.Popen[bytes] | None:
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
        str(QWEN_SAMPLE_RATE),
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
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is mandatory for JeanClawd Voicebox, but no usable NVIDIA GPU was detected."
        )
    logger.info("CUDA ready: %s", torch.cuda.get_device_name(0))
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

        model = await run_in_threadpool(_get_tts)
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
            sample_rate = QWEN_SAMPLE_RATE
            with _inference_lock, torch.inference_mode():
                for segment in segments:
                    wavs, sample_rate = model.generate_custom_voice(
                        text=segment.text,
                        language=_qwen_language(request.language),
                        speaker=request.speaker,
                        instruct=_segment_instruction(segment, request),
                        non_streaming_mode=True,
                    )
                    audio = wavs[0]
                    if isinstance(audio, torch.Tensor):
                        audio = audio.detach().cpu().numpy()
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
        sample_rate = QWEN_SAMPLE_RATE
        debug_player = None
        try:
            with _inference_lock:
                stream_index = 0
                for segment in segments:
                    instruction = _segment_instruction(segment, request)
                    for text_chunk in _split_text_for_streaming(segment.text):
                        with torch.inference_mode():
                            wavs, sample_rate = model.generate_custom_voice(
                                text=text_chunk,
                                language=_qwen_language(request.language),
                                speaker=request.speaker,
                                instruct=instruction,
                                # This selects Qwen's lower-latency text path. The
                                # package still returns one waveform per segment.
                                non_streaming_mode=False,
                            )
                        pcm = _pcm16(wavs[0])
                        stream_index += 1
                        logger.info(
                            "[%s] streamed segment=%d emotion=%s characters=%d",
                            request_id,
                            stream_index,
                            segment.emotion.value,
                            len(text_chunk),
                        )
                        if _debug_play_audio and debug_player is None:
                            debug_player = _start_debug_pcm_playback()
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
        media_type=f"audio/L16;rate={QWEN_SAMPLE_RATE};channels=1",
        headers={
            "X-Request-ID": request_id,
            "X-Audio-Sample-Rate": str(QWEN_SAMPLE_RATE),
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
