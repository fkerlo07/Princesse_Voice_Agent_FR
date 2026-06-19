"""
stt.py — Speech-to-text: Gemma 4 E2B audio transcription + OpenWakeWord detection.
Audio is accumulated after wake word, encoded as WAV, then sent to Gemma 4 E2B
via Ollama's OpenAI-compatible /v1/chat/completions endpoint.
"""
import asyncio
import base64
import io
import time
import wave

import aiohttp
import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel

from config import (
    GEMMA4_MODEL, MIC_BLOCKSIZE, MIC_DTYPE, MIC_RATE,
    OLLAMA_BASE_URL, OWW_MODEL, OWW_THRESHOLD, LISTEN_TIMEOUT_S,
)
from ui_server import broadcast, broadcast_sync


# ── Agent state ───────────────────────────────────────────────────────────────
from enum import Enum, auto


class State(Enum):
    SLEEPING  = auto()
    LISTENING = auto()
    THINKING  = auto()


# ── Silence detection ─────────────────────────────────────────────────────────
_SILENCE_RMS    = 400   # int16 RMS below this = silence
_SILENCE_CHUNKS = 12    # 12 × 80 ms = ~1 s of silence → commit
_MIN_SPEECH_CHUNKS = 4  # ignore bursts < 4 × 80 ms = 320 ms


# ── Audio helpers ─────────────────────────────────────────────────────────────

def _pcm_to_wav_b64(pcm_bytes: bytes, rate: int = MIC_RATE) -> str:
    """Encode raw int16 PCM bytes as a base64 WAV string."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # int16 = 2 bytes
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


async def _transcribe_gemma4(audio_bytes: bytes) -> str:
    """
    Send accumulated audio to Gemma 4 E2B via Ollama's OpenAI-compatible endpoint.
    Returns the transcribed French text, or "" if nothing was understood.
    """
    wav_b64 = _pcm_to_wav_b64(audio_bytes)

    payload = {
        "model": GEMMA4_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": wav_b64, "format": "wav"},
                },
                {
                    "type": "text",
                    "text": (
                        "Transcris exactement ce que dit la personne en français. "
                        "Réponds UNIQUEMENT avec la transcription brute, sans ponctuation "
                        "ajoutée, sans majuscule inutile, sans commentaire."
                    ),
                },
            ],
        }],
        "stream": False,
        "temperature": 0,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=20),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[Gemma4 STT error]: {e}", flush=True)
        return ""


# ── STT worker ────────────────────────────────────────────────────────────────

async def stt_worker(
    audio_q:   asyncio.Queue,
    text_q:    asyncio.Queue,
    state_ref: list,
) -> None:
    print("STT prêt (Gemma 4 E2B audio). Dites 'Hey Jarvis' pour m'activer.\n", flush=True)
    await broadcast({"type": "state", "state": "sleeping"})

    audio_buffer: list[bytes] = []
    listen_start = 0.0
    silence_count = 0

    while True:
        data = await audio_q.get()
        if data is None:
            audio_q.task_done()
            break

        state = state_ref[0]

        if state == State.LISTENING:
            if listen_start == 0.0:
                listen_start = time.monotonic()
                audio_buffer = []
                silence_count = 0

            audio_buffer.append(data)

            chunk = np.frombuffer(data, dtype=np.int16)
            rms   = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            silence_count = silence_count + 1 if rms < _SILENCE_RMS else 0

            elapsed    = time.monotonic() - listen_start
            has_enough = len(audio_buffer) >= _MIN_SPEECH_CHUNKS
            should_commit = has_enough and (
                silence_count >= _SILENCE_CHUNKS
                or elapsed >= LISTEN_TIMEOUT_S
            )

            if should_commit:
                state_ref[0] = State.THINKING
                await broadcast({"type": "state", "state": "thinking"})

                pcm = b"".join(audio_buffer)
                listen_start  = 0.0
                audio_buffer  = []
                silence_count = 0

                text = await _transcribe_gemma4(pcm)
                print(f"[Gemma4 STT]: {text!r}", flush=True)

                if text:
                    await text_q.put(text)
                else:
                    state_ref[0] = State.SLEEPING
                    await broadcast({"type": "state", "state": "sleeping"})
                    print("💤 Rien compris → retour en veille.", flush=True)
        else:
            audio_buffer  = []
            listen_start  = 0.0
            silence_count = 0

        audio_q.task_done()


# ── Microphone + OWW ─────────────────────────────────────────────────────────

def load_oww() -> OWWModel:
    print("Chargement OpenWakeWord...", flush=True)
    return OWWModel(wakeword_models=[OWW_MODEL], inference_framework="onnx")


def start_microphone(
    audio_q:   asyncio.Queue,
    loop:      asyncio.AbstractEventLoop,
    oww:       OWWModel,
    state_ref: list,
    tts_abort,
) -> sd.RawInputStream:

    async def _on_wake(score: float) -> None:
        if state_ref[0] == State.THINKING:
            tts_abort.set()
            print(f"\n🛑 Interrompu ! J'écoute... (score={score:.2f})", flush=True)
        else:
            print(f"🟢 Jarvis activé (score={score:.2f}) — j'écoute...", flush=True)
        state_ref[0] = State.LISTENING
        await broadcast({"type": "state", "state": "listening"})

    def mic_callback(indata: bytes, frames, t, status):
        data     = bytes(indata)
        audio_np = np.frombuffer(data, dtype=np.int16)

        if state_ref[0] in (State.SLEEPING, State.THINKING):
            scores = oww.predict(audio_np)
            score  = max(
                scores.get("hey_jarvis",      0),
                scores.get("hey_jarvis_v0.1", 0),
            )
            if score >= OWW_THRESHOLD:
                oww.reset()
                asyncio.run_coroutine_threadsafe(_on_wake(score), loop)
        else:
            oww.predict(audio_np)

        loop.call_soon_threadsafe(audio_q.put_nowait, data)

    stream = sd.RawInputStream(
        samplerate=MIC_RATE,
        blocksize=MIC_BLOCKSIZE,
        dtype=MIC_DTYPE,
        channels=1,
        callback=mic_callback,
    )
    stream.start()
    return stream
