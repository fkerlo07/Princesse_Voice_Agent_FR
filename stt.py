"""
stt.py — Speech-to-text: Gemma 4 E2B (transformers/MPS) + OpenWakeWord detection.
Audio is accumulated after wake word, then sent to gemma4_local.transcribe().
"""
import asyncio
import time

import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel

import gemma4_local
from config import (
    MIC_BLOCKSIZE, MIC_DTYPE, MIC_RATE,
    OWW_MODEL, OWW_THRESHOLD, LISTEN_TIMEOUT_S,
)
from ui_server import broadcast, broadcast_sync


# ── Agent state ───────────────────────────────────────────────────────────────
from enum import Enum, auto


class State(Enum):
    SLEEPING  = auto()
    LISTENING = auto()
    THINKING  = auto()


# ── Silence detection ─────────────────────────────────────────────────────────
_SILENCE_RMS      = 400   # int16 RMS below this = silence
_SILENCE_CHUNKS   = 12    # 12 × 80 ms ≈ 1 s of silence → commit
_MIN_SPEECH_CHUNKS = 4    # ignore bursts < 4 × 80 ms = 320 ms


# ── STT worker ────────────────────────────────────────────────────────────────

async def stt_worker(
    audio_q:   asyncio.Queue,
    text_q:    asyncio.Queue,
    state_ref: list,
) -> None:
    print("STT prêt (Gemma 4 E2B / MPS). Dites 'Hey Jarvis' pour m'activer.\n", flush=True)
    await broadcast({"type": "state", "state": "sleeping"})

    audio_buffer: list[bytes] = []
    listen_start  = 0.0
    silence_count = 0

    while True:
        data = await audio_q.get()
        if data is None:
            audio_q.task_done()
            break

        state = state_ref[0]

        if state == State.LISTENING:
            if listen_start == 0.0:
                listen_start  = time.monotonic()
                audio_buffer  = []
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

                # Run blocking transcription in executor (does not block event loop)
                text = await asyncio.get_event_loop().run_in_executor(
                    None, gemma4_local.transcribe, pcm
                )
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


# ── Microphone + OpenWakeWord ─────────────────────────────────────────────────

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
