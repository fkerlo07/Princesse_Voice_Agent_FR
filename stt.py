"""
stt.py — Speech-to-text: faster-whisper FR transcription + OpenWakeWord detection
"""
import asyncio
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openwakeword.model import Model as OWWModel

from config import (
    MIC_RATE, MIC_DTYPE, MIC_BLOCKSIZE,
    OWW_MODEL, OWW_THRESHOLD,
    LISTEN_TIMEOUT_S,
    WHISPER_MODEL,
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
_SILENCE_CHUNKS = 12    # 12 × 80ms = ~1s of silence → commit early
_MIN_SPEECH_CHUNKS = 4  # ignore bursts shorter than 4 × 80ms = 320ms


# ── Whisper STT worker ────────────────────────────────────────────────────────

async def stt_worker(
    audio_q:   asyncio.Queue,
    text_q:    asyncio.Queue,
    state_ref: list,
) -> None:
    print("Chargement Whisper (faster-whisper small)...", flush=True)
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("Prêt. Dites 'Hey Jarvis' pour m'activer.\n", flush=True)
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
            rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
            silence_count = silence_count + 1 if rms < _SILENCE_RMS else 0

            elapsed = time.monotonic() - listen_start
            has_enough = len(audio_buffer) >= _MIN_SPEECH_CHUNKS
            should_commit = has_enough and (
                silence_count >= _SILENCE_CHUNKS
                or elapsed >= LISTEN_TIMEOUT_S
            )

            if should_commit:
                state_ref[0] = State.THINKING
                await broadcast({"type": "state", "state": "thinking"})

                audio_np = np.frombuffer(b"".join(audio_buffer), dtype=np.int16)
                audio_f32 = audio_np.astype(np.float32) / 32768.0

                listen_start = 0.0
                audio_buffer = []
                silence_count = 0

                def transcribe(audio=audio_f32):
                    segments, _ = model.transcribe(
                        audio,
                        language="fr",
                        beam_size=5,
                        vad_filter=True,
                        vad_parameters={
                            "threshold": 0.30,            # more sensitive — catches short words
                            "min_speech_duration_ms": 80, # accept single syllables
                            "min_silence_duration_ms": 200,
                        },
                    )
                    return " ".join(s.text for s in segments).strip()

                text = await asyncio.get_event_loop().run_in_executor(None, transcribe)
                print(f"[Whisper]: {text!r}", flush=True)

                if text:
                    await text_q.put(text)
                else:
                    state_ref[0] = State.SLEEPING
                    await broadcast({"type": "state", "state": "sleeping"})
                    print("💤 Rien compris → retour en veille.", flush=True)
        else:
            audio_buffer = []
            listen_start = 0.0
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
