"""
stt.py — Speech-to-text: Vosk FR transcription + OpenWakeWord detection
"""
import asyncio
import json
import time

import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel
from vosk import KaldiRecognizer
from vosk import Model as VoskModel

from config import (
    MIC_RATE, MIC_DTYPE, MIC_BLOCKSIZE,
    OWW_MODEL, OWW_THRESHOLD,
    VOSK_MODEL_FR, LISTEN_TIMEOUT_S,
)
from ui_server import broadcast, broadcast_sync


# ── Agent state ───────────────────────────────────────────────────────────────
from enum import Enum, auto


class State(Enum):
    SLEEPING  = auto()
    LISTENING = auto()
    THINKING  = auto()


# ── Vosk FR worker ────────────────────────────────────────────────────────────

async def stt_worker(
    audio_q:   asyncio.Queue,
    text_q:    asyncio.Queue,
    state_ref: list,
) -> None:
    """Transcribes commands with French Vosk when state is LISTENING.
    Wake-word detection happens in the mic callback thread (see start_microphone).
    """
    print("Chargement Vosk FR...", flush=True)
    rec_fr = KaldiRecognizer(VoskModel(VOSK_MODEL_FR), MIC_RATE)
    rec_fr.SetWords(False)
    print("Prêt. Dites 'Hey Jarvis' pour m'activer.\n", flush=True)
    await broadcast({"type": "state", "state": "sleeping"})

    listen_start = 0.0

    while True:
        data = await audio_q.get()
        if data is None:
            audio_q.task_done()
            break

        state = state_ref[0]

        if state == State.LISTENING:
            if listen_start == 0.0:
                listen_start = time.monotonic()

            if rec_fr.AcceptWaveform(data):
                text = json.loads(rec_fr.Result()).get("text", "").strip()
                print(f"[FR final]: {text!r}", flush=True)
                if text:
                    listen_start = 0.0
                    rec_fr.Reset()
                    state_ref[0] = State.THINKING
                    await broadcast({"type": "state", "state": "thinking"})
                    await text_q.put(text)
            else:
                partial = json.loads(rec_fr.PartialResult()).get("partial", "")
                if partial:
                    print(f"[FR partial]: {partial}", flush=True)

                # Timeout — force final result after LISTEN_TIMEOUT_S seconds
                if time.monotonic() - listen_start > LISTEN_TIMEOUT_S:
                    text = json.loads(rec_fr.FinalResult()).get("text", "").strip()
                    print(f"[FR timeout]: {text!r}", flush=True)
                    listen_start = 0.0
                    rec_fr.Reset()
                    if text:
                        state_ref[0] = State.THINKING
                        await broadcast({"type": "state", "state": "thinking"})
                        await text_q.put(text)
                    else:
                        state_ref[0] = State.SLEEPING
                        await broadcast({"type": "state", "state": "sleeping"})
                        print("💤 Timeout → retour en veille.", flush=True)
        else:
            listen_start = 0.0
            rec_fr.Reset()

        audio_q.task_done()


# ── Microphone + OWW ─────────────────────────────────────────────────────────

def load_oww() -> OWWModel:
    """Load OpenWakeWord model (blocking — run in executor)."""
    print("Chargement OpenWakeWord...", flush=True)
    return OWWModel(wakeword_models=[OWW_MODEL], inference_framework="onnx")


def start_microphone(
    audio_q:   asyncio.Queue,
    loop:      asyncio.AbstractEventLoop,
    oww:       OWWModel,
    state_ref: list,
    tts_abort,          # threading.Event from tts.py
) -> sd.RawInputStream:
    """Start microphone stream. OWW runs in the PortAudio callback thread
    (non-blocking for asyncio). Audio chunks are also forwarded to audio_q
    for Vosk FR transcription.
    """

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

        # OWW runs directly in this PortAudio thread
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
            oww.predict(audio_np)   # keep OWW warm during LISTENING

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
