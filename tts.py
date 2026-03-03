"""
tts.py — Piper TTS: synthesis, viseme scheduling, playback
"""
import asyncio
import re
import threading
import time

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice

from config import PIPER_MODEL, MIN_TTS_CHUNK_LEN
from ui_server import broadcast, broadcast_sync

# Shared abort flag (set by main when wake word fires during speech)
tts_abort = threading.Event()

SENTENCE_RE = re.compile(r'(?<=[.!?])\s+|(?<=\n)')
MARKDOWN_RE = re.compile(r'[*_`#~\[\]]+')

# ── IPA phoneme → viseme ──────────────────────────────────────────────────────
_PHONEME_VISEME: dict[str, str] = {
    "a": "A", "ɑ": "A", "æ": "A", "ɔ": "A",
    "e": "E", "ɛ": "E", "i": "E", "y": "E", "j": "E",
    "o": "O", "u": "O", "ø": "O", "œ": "O", "w": "O", "ʉ": "O",
    "ə": "U", "œ̃": "U",
    "ã": "A", "ɛ̃": "E", "ɔ̃": "O",
    "p": "closed", "b": "closed", "m": "closed",
    "f": "U",  "v": "U",
    "ʃ": "O",  "ʒ": "O",
    "_": "rest", " ": "rest", ".": "rest",
}


def phoneme_to_viseme(ph: str) -> str:
    return _PHONEME_VISEME.get(ph, "rest")


def _schedule_visemes(
    visemes: list[str],
    durations_s: list[float],
    abort_event: threading.Event,
) -> None:
    for vis, dur in zip(visemes, durations_s):
        if abort_event.is_set():
            break
        broadcast_sync({"type": "viseme", "viseme": vis})
        time.sleep(dur)


# ── Text helpers ──────────────────────────────────────────────────────────────

def clean_for_tts(text: str) -> str:
    text = MARKDOWN_RE.sub("", text)
    return re.sub(r'\s+', ' ', text).strip()


# ── Voice (lazy singleton) ────────────────────────────────────────────────────
_voice: PiperVoice | None = None


def get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        print("Chargement du modèle Piper TTS...", flush=True)
        _voice = PiperVoice.load(PIPER_MODEL)
    return _voice


# ── Core playback ─────────────────────────────────────────────────────────────

def _speak_blocking(sentence: str) -> None:
    """Stream Piper chunks into a persistent OutputStream.
    For each chunk, a viseme thread starts at the same moment audio starts
    writing — so mouth movements lock to the audio in real-time.
    """
    voice = get_voice()
    broadcast_sync({"type": "state", "state": "speaking"})

    chunks_iter = iter(voice.synthesize(sentence))
    first_chunk = next(chunks_iter, None)
    if first_chunk is None or tts_abort.is_set():
        return

    sample_rate = first_chunk.sample_rate
    stream = sd.OutputStream(
        samplerate=sample_rate, channels=1, dtype="float32", latency="low"
    )
    stream.start()

    def _play_chunk_with_visemes(
        audio: np.ndarray,
        alignments: list,
        phonemes: list,
    ) -> None:
        """Write one Piper chunk to the stream.
        Starts a viseme thread simultaneously — tries phoneme_alignments first
        (sample-accurate), falls back to phonemes with even distribution.
        """
        duration_s = len(audio) / sample_rate
        vis_t = None

        if alignments:
            visemes   = [phoneme_to_viseme(pa.phoneme) for pa in alignments]
            durations = [pa.num_samples / sample_rate   for pa in alignments]
        elif phonemes:
            raw = [phoneme_to_viseme(p) for p in phonemes]
            visemes = [raw[0]] if raw else ["rest"]
            for v in raw[1:]:
                if v != visemes[-1]:
                    visemes.append(v)
            n         = max(len(visemes), 1)
            durations = [duration_s / n] * n
        else:
            visemes = durations = None

        if visemes:
            vis_t = threading.Thread(
                target=_schedule_visemes,
                args=(visemes, durations, tts_abort),
                daemon=True,
            )
            vis_t.start()   # starts BEFORE audio write — locks timing

        seg = sample_rate // 10   # 100ms slices for abort granularity
        for i in range(0, len(audio), seg):
            if tts_abort.is_set():
                break
            stream.write(audio[i: i + seg].reshape(-1, 1))

        if vis_t:
            vis_t.join(timeout=5.0)

    try:
        _play_chunk_with_visemes(
            first_chunk.audio_float_array,
            first_chunk.phoneme_alignments or [],
            first_chunk.phonemes           or [],
        )
        for chunk in chunks_iter:
            if tts_abort.is_set():
                break
            _play_chunk_with_visemes(
                chunk.audio_float_array,
                chunk.phoneme_alignments or [],
                chunk.phonemes           or [],
            )

    finally:
        stream.stop()
        stream.close()
        broadcast_sync({"type": "viseme", "viseme": "rest"})


async def speak(sentence: str) -> None:
    sentence = clean_for_tts(sentence)
    if not sentence:
        return
    await asyncio.get_event_loop().run_in_executor(None, _speak_blocking, sentence)


# ── TTS worker ────────────────────────────────────────────────────────────────

async def tts_worker(tts_q: asyncio.Queue) -> None:
    while True:
        sentence = await tts_q.get()
        if sentence is None:
            tts_q.task_done()
            break
        if tts_abort.is_set():
            tts_q.task_done()
            continue
        try:
            await speak(sentence)
        except Exception as e:
            print(f"\n[TTS error: {e}]", flush=True)
        finally:
            tts_q.task_done()
