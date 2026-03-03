"""
Jarvis voice agent + WebSocket UI server
  • Vosk STT  →  wake word  →  router  →  DuckDuckGo (optional)
  • Ollama streaming  →  Piper TTS
  • aiohttp WebSocket server (port 8080) → animated SVG face UI
"""

import asyncio
import json
import re
import threading
from enum import Enum, auto

import aiohttp
import aiohttp.web
import numpy as np
import sounddevice as sd
from ddgs import DDGS
from openwakeword.model import Model as OWWModel
from vosk import KaldiRecognizer, Model as VoskModel
from piper.voice import PiperVoice

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL  = "http://127.0.0.1:11434/api/chat"
MODEL       = "ministral-3:3b"
UI_PORT     = 8080

PIPER_MODEL    = "/Users/floriankerlogot/models/fr_FR-siwis-medium.onnx"
VOSK_MODEL_FR  = "/Users/floriankerlogot/models/vosk-model-small-fr-0.22"
OWW_THRESHOLD  = 0.5   # confidence score to trigger wake word

MIC_RATE    = 16000
MIC_DTYPE   = "int16"

WAKE_WORDS  = {"jarvis", "hey jarvis"}

SYSTEM_PROMPT = (
    "Tu es Jarvis, un assistant vocal concis qui parle français. "
    "Réponds en 1 à 3 courtes phrases, comme si tu parlais à voix haute. "
    "N'utilise jamais de markdown, de puces, d'astérisques, de dièses ou de blocs de code. "
    "Sois direct, naturel et sympathique."
)

ROUTER_PROMPT = (
    "Tu es un routeur. Décide si la question de l'utilisateur nécessite une recherche web "
    "pour des informations actuelles, ou peut être répondue depuis tes connaissances. "
    "Réponds UNIQUEMENT avec du JSON valide, sans explication. "
    'Format : {"needs_search": true, "query": "requête de recherche concise"} '
    'ou {"needs_search": false}'
)

SENTENCE_RE  = re.compile(r'(?<=[.!?])\s+|(?<=\n)')
MARKDOWN_RE  = re.compile(r'[*_`#~\[\]]+')
tts_abort    = threading.Event()

# ── WebSocket broadcast ───────────────────────────────────────────────────────
_ws_clients: set = set()
_main_loop: asyncio.AbstractEventLoop | None = None


async def broadcast(data: dict) -> None:
    global _ws_clients
    if not _ws_clients:
        return
    msg  = json.dumps(data, ensure_ascii=False)
    dead = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_str(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)   # in-place, no rebinding


def broadcast_sync(data: dict) -> None:
    """Thread-safe broadcast — callable from TTS thread."""
    if _main_loop and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(data), _main_loop)

# ── State ─────────────────────────────────────────────────────────────────────

class State(Enum):
    SLEEPING  = auto()
    LISTENING = auto()
    THINKING  = auto()


# IPA phoneme → mouth shape viseme
# Covers French IPA; extend as needed
_PHONEME_VISEME: dict[str, str] = {
    # Voiced open vowels
    "a": "A", "ɑ": "A", "æ": "A", "ɔ": "A",
    # Front/mid vowels
    "e": "E", "ɛ": "E", "i": "E", "y": "E", "j": "E",
    # Rounded/back vowels
    "o": "O", "u": "O", "ø": "O", "œ": "O", "w": "O", "ʉ": "O",
    # Neutral / schwa
    "ə": "U", "œ̃": "U",
    # Nasal vowels
    "ã": "A", "ɛ̃": "E", "ɔ̃": "O",
    # Bilabial stops/nasal → lips closed
    "p": "closed", "b": "closed", "m": "closed",
    # Labiodental
    "f": "U", "v": "U",
    # Postalveolar / palato-alveolar → rounded
    "ʃ": "O", "ʒ": "O",
    # Silence / padding tokens
    "_": "rest", " ": "rest", ".": "rest",
}


def phoneme_to_viseme(ph: str) -> str:
    return _PHONEME_VISEME.get(ph, "rest")


def _schedule_visemes(
    visemes: list[str],
    durations_s: list[float],
    abort_event: threading.Event,
) -> None:
    """Send timed viseme events from a background thread."""
    import time
    for vis, dur in zip(visemes, durations_s):
        if abort_event.is_set():
            break
        broadcast_sync({"type": "viseme", "viseme": vis})
        time.sleep(dur)

# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_for_tts(text: str) -> str:
    text = MARKDOWN_RE.sub("", text)
    return re.sub(r'\s+', ' ', text).strip()

# ── Piper TTS ─────────────────────────────────────────────────────────────────
_voice: PiperVoice | None = None


def get_voice() -> PiperVoice:
    global _voice
    if _voice is None:
        print("Chargement du modèle Piper TTS...", flush=True)
        _voice = PiperVoice.load(PIPER_MODEL)
    return _voice


def _speak_blocking(sentence: str) -> None:
    voice = get_voice()
    broadcast_sync({"type": "state", "state": "speaking"})

    # ── Collect ALL chunks before playback to avoid inter-chunk gaps ──
    audio_parts:  list = []
    all_alignments: list = []
    all_phonemes:   list = []
    sample_rate = None

    for chunk in voice.synthesize(sentence):
        if tts_abort.is_set():
            return
        if sample_rate is None:
            sample_rate = chunk.sample_rate
        audio_parts.append(chunk.audio_float_array)
        if chunk.phoneme_alignments:
            all_alignments.extend(chunk.phoneme_alignments)
        elif chunk.phonemes:
            all_phonemes.extend(chunk.phonemes)

    if tts_abort.is_set() or not audio_parts:
        return

    # Single concatenated array → single OutputStream → no gaps
    audio      = np.concatenate(audio_parts)
    duration_s = len(audio) / sample_rate

    # ── Viseme timeline ──
    if all_alignments:
        visemes   = [phoneme_to_viseme(pa.phoneme) for pa in all_alignments]
        durations = [pa.num_samples / sample_rate   for pa in all_alignments]
    elif all_phonemes:
        raw = [phoneme_to_viseme(p) for p in all_phonemes]
        visemes = [raw[0]] if raw else ["rest"]
        for v in raw[1:]:
            if v != visemes[-1]:
                visemes.append(v)
        n         = max(len(visemes), 1)
        durations = [duration_s / n] * n
    else:
        visemes   = ["rest"]
        durations = [duration_s]

    done = threading.Event()

    def _play():
        """Play audio in thread, checking tts_abort periodically."""
        chunk_size = sample_rate // 10   # 100ms slices for abort responsiveness
        with sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32",
            latency="low",
        ) as stream:
            for i in range(0, len(audio), chunk_size):
                if tts_abort.is_set():
                    break
                seg = audio[i : i + chunk_size].reshape(-1, 1)
                stream.write(seg)
        done.set()

    viseme_t = threading.Thread(
        target=_schedule_visemes,
        args=(visemes, durations, tts_abort),
        daemon=True,
    )
    play_t = threading.Thread(target=_play, daemon=True)
    play_t.start()
    viseme_t.start()
    done.wait()
    play_t.join(timeout=2.0)
    viseme_t.join(timeout=1.0)

    broadcast_sync({"type": "viseme", "viseme": "rest"})


async def speak(sentence: str) -> None:
    sentence = clean_for_tts(sentence)
    if not sentence:
        return
    await asyncio.get_event_loop().run_in_executor(None, _speak_blocking, sentence)


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

# ── Vosk STT ──────────────────────────────────────────────────────────────────

def contains_wake_word(text: str) -> bool:
    t = text.lower().strip()
    # Match "jarvis", "jar vis", "jarvice", "hey jarvis" etc.
    return "jarv" in t or any(w in t for w in WAKE_WORDS)


async def stt_worker(
    audio_q: asyncio.Queue,
    text_q:  asyncio.Queue,
    state_ref: list,
) -> None:
    """Only handles French Vosk transcription when in LISTENING state.
    OWW wake-word detection runs in the mic callback thread instead."""
    print("Chargement Vosk FR...", flush=True)
    rec_fr = KaldiRecognizer(VoskModel(VOSK_MODEL_FR), MIC_RATE)
    rec_fr.SetWords(False)
    print("Prêt. Dites 'Hey Jarvis' pour m'activer.\n", flush=True)
    await broadcast({"type": "state", "state": "sleeping"})

    _listening_start: float = 0.0
    import time as _time

    while True:
        data = await audio_q.get()
        if data is None:
            audio_q.task_done()
            break

        state = state_ref[0]

            # (state debug removed — confirmed working)

        if state == State.LISTENING:
            if _listening_start == 0.0:
                _listening_start = _time.monotonic()

            got = rec_fr.AcceptWaveform(data)
            p = json.loads(rec_fr.PartialResult()).get("partial", "")
            if p:
                print(f"[FR partial]: {p}", flush=True)
            if got:
                text = json.loads(rec_fr.Result()).get("text", "").strip()
                print(f"[FR final]: {text!r}", flush=True)
                if text:
                    _listening_start = 0.0
                    rec_fr.Reset()
                    state_ref[0] = State.THINKING
                    await broadcast({"type": "state", "state": "thinking"})
                    await text_q.put(text)

            # Timeout: force final result after 5s with no transcription
            elif _time.monotonic() - _listening_start > 5.0:
                text = json.loads(rec_fr.FinalResult()).get("text", "").strip()
                print(f"[FR timeout final]: {text!r}", flush=True)
                _listening_start = 0.0
                rec_fr.Reset()
                if text:
                    state_ref[0] = State.THINKING
                    await broadcast({"type": "state", "state": "thinking"})
                    await text_q.put(text)
                else:
                    state_ref[0] = State.SLEEPING
                    await broadcast({"type": "state", "state": "sleeping"})
                    print("💤 Timeout LISTEN → retour en veille.", flush=True)
        else:
            _listening_start = 0.0
            rec_fr.Reset()

        audio_q.task_done()


def start_microphone(
    audio_q: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    oww: "OWWModel",
    state_ref: list,
):
    """OWW runs directly in PortAudio callback thread (non-blocking for asyncio).
    Audio also goes to audio_q for Vosk FR transcription."""

    async def _on_wake(score: float):
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

        # ── OWW: runs in this PortAudio thread (same as test_vosk.py) ──
        if state_ref[0] in (State.SLEEPING, State.THINKING):
            scores = oww.predict(audio_np)
            score  = max(scores.get("hey_jarvis", 0),
                         scores.get("hey_jarvis_v0.1", 0))
            if score >= OWW_THRESHOLD:
                oww.reset()
                asyncio.run_coroutine_threadsafe(_on_wake(score), loop)
        else:
            oww.predict(audio_np)   # keep OWW warm during LISTENING

        # Always send audio to async queue for Vosk FR
        loop.call_soon_threadsafe(audio_q.put_nowait, data)

    stream = sd.RawInputStream(
        samplerate=MIC_RATE,
        blocksize=1280,          # 80ms — required by OpenWakeWord
        dtype=MIC_DTYPE, channels=1,
        callback=mic_callback,
    )
    stream.start()
    return stream

# ── Tool calling ──────────────────────────────────────────────────────────────

async def router_call(command: str, history: list[dict]) -> dict:
    messages = [
        {"role": "system", "content": ROUTER_PROMPT},
        *history[1:],
        {"role": "user", "content": command},
    ]
    payload = {"model": MODEL, "messages": messages, "stream": False, "format": "json"}
    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return json.loads(data["message"]["content"])


def _search_web_blocking(query: str, max_results: int = 4) -> list[dict]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("href",  ""),
                "snippet": r.get("body",  ""),
            })
    return results


async def search_web(query: str) -> list[dict]:
    return await asyncio.get_event_loop().run_in_executor(None, _search_web_blocking, query)

# ── Ollama streaming ──────────────────────────────────────────────────────────

async def stream_response(
    messages: list[dict],
    tts_q: asyncio.Queue,
    search_results: list[dict] | None = None,
) -> str:
    msgs = list(messages)
    if search_results:
        snippets = "\n".join(
            f"- {r['title']}: {r['snippet']}" for r in search_results
        )
        msgs = list(messages[:-1]) + [{
            "role": "user",
            "content": (
                f"{messages[-1]['content']}\n\n"
                f"[Résultats de recherche web]\n{snippets}\n"
                "Réponds en t'appuyant sur ces résultats. Sois concis."
            ),
        }]

    payload    = {"model": MODEL, "messages": msgs, "stream": True}
    full_reply = ""
    buffer     = ""

    async with aiohttp.ClientSession() as session:
        async with session.post(OLLAMA_URL, json=payload) as response:
            response.raise_for_status()
            print(f"\n[Jarvis] : ", end="", flush=True)

            async for line in response.content:
                line = line.strip()
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                print(token, end="", flush=True)
                full_reply += token
                buffer     += token

                # Stream token to UI
                await broadcast({"type": "token", "token": token})

                parts = SENTENCE_RE.split(buffer)
                if len(parts) > 1:
                    pending = " ".join(s.strip() for s in parts[:-1] if s.strip())
                    # Only queue for TTS when we have enough text (avoids micro-gaps)
                    if pending and len(pending) >= 80:
                        await tts_q.put(pending)
                        buffer = parts[-1]
                    elif len(parts) > 3:
                        # Many small sentences accumulated — flush anyway
                        await tts_q.put(pending)
                        buffer = parts[-1]

                if chunk.get("done"):
                    break

    if buffer.strip():
        await tts_q.put(buffer.strip())
    print()
    return full_reply


async def process_command(
    command: str,
    messages: list[dict],
    tts_q: asyncio.Queue,
) -> str:
    await broadcast({"type": "state", "state": "thinking"})
    print("\n[Routeur] : réflexion...", flush=True)
    try:
        decision = await router_call(command, messages)
    except Exception as e:
        print(f"[Erreur routeur : {e}]", flush=True)
        decision = {"needs_search": False}

    search_results = None
    if decision.get("needs_search"):
        query = decision.get("query", command)
        print(f"[🔍 Recherche : {query}]", flush=True)
        await broadcast({"type": "search", "query": query})
        try:
            search_results = await search_web(query)
            print(f"[✅ {len(search_results)} résultat(s)]", flush=True)
        except Exception as e:
            print(f"[Erreur recherche : {e}]", flush=True)

    return await stream_response(messages, tts_q, search_results=search_results)

# ── aiohttp WebSocket server ──────────────────────────────────────────────────

async def ws_handler(request: aiohttp.web.Request) -> aiohttp.web.WebSocketResponse:
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    _ws_clients.add(ws)
    try:
        async for _ in ws:
            pass   # client sends nothing; we only push
    finally:
        _ws_clients.discard(ws)
    return ws


async def index_handler(request: aiohttp.web.Request) -> aiohttp.web.FileResponse:
    return aiohttp.web.FileResponse(
        "/Users/floriankerlogot/PrincessdAgent/ui.html"
    )

# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    global _main_loop
    _main_loop = asyncio.get_event_loop()

    # Start aiohttp web server
    app = aiohttp.web.Application()
    app.router.add_get("/",   index_handler)
    app.router.add_get("/ws", ws_handler)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "localhost", UI_PORT)
    await site.start()
    print(f"🌐 UI disponible sur http://localhost:{UI_PORT}", flush=True)

    audio_q:   asyncio.Queue = asyncio.Queue()
    text_q:    asyncio.Queue = asyncio.Queue()
    tts_q:     asyncio.Queue = asyncio.Queue()
    state_ref: list          = [State.SLEEPING]

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    tts_task = asyncio.create_task(tts_worker(tts_q))
    stt_task = asyncio.create_task(stt_worker(audio_q, text_q, state_ref))

    print("Chargement OpenWakeWord (hey_jarvis)...", flush=True)
    oww = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: OWWModel(wakeword_models=["hey_jarvis"], inference_framework="onnx"),
    )
    print("OWW prêt.", flush=True)

    mic = start_microphone(audio_q, _main_loop, oww, state_ref)

    try:
        while True:
            command = await text_q.get()
            text_q.task_done()

            print(f"\n🎤 Vous : {command}")
            await broadcast({"type": "message", "role": "user", "content": command})

            messages.append({"role": "user", "content": command})
            tts_abort.clear()
            reply = await process_command(command, messages, tts_q)
            messages.append({"role": "assistant", "content": reply})

            await broadcast({"type": "message", "role": "assistant", "content": reply})

            await tts_q.join()

            if state_ref[0] == State.LISTENING:
                print("\n🟢 Jarvis écoute votre commande...", flush=True)
            else:
                state_ref[0] = State.SLEEPING
                await broadcast({"type": "state", "state": "sleeping"})
                print("\n💤 En veille... dites 'Hey Jarvis' pour me réveiller.\n", flush=True)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nArrêt...")

    finally:
        mic.stop()
        mic.close()
        await audio_q.put(None)
        await tts_q.put(None)
        await asyncio.gather(stt_task, tts_task, return_exceptions=True)
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
