"""
llm.py — Routing + streaming answer via Gemma 4 E2B (transformers / MPS).
Keyword heuristic handles routing; gemma4_local.stream_generate() produces answers.
"""
import asyncio
from datetime import datetime
from typing import Optional

import gemma4_local
from config import MIN_TTS_CHUNK_LEN, SYSTEM_PROMPT
from tts import SENTENCE_RE
from tools import play_audio, search_story, search_web
from ui_server import broadcast


# ── Keyword routing sets ───────────────────────────────────────────────────────

_FRESHNESS_KW = {
    "aujourd'hui", "maintenant", "hier", "ce soir", "cette semaine",
    "ce matin", "ce midi", "dernier", "dernière", "récent", "récente",
    "actuellement", "en ce moment", "en direct", "live",
    "résultat", "résultats", "score", "classement", "match", "gp",
    "moto", "formule", "f1", "nba", "ligue", "coupe", "championnat",
    "nouvelles", "news", "info", "actualité", "météo", "température",
    "bourse", "cours", "prix", "action",
}

_STORY_KW = {"raconte", "histoire", "conte", "joue", "lis"}

_EDU_KW: dict[str, set[str]] = {
    "learn_alphabet":  {"alphabet", "lettres", "lettre"},
    "learn_reading":   {"lire", "lecture", "épeler", "lisons"},
    "learn_counting":  {"compter", "chiffres", "chiffre", "compte"},
    "learn_geography": {"géographie", "geographie", "carte du monde"},
    "learn_planets":   {"planètes", "planetes", "planète", "système solaire"},
}


# ── Routing ────────────────────────────────────────────────────────────────────

def _route(command: str) -> tuple[str, str]:
    """
    Fast keyword routing. Returns (action, query).
    Falls back to 'none' (direct LLM answer) for everything else.
    """
    low   = command.lower()
    words = set(low.split())

    for action, kws in _EDU_KW.items():
        if words & kws:
            print(f"\n[⚡ → {action}]", flush=True)
            return action, ""

    if words & _STORY_KW:
        return "play_story", command

    if any(kw in low for kw in _FRESHNESS_KW):
        today = datetime.now().strftime("%d %B %Y")
        query = f"{command} {today}"
        print(f"\n[⚡ → web_search : {query!r}]", flush=True)
        return "web_search", query

    return "none", command


# ── Streaming answer ───────────────────────────────────────────────────────────

async def stream_response(
    messages:       list[dict],
    tts_q:          asyncio.Queue,
    search_results: list[dict] | None = None,
) -> str:
    """Stream a response from Gemma 4 E2B, flushing chunks to TTS at sentence boundaries."""

    # Inject search results into last user message
    if search_results:
        snippets = "\n".join(f"- {r['title']}: {r['snippet']}" for r in search_results)
        today    = datetime.now().strftime("%d %B %Y")
        msgs     = list(messages)
        msgs[-1] = {**msgs[-1], "content": (
            f"{msgs[-1]['content']}\n\n"
            f"[Résultats web du {today}]\n{snippets}\n"
            "Réponds uniquement sur base de ces résultats. Sois concis."
        )}
    else:
        msgs = messages

    # Convert to the format gemma4_local expects (system/user/assistant roles)
    hf_msgs = []
    for m in msgs:
        hf_msgs.append({"role": m["role"], "content": m["content"]})

    full_reply = ""
    buffer     = ""
    loop       = asyncio.get_event_loop()
    print("\n[Jarvis] : ", end="", flush=True)

    # Collect tokens via thread-safe queue bridged to asyncio
    token_q: asyncio.Queue = asyncio.Queue()

    def _on_token(tok: str) -> None:
        loop.call_soon_threadsafe(token_q.put_nowait, tok)

    # Run generation in executor (non-blocking)
    async def _generate():
        return await loop.run_in_executor(
            None, gemma4_local.stream_generate, hf_msgs, _on_token
        )

    gen_task = asyncio.create_task(_generate())

    # Consume tokens as they arrive
    while True:
        try:
            token = token_q.get_nowait()
        except asyncio.QueueEmpty:
            if gen_task.done():
                break
            await asyncio.sleep(0.01)
            continue

        print(token, end="", flush=True)
        full_reply += token
        buffer     += token
        await broadcast({"type": "token", "token": token})

        parts = SENTENCE_RE.split(buffer)
        if len(parts) > 1:
            pending = " ".join(s.strip() for s in parts[:-1] if s.strip())
            if pending and (MIN_TTS_CHUNK_LEN == 0 or len(pending) >= MIN_TTS_CHUNK_LEN):
                await tts_q.put(pending)
                buffer = parts[-1]

    # Drain any remaining tokens after gen_task finishes
    while not token_q.empty():
        token = token_q.get_nowait()
        print(token, end="", flush=True)
        full_reply += token
        buffer     += token
        await broadcast({"type": "token", "token": token})

    if buffer.strip():
        await tts_q.put(buffer.strip())
    print()
    return full_reply


# ── Main orchestrator ─────────────────────────────────────────────────────────

async def process_command(
    command:   str,
    messages:  list[dict],
    tts_q:     asyncio.Queue,
    text_q:    asyncio.Queue,
    state_ref: list,
) -> str:
    await broadcast({"type": "state", "state": "thinking"})

    action, query = _route(command)

    if action == "play_story":
        story = search_story(query)
        if story and story["path"]:
            reply = f"Je vous raconte {story.get('title', 'l\'histoire')} tout de suite."
            await tts_q.put(reply)
            await play_audio(story["path"])
            return reply
        reply = "Désolé, je n'ai pas trouvé d'histoire."
        await tts_q.put(reply)
        return reply

    if action == "learn_alphabet":
        from tools.alphabet import learn_alphabet
        return await learn_alphabet(tts_q, text_q, state_ref)

    if action == "learn_reading":
        from tools.reading import learn_reading
        return await learn_reading(tts_q, text_q, state_ref)

    if action == "learn_counting":
        from tools.counting import learn_counting
        return await learn_counting(tts_q, text_q, state_ref)

    if action == "learn_geography":
        from tools.geography import learn_geography
        return await learn_geography(tts_q, text_q, state_ref)

    if action == "learn_planets":
        from tools.planets import learn_planets
        return await learn_planets(tts_q, text_q, state_ref)

    search_results = None
    if action == "web_search":
        await broadcast({"type": "search", "query": query})
        try:
            search_results = await search_web(query)
            print(f"[✅ {len(search_results)} résultat(s)]", flush=True)
        except Exception as e:
            print(f"[Erreur recherche : {e}]", flush=True)

    return await stream_response(messages, tts_q, search_results=search_results)
