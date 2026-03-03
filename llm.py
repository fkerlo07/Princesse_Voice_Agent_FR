"""
llm.py — LLM calls: non-streaming router (gemma3:1b) + streaming response
"""
import asyncio
import json
from datetime import datetime

import aiohttp

from config import (
    OLLAMA_URL, LLM_MODEL, ROUTER_MODEL, OLLAMA_KEEP_ALIVE,
    SYSTEM_PROMPT, ROUTER_PROMPT, MIN_TTS_CHUNK_LEN,
)
from tts import SENTENCE_RE
from tools import search_web, search_story, play_audio
from tools.alphabet import learn_alphabet
from tools.reading import learn_reading
from tools.counting import learn_counting
from ui_server import broadcast

# ── Shared HTTP session (reused across all requests) ──────────────────────────
_session: aiohttp.ClientSession | None = None


async def get_session() -> aiohttp.ClientSession:
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


# Words/phrases that signal the query needs fresh data from the web
_FRESHNESS_KEYWORDS = {
    # Time markers
    "aujourd'hui", "maintenant", "hier", "ce soir", "cette semaine",
    "ce matin", "ce midi", "dernier", "dernière", "récent", "récente",
    "actuellement", "en ce moment", "en direct", "live",
    # Sport / events
    "résultat", "résultats", "score", "classement", "match", "gp",
    "moto", "formule", "f1", "nba", "ligue", "coupe", "championnat",
    # News
    "nouvelles", "news", "info", "actualité", "météo", "température",
    "bourse", "cours", "prix", "action",
    # Who/what is X right now
    "qui est", "qu'est-ce qui", "quel est le", "quelle est la",
}


def _needs_search_heuristic(command: str) -> bool | None:
    """Return True  → search needed (skip router, search directly)
           Return False → search not needed (skip router, answer directly)
           Return None  → uncertain, call router LLM
    """
    low = command.lower()
    word_count = len(low.split())

    # Very short queries are almost never search-worthy
    if word_count <= 3 and not any(kw in low for kw in _FRESHNESS_KEYWORDS):
        return False

    # Contains a freshness keyword → definitely search
    if any(kw in low for kw in _FRESHNESS_KEYWORDS):
        return True

    # Long conversational queries with no freshness signal → skip router
    if word_count > 4 and not any(kw in low for kw in _FRESHNESS_KEYWORDS):
        # We check story keywords here purely to skip the fast-path return 
        # so it definitely goes to the router if they ask for a story
        story_kws = ["raconte", "histoire", "conte", "joue", "lis"]
        if any(kw in low for kw in story_kws):
            return None
        return False

    return None   # unclear — let the LLM router decide

async def router_call(command: str) -> dict:
    """Ask gemma3:1b whether a web search is needed.
    keep_alive=-1 keeps the model hot in Ollama for the next call.
    """
    today = datetime.now().strftime("%A %d %B %Y")   # e.g. "vendredi 28 évrier 2026"
    date_ctx = f"Nous sommes le {today}. "
    messages = [
        {"role": "system", "content": date_ctx + ROUTER_PROMPT},
        {"role": "user", "content": command},
    ]
    payload = {
        "model":      ROUTER_MODEL,
        "messages":   messages,
        "stream":     False,
        "format":     "json",
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    session = await get_session()
    async with session.post(OLLAMA_URL, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()
        return json.loads(data["message"]["content"])


# ── Streaming answer (main model) ─────────────────────────────────────────────

async def stream_response(
    messages:       list[dict],
    tts_q:          asyncio.Queue,
    search_results: list[dict] | None = None,
) -> str:
    """Stream tokens from LLM_MODEL. Queues TTS chunks immediately at sentence
    boundaries (no minimum length delay)."""
    msgs = list(messages)

    if search_results:
        snippets = "\n".join(
            f"- {r['title']}: {r['snippet']}" for r in search_results
        )
        today = datetime.now().strftime("%d %B %Y")
        msgs = list(messages[:-1]) + [{
            "role": "user",
            "content": (
                f"{messages[-1]['content']}\n\n"
                f"[Nous sommes le {today}. Résultats de recherche web récents]\n{snippets}\n"
                "Réponds uniquement sur base de ces résultats récents. Sois concis et précis."
            ),
        }]

    payload = {
        "model":      LLM_MODEL,
        "messages":   msgs,
        "stream":     True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
    }
    full_reply = ""
    buffer     = ""

    session = await get_session()
    async with session.post(OLLAMA_URL, json=payload) as response:
        response.raise_for_status()
        print("\n[Jarvis] : ", end="", flush=True)

        async for line in response.content:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            print(token, end="", flush=True)
            full_reply += token
            buffer     += token

            await broadcast({"type": "token", "token": token})

            # Flush to TTS at every sentence boundary
            parts = SENTENCE_RE.split(buffer)
            if len(parts) > 1:
                pending = " ".join(s.strip() for s in parts[:-1] if s.strip())
                if pending and (MIN_TTS_CHUNK_LEN == 0 or len(pending) >= MIN_TTS_CHUNK_LEN):
                    await tts_q.put(pending)
                    buffer = parts[-1]

            if chunk.get("done"):
                break

    if buffer.strip():
        await tts_q.put(buffer.strip())
    print()
    return full_reply


# ── Orchestrator ───────────────────────────────────────────────────────────────

async def process_command(
    command:  str,
    messages: list[dict],
    tts_q:    asyncio.Queue,
    text_q:   asyncio.Queue,
    state_ref: list,
) -> str:
    """Route → optional web search → streaming answer.
    Fast-path: keyword heuristic avoids the router LLM call for ~80% of queries.
    """
    await broadcast({"type": "state", "state": "thinking"})

    # ── 1. Try local heuristic first (0ms) ──
    heuristic = _needs_search_heuristic(command)

    if heuristic is True:
        today = datetime.now().strftime("%d %B %Y")
        query = f"{command} {today}"
        print(f"\n[⚡ Heuristique → recherche : {query}]", flush=True)
        await broadcast({"type": "search", "query": command})
        try:
            search_results = await search_web(query)
            print(f"[✅ {len(search_results)} résultat(s)]", flush=True)
        except Exception as e:
            print(f"[Erreur recherche : {e}]", flush=True)
            search_results = None
        return await stream_response(messages, tts_q, search_results=search_results)

    if heuristic is False:
        print(f"\n[⚡ Heuristique → pas de recherche]", flush=True)
        return await stream_response(messages, tts_q)

    # ── 2. Uncertain — call router LLM ──
    print("\n[Routeur LLM] : réflexion...", flush=True)
    try:
        decision = await router_call(command)
    except Exception as e:
        print(f"[Erreur routeur : {e}]", flush=True)
        decision = {"action": "none"}

    action = decision.get("action", "none")
    query = decision.get("query", command)
    
    if action == "play_story":
        print(f"[🔍 Recherche Histoire via LLM : {query}]", flush=True)
        story = search_story(query)
        if story and story["path"]:
            title = story.get("title", "l'histoire")
            reply_text = f"Je vous raconte {title} tout de suite."
            await tts_q.put(reply_text)
            await play_audio(story["path"])
            return reply_text
        else:
            reply_text = "Désolé, je n'ai pas trouvé d'histoire correspondant à votre demande."
            await tts_q.put(reply_text)
            return reply_text

    if action == "learn_alphabet":
        reply_text = await learn_alphabet(tts_q, text_q, state_ref)
        return reply_text

    if action == "learn_reading":
        reply_text = await learn_reading(tts_q, text_q, state_ref)
        return reply_text
        
    if action == "learn_counting":
        reply_text = await learn_counting(tts_q, text_q, state_ref)
        return reply_text

    search_results = None
    if action == "web_search":
        print(f"[🔍 Recherche Web via LLM: {query}]", flush=True)
        await broadcast({"type": "search", "query": query})
        try:
            search_results = await search_web(query)
            print(f"[✅ {len(search_results)} résultat(s)]", flush=True)
        except Exception as e:
            print(f"[Erreur recherche : {e}]", flush=True)

    return await stream_response(messages, tts_q, search_results=search_results)
