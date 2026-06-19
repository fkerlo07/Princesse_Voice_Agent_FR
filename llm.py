"""
llm.py — LangGraph routing + LangChain streaming, single model: Gemma 4 E2B.
Router and answer generation both use the same gemma4:e2b model via Ollama.
"""
import asyncio
import json
from datetime import datetime
from typing import Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from config import (
    GEMMA4_MODEL, MIN_TTS_CHUNK_LEN, OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE, ROUTER_PROMPT,
)
from tts import SENTENCE_RE
from tools import play_audio, search_story, search_web
from ui_server import broadcast


# ── Single LLM singleton ──────────────────────────────────────────────────────

_llm: ChatOllama | None = None


def _get_llm(json_mode: bool = False) -> ChatOllama:
    """Return a ChatOllama instance for gemma4:e2b."""
    global _llm
    if _llm is None or json_mode:
        return ChatOllama(
            model=GEMMA4_MODEL,
            base_url=OLLAMA_BASE_URL,
            keep_alive=OLLAMA_KEEP_ALIVE,
            format="json" if json_mode else None,
            temperature=0 if json_mode else None,
        )
    return _llm


def _init_llm() -> None:
    global _llm
    if _llm is None:
        _llm = ChatOllama(
            model=GEMMA4_MODEL,
            base_url=OLLAMA_BASE_URL,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )


# ── Routing graph state ───────────────────────────────────────────────────────

class RoutingState(TypedDict):
    command: str
    action: str
    query: str
    search_results: Optional[list[dict]]


# ── Keyword sets ──────────────────────────────────────────────────────────────

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
    "learn_geography": {"géographie", "geographie", "carte du monde", "géographique"},
    "learn_planets":   {"planètes", "planetes", "planete", "planète", "système solaire", "cosmos"},
}


# ── Graph nodes ───────────────────────────────────────────────────────────────

def heuristic_node(state: RoutingState) -> dict:
    low   = state["command"].lower()
    words = set(low.split())

    for action, kws in _EDU_KW.items():
        if words & kws:
            print(f"\n[⚡ Heuristique → {action}]", flush=True)
            return {"action": action, "query": ""}

    if any(kw in low for kw in _FRESHNESS_KW):
        today = datetime.now().strftime("%d %B %Y")
        query = f"{state['command']} {today}"
        print(f"\n[⚡ Heuristique → web_search : {query!r}]", flush=True)
        return {"action": "web_search", "query": query}

    if words & _STORY_KW:
        return {"action": "__uncertain__", "query": state["command"]}

    if len(low.split()) <= 3:
        return {"action": "none", "query": ""}

    return {"action": "none", "query": ""}


async def router_llm_node(state: RoutingState) -> dict:
    today = datetime.now().strftime("%A %d %B %Y")
    messages = [
        SystemMessage(content=f"Nous sommes le {today}. " + ROUTER_PROMPT),
        HumanMessage(content=state["command"]),
    ]
    print("\n[Routeur Gemma4] réflexion...", flush=True)
    try:
        response = await _get_llm(json_mode=True).ainvoke(messages)
        decision = json.loads(response.content)
        action = decision.get("action", "none")
        query  = decision.get("query", state["command"])
        if action == "web_search":
            query = f"{query} {datetime.now().strftime('%d %B %Y')}"
        print(f"[Routeur Gemma4] → {action} : {query!r}", flush=True)
    except Exception as e:
        print(f"[Erreur routeur : {e}] → réponse directe", flush=True)
        action, query = "none", state["command"]
    return {"action": action, "query": query}


async def web_search_node(state: RoutingState) -> dict:
    print(f"\n[🔍 Recherche Web : {state['query']!r}]", flush=True)
    await broadcast({"type": "search", "query": state["query"]})
    try:
        results = await search_web(state["query"])
        print(f"[✅ {len(results)} résultat(s)]", flush=True)
    except Exception as e:
        print(f"[Erreur recherche : {e}]", flush=True)
        results = None
    return {"search_results": results}


# ── Graph edges ───────────────────────────────────────────────────────────────

def _route_heuristic(state: RoutingState) -> str:
    action = state["action"]
    if action == "web_search":    return "web_search"
    if action == "__uncertain__": return "router_llm"
    return END


def _route_after_router(state: RoutingState) -> str:
    if state["action"] == "web_search": return "web_search"
    return END


# ── Compiled routing graph ────────────────────────────────────────────────────

def _build_routing_graph():
    g = StateGraph(RoutingState)
    g.add_node("heuristic",  heuristic_node)
    g.add_node("router_llm", router_llm_node)
    g.add_node("web_search", web_search_node)
    g.set_entry_point("heuristic")
    g.add_conditional_edges("heuristic", _route_heuristic, {
        "web_search": "web_search",
        "router_llm": "router_llm",
        END: END,
    })
    g.add_conditional_edges("router_llm", _route_after_router, {
        "web_search": "web_search",
        END: END,
    })
    g.add_edge("web_search", END)
    return g.compile()


_routing_graph = _build_routing_graph()


# ── Streaming answer ──────────────────────────────────────────────────────────

async def stream_response(
    messages: list[dict],
    tts_q: asyncio.Queue,
    search_results: list[dict] | None = None,
) -> str:
    lc_msgs = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            lc_msgs.append(SystemMessage(content=content))
        elif role == "user":
            lc_msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            lc_msgs.append(AIMessage(content=content))

    if search_results:
        snippets = "\n".join(f"- {r['title']}: {r['snippet']}" for r in search_results)
        today    = datetime.now().strftime("%d %B %Y")
        last     = lc_msgs[-1]
        lc_msgs[-1] = HumanMessage(content=(
            f"{last.content}\n\n"
            f"[Résultats web du {today}]\n{snippets}\n"
            "Réponds uniquement sur base de ces résultats. Sois concis."
        ))

    full_reply = ""
    buffer     = ""
    print("\n[Jarvis] : ", end="", flush=True)

    async for chunk in _get_llm().astream(lc_msgs):
        token = chunk.content
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

    init: RoutingState = {
        "command": command,
        "action":  "",
        "query":   command,
        "search_results": None,
    }
    result         = await _routing_graph.ainvoke(init)
    action         = result["action"]
    search_results = result.get("search_results")

    if action == "play_story":
        story = search_story(result["query"])
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

    return await stream_response(messages, tts_q, search_results=search_results)
