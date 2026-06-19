"""
llm.py — LangGraph routing graph + LangChain Ollama streaming
"""
import asyncio
import json
from datetime import datetime
from typing import TypedDict, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from config import (
    LLM_MODEL, MIN_TTS_CHUNK_LEN, OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE, ROUTER_MODEL, ROUTER_PROMPT,
)
from tts import SENTENCE_RE
from tools import play_audio, search_story, search_web
from ui_server import broadcast


# ── LLM singletons ────────────────────────────────────────────────────────────

_router_llm: ChatOllama | None = None
_answer_llm: ChatOllama | None = None


def _get_router_llm() -> ChatOllama:
    global _router_llm
    if _router_llm is None:
        _router_llm = ChatOllama(
            model=ROUTER_MODEL,
            base_url=OLLAMA_BASE_URL,
            format="json",
            temperature=0,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
    return _router_llm


def _get_answer_llm() -> ChatOllama:
    global _answer_llm
    if _answer_llm is None:
        _answer_llm = ChatOllama(
            model=LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
    return _answer_llm


# ── Routing graph state ────────────────────────────────────────────────────────

class RoutingState(TypedDict):
    command: str
    action: str                     # routing decision
    query: str                      # refined query for search / story
    search_results: Optional[list[dict]]


# ── Keyword sets ───────────────────────────────────────────────────────────────

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
    "learn_alphabet":   {"alphabet", "lettres", "lettre"},
    "learn_reading":    {"lire", "lecture", "épeler", "lisons"},
    "learn_counting":   {"compter", "chiffres", "chiffre", "compte"},
    "learn_geography":  {"géographie", "geographie", "carte du monde", "géographique"},
    "learn_planets":    {"planètes", "planetes", "planete", "planète", "système solaire", "cosmos"},
}


# ── Graph nodes ────────────────────────────────────────────────────────────────

def heuristic_node(state: RoutingState) -> dict:
    """Zero-latency keyword routing — avoids the router LLM for obvious cases."""
    low = state["command"].lower()
    words = set(low.split())

    # Educational tools (most specific — check first)
    for action, kws in _EDU_KW.items():
        if words & kws:
            print(f"\n[⚡ Heuristique → {action}]", flush=True)
            return {"action": action, "query": ""}

    # Fresh-data query → web search
    if any(kw in low for kw in _FRESHNESS_KW):
        today = datetime.now().strftime("%d %B %Y")
        query = f"{state['command']} {today}"
        print(f"\n[⚡ Heuristique → web_search : {query!r}]", flush=True)
        return {"action": "web_search", "query": query}

    # Story keywords → let LLM router handle (needs query refinement)
    if words & _STORY_KW:
        return {"action": "__uncertain__", "query": state["command"]}

    # Short query with no signal → direct answer
    if len(low.split()) <= 3:
        print("\n[⚡ Heuristique → réponse directe (courte)]", flush=True)
        return {"action": "none", "query": ""}

    # Long conversational query → direct answer
    print("\n[⚡ Heuristique → réponse directe (conversationnelle)]", flush=True)
    return {"action": "none", "query": ""}


async def router_llm_node(state: RoutingState) -> dict:
    """LLM router: gemma3:1b classifies intent and refines the query."""
    today = datetime.now().strftime("%A %d %B %Y")
    messages = [
        SystemMessage(content=f"Nous sommes le {today}. " + ROUTER_PROMPT),
        HumanMessage(content=state["command"]),
    ]
    print("\n[Routeur LLM] réflexion...", flush=True)
    try:
        response = await _get_router_llm().ainvoke(messages)
        decision = json.loads(response.content)
        action = decision.get("action", "none")
        query = decision.get("query", state["command"])
        if action == "web_search":
            today_str = datetime.now().strftime("%d %B %Y")
            query = f"{query} {today_str}"
        print(f"[Routeur LLM] → {action} : {query!r}", flush=True)
    except Exception as e:
        print(f"[Erreur routeur : {e}] → réponse directe", flush=True)
        action, query = "none", state["command"]
    return {"action": action, "query": query}


async def web_search_node(state: RoutingState) -> dict:
    """Execute DuckDuckGo search and store results in state."""
    print(f"\n[🔍 Recherche Web : {state['query']!r}]", flush=True)
    await broadcast({"type": "search", "query": state["query"]})
    try:
        results = await search_web(state["query"])
        print(f"[✅ {len(results)} résultat(s)]", flush=True)
    except Exception as e:
        print(f"[Erreur recherche : {e}]", flush=True)
        results = None
    return {"search_results": results}


# ── Conditional edge functions ─────────────────────────────────────────────────

def _route_heuristic(state: RoutingState) -> str:
    action = state["action"]
    if action == "web_search":    return "web_search"
    if action == "__uncertain__": return "router_llm"
    return END   # "none" or any edu/story action → terminate routing here


def _route_after_llm(state: RoutingState) -> str:
    if state["action"] == "web_search": return "web_search"
    return END


# ── Compiled routing graph ─────────────────────────────────────────────────────

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
    g.add_conditional_edges("router_llm", _route_after_llm, {
        "web_search": "web_search",
        END: END,
    })
    g.add_edge("web_search", END)
    return g.compile()


_routing_graph = _build_routing_graph()


# ── LangChain streaming answer ─────────────────────────────────────────────────

async def stream_response(
    messages: list[dict],
    tts_q: asyncio.Queue,
    search_results: list[dict] | None = None,
) -> str:
    """Stream tokens from LLM_MODEL via LangChain, flushing TTS at sentence boundaries."""
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
        today = datetime.now().strftime("%d %B %Y")
        last = lc_msgs[-1]
        lc_msgs[-1] = HumanMessage(content=(
            f"{last.content}\n\n"
            f"[Nous sommes le {today}. Résultats de recherche web récents]\n{snippets}\n"
            "Réponds uniquement sur base de ces résultats récents. Sois concis et précis."
        ))

    full_reply = ""
    buffer = ""
    print("\n[Jarvis] : ", end="", flush=True)

    async for chunk in _get_answer_llm().astream(lc_msgs):
        token = chunk.content
        print(token, end="", flush=True)
        full_reply += token
        buffer += token

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


# ── Main orchestrator ──────────────────────────────────────────────────────────

async def process_command(
    command: str,
    messages: list[dict],
    tts_q: asyncio.Queue,
    text_q: asyncio.Queue,
    state_ref: list,
) -> str:
    await broadcast({"type": "state", "state": "thinking"})

    # Run routing graph
    init: RoutingState = {
        "command": command,
        "action": "",
        "query": command,
        "search_results": None,
    }
    result = await _routing_graph.ainvoke(init)
    action = result["action"]
    search_results = result.get("search_results")

    if action == "play_story":
        story = search_story(result["query"])
        if story and story["path"]:
            title = story.get("title", "l'histoire")
            reply = f"Je vous raconte {title} tout de suite."
            await tts_q.put(reply)
            await play_audio(story["path"])
            return reply
        reply = "Désolé, je n'ai pas trouvé d'histoire correspondant à votre demande."
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

    # "none" or "web_search" (results already fetched by graph node)
    return await stream_response(messages, tts_q, search_results=search_results)
