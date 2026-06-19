"""
config.py — Jarvis configuration
Edit this file to change models, paths, thresholds, and prompts.
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_URL      = "http://127.0.0.1:11434/api/chat"

# Single model handles STT (audio understanding) + LLM response
GEMMA4_MODEL   = "gemma4:e2b"
OLLAMA_KEEP_ALIVE = -1

# ── Paths ─────────────────────────────────────────────────────────────────────
PIPER_MODEL    = "/Users/floriankerlogot/models/fr_FR-siwis-medium.onnx"
UI_HTML        = ROOT / "ui.html"
CHROMA_DB_DIR  = ROOT / "chroma_db"
STORIES_DIR    = ROOT / "stories"

# ── Microphone ────────────────────────────────────────────────────────────────
MIC_RATE      = 16_000
MIC_DTYPE     = "int16"
MIC_BLOCKSIZE = 1280           # 80 ms — required by OpenWakeWord

# ── Wake word ─────────────────────────────────────────────────────────────────
OWW_MODEL     = "hey_jarvis"
OWW_THRESHOLD = 0.5

# ── UI server ─────────────────────────────────────────────────────────────────
UI_PORT = 8080

# ── Tool calling ──────────────────────────────────────────────────────────────
SEARCH_MAX_RESULTS = 4
MIN_TTS_CHUNK_LEN  = 0

# ── Listening timeout ─────────────────────────────────────────────────────────
LISTEN_TIMEOUT_S = 6.0

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Tu es Jarvis, un assistant vocal concis qui parle français. "
    "Réponds en 1 à 3 courtes phrases, comme si tu parlais à voix haute. "
    "N'utilise jamais de markdown, de puces, d'astérisques, de dièses ou de blocs de code. "
    "Sois direct, naturel et sympathique."
)

ROUTER_PROMPT = (
    "Tu es un routeur. Analyse la demande de l'utilisateur et choisis l'outil approprié.\n"
    "Les actions possibles sont :\n"
    "- 'web_search': si la question nécessite des informations de recherche web.\n"
    "- 'play_story': si l'utilisateur demande à écouter une histoire ou un conte.\n"
    "- 'learn_alphabet': si l'utilisateur veut apprendre l'alphabet.\n"
    "- 'learn_reading': si l'utilisateur veut apprendre à lire.\n"
    "- 'learn_counting': si l'utilisateur veut apprendre à compter.\n"
    "- 'learn_geography': si l'utilisateur veut apprendre la géographie ou les pays du monde.\n"
    "- 'learn_planets': si l'utilisateur veut apprendre les planètes du système solaire.\n"
    "- 'none': si aucune de ces actions n'est requise.\n\n"
    "Réponds UNIQUEMENT avec du JSON valide, sans explication.\n"
    "Format: {\"action\": \"web_search\"|\"play_story\"|\"learn_alphabet\"|\"learn_reading\"|\"learn_counting\"|\"learn_geography\"|\"learn_planets\"|\"none\", \"query\": \"requête ou vide\"}"
)
