"""
config.py — Jarvis configuration
Edit this file to change models, paths, thresholds, and prompts.
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent

# ── Ollama ────────────────────────────────────────────────────────────────────
OLLAMA_URL    = "http://127.0.0.1:11434/api/chat"
LLM_MODEL     = "ministral-3:3b"   # main model — answers the user
ROUTER_MODEL  = "gemma3:1b"        # small model — decides if web search needed

# ── Paths ─────────────────────────────────────────────────────────────────────
PIPER_MODEL   = "/Users/floriankerlogot/models/fr_FR-siwis-medium.onnx"
VOSK_MODEL_FR = "/Users/floriankerlogot/models/vosk-model-small-fr-0.22"
UI_HTML       = ROOT / "ui.html"
CHROMA_DB_DIR = ROOT / "chroma_db"
STORIES_DIR   = ROOT / "stories"

# ── Microphone ────────────────────────────────────────────────────────────────
MIC_RATE      = 16_000
MIC_DTYPE     = "int16"
MIC_BLOCKSIZE = 1280           # 80 ms — required by OpenWakeWord

# ── Wake word ─────────────────────────────────────────────────────────────────
OWW_MODEL     = "hey_jarvis"
OWW_THRESHOLD = 0.5            # detection confidence threshold

# ── UI server ─────────────────────────────────────────────────────────────────
UI_PORT = 8080

# ── Tool calling ──────────────────────────────────────────────────────────────
SEARCH_MAX_RESULTS = 4
# Speak as soon as ANY sentence boundary is hit (set >0 to buffer more text first)
MIN_TTS_CHUNK_LEN  = 0
# How long Ollama keeps models in memory (-1 = forever, keeps router+LLM both hot)
OLLAMA_KEEP_ALIVE  = -1

# ── Listening timeout ─────────────────────────────────────────────────────────
LISTEN_TIMEOUT_S = 5.0         # seconds before giving up and returning to sleep

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
    "- 'learn_alphabet': si l'utilisateur veut apprendre l'alphabet (ex: 'je veux apprendre l\\'alphabet').\n"
    "- 'learn_reading': si l'utilisateur veut apprendre à lire (ex: 'apprends moi à lire').\n"
    "- 'learn_counting': si l'utilisateur veut apprendre à compter (ex: 'apprends moi à compter').\n"
    "- 'none': si aucune de ces actions n'est requise.\n\n"
    "Réponds UNIQUEMENT avec du JSON valide, au format suivant, sans explication.\n"
    "Format attendu: {\"action\": \"web_search\" | \"play_story\" | \"learn_alphabet\" | \"learn_reading\" | \"learn_counting\" | \"none\", \"query\": \"requête si nécessaire, sinon vide\"}"
)
