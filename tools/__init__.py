"""
tools.py — External tools Jarvis can use (currently: DuckDuckGo web search, ChromaDB story playback)
"""
import asyncio
import os
import subprocess
import time
from pathlib import Path

from ddgs import DDGS
import chromadb

from config import SEARCH_MAX_RESULTS, CHROMA_DB_DIR, STORIES_DIR
from tts import tts_abort
from ui_server import broadcast_sync


# ── Web Search Tool ────────────────────────────────────────────────────────────

def _search_blocking(query: str) -> list[dict]:
    """Synchronous DuckDuckGo search — runs in a thread executor."""
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=SEARCH_MAX_RESULTS):
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("href",  ""),
                "snippet": r.get("body",  ""),
            })
    return results


async def search_web(query: str) -> list[dict]:
    """Async DuckDuckGo search."""
    return await asyncio.get_event_loop().run_in_executor(None, _search_blocking, query)


# ── Story Tool (ChromaDB + MPV) ───────────────────────────────────────────────

os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(STORIES_DIR, exist_ok=True)

try:
    _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    _stories_collection = _chroma_client.get_or_create_collection(name="stories")
except Exception as e:
    print(f"[ChromaDB Init Error]: {e}")
    _chroma_client = None
    _stories_collection = None


def search_story(query: str, n_results: int = 1) -> dict | None:
    """Search for the most relevant story in ChromaDB based on the query."""
    if not _stories_collection:
        print("[Erreur] ChromaDB n'est pas initialisé.")
        return None

    try:
        results = _stories_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        if results and results['ids'] and results['ids'][0]:
            best_id = results['ids'][0][0]
            metadata = results['metadatas'][0][0]
            
            return {
                "id": best_id,
                "path": metadata.get("path", ""),
                "title": metadata.get("title", best_id)
            }
    except Exception as e:
        print(f"[ChromaDB Search Error]: {e}")
    
    return None


def add_story(story_id: str, text_summary: str, file_path: str, title: str = "") -> None:
    """Add or update a story in the vector database."""
    if not _stories_collection:
        print("[Erreur] ChromaDB n'est pas initialisé.")
        return

    try:
        _stories_collection.upsert(
            ids=[story_id],
            documents=[text_summary],
            metadatas=[{"path": file_path, "title": title or story_id}]
        )
        print(f"[Succès] Histoire '{story_id}' ajoutée/mise à jour dans ChromaDB.")
    except Exception as e:
        print(f"[ChromaDB Insert Error]: {e}")


def play_audio_blocking(filepath: str) -> None:
    """Play audio file using mpv, allowing interruption via tts_abort."""
    if not os.path.exists(filepath):
        print(f"[Erreur] Fichier audio introuvable : {filepath}", flush=True)
        return

    print(f"\n🎵 Lecture de l'histoire : {filepath}", flush=True)
    broadcast_sync({"type": "state", "state": "speaking"})

    process = subprocess.Popen(
        ["mpv", "--no-video", "--really-quiet", filepath],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    try:
        while process.poll() is None:
            if tts_abort.is_set():
                print(f"🛑 Arrêt de l'histoire demandé !", flush=True)
                process.terminate()
                time.sleep(0.1)
                if process.poll() is None:
                    process.kill()
                break
            time.sleep(0.1)
    except Exception as e:
        print(f"[Erreur lecture mpv]: {e}")
        if process.poll() is None:
            process.terminate()
    finally:
        broadcast_sync({"type": "state", "state": "sleeping"})
        print("Fin de l'histoire.", flush=True)

async def play_audio(filepath: str) -> None:
    """Async wrapper for story playback."""
    await asyncio.get_event_loop().run_in_executor(None, play_audio_blocking, filepath)
