import asyncio
import random
from ui_server import broadcast
from stt import State

WORDS = [
    "CHAT", "CHIEN", "PAPA", "MAMAN", "SOLEIL", "LUNE", "OISEAU", "FLEUR", "EAU", "FEU", "MAISON", "ARBRE"
]

async def learn_reading(tts_q: asyncio.Queue, text_q: asyncio.Queue, state_ref: list) -> str:
    """Outil d'apprentissage de la lecture (épeler le mot)."""
    try:
        word = random.choice(WORDS)
        await broadcast({"type": "edu_reading", "word": word})
        
        await tts_q.put(f"Le mot est {word}.")
        await tts_q.join()
    
        for i, letter in enumerate(word):
            await broadcast({"type": "edu_reading_zoom", "index": i})
            await tts_q.put(f"[{letter}]")
            await tts_q.join()
            await asyncio.sleep(0.3)
            
        await broadcast({"type": "edu_reading_unzoom"})
        await tts_q.put(f"Et tout ensemble, ça fait : {word}.")
        await tts_q.join()
        
        # LLM explication courte
        await broadcast({"type": "state", "state": "thinking"})
        prompt = f"Explique très brièvement ce qu'est un(e) {word} en une seule petite phrase simple pour un enfant de 4 ans."
        messages = [{"role": "user", "content": prompt}]
        # Import local pour éviter l'import circulaire avec llm.py
        from llm import stream_response
        # stream_response va automatiser le TTS avec tts_q et envoyer les tokens à l'UI
        await stream_response(messages, tts_q)
        await tts_q.join()
    
        await broadcast({"type": "edu_hide"})
        return f"Le mot {word} a été épelé et lu à l'utilisateur."
    except asyncio.CancelledError:
        print("[Lecture] Apprentissage annulé par l'utilisateur.", flush=True)
        await broadcast({"type": "edu_hide"})
        state_ref[0] = State.SLEEPING
        await broadcast({"type": "state", "state": "sleeping"})
        raise
