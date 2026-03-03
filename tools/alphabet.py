import asyncio
import string
from ui_server import broadcast
from stt import State

LETTER_MAP = {
    "A": ["a", "ah", "ha"],
    "B": ["b", "bé", "baie", "be"],
    "C": ["c", "cé", "s'est", "sait", "ses", "ce"],
    "D": ["d", "dé", "des", "dès", "de"],
    "E": ["e", "eux", "heuh", "euh"],
    "F": ["f", "ef", "effe"],
    "G": ["g", "gé", "j'ai", "ge"],
    "H": ["h", "ache", "hache"],
    "I": ["i", "y", "il", "ils", "hi"],
    "J": ["j", "ji", "j'y"],
    "K": ["k", "cas", "qu'a", "ka"],
    "L": ["l", "elle", "ailes", "aile", "el"],
    "M": ["m", "aime", "ème", "em"],
    "N": ["n", "haine", "aine", "en"],
    "O": ["o", "au", "eau", "haut", "oh"],
    "P": ["p", "pé", "paix", "pe"],
    "Q": ["q", "cul", "cu"],
    "R": ["r", "air", "aire", "erre", "er"],
    "S": ["s", "esse", "est-ce", "es"],
    "T": ["t", "té", "thé", "te"],
    "U": ["u", "eu", "uh"],
    "V": ["v", "vé", "vais"],
    "W": ["w", "double v", "double vé"],
    "X": ["x", "ixe", "ix"],
    "Y": ["y", "i grec", "igrec"],
    "Z": ["z", "zède"]
}

async def learn_alphabet(tts_q: asyncio.Queue, text_q: asyncio.Queue, state_ref: list) -> str:
    """Outil interactif d'apprentissage de l'alphabet."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    try:
        # Affichage de la grille d'alphabet
        await broadcast({
            "type": "edu_grid", 
            "items": list(letters), 
            "gridClass": "grid-alpha"
        })
        
        for i, letter in enumerate(letters):
            await broadcast({"type": "edu_reading_zoom", "index": i})
            await tts_q.put(f"[{letter}]") # Let TTS engine say it
            await tts_q.put("À toi !")
            await tts_q.join()
            
            errors = 0
            while True:
                # Force listening state for STT bypass
                state_ref[0] = State.LISTENING
                await broadcast({"type": "state", "state": "listening"})
                print(f"[Alphabet] Attente de la lettre {letter}...", flush=True)
                
                user_input = await text_q.get()
                text_q.task_done()
                
                user_input_lower = user_input.lower().strip()
                user_input_clean = user_input_lower.translate(str.maketrans('', '', string.punctuation)).strip()
                words = user_input_clean.split()
                print(f"[Alphabet] Mots nettoyés: {words}", flush=True)
                
                # Stop words to abort early
                if any(w in words for w in ["arrête", "stop", "quitter", "fin"]):
                    await tts_q.put("D'accord, on arrête l'alphabet pour l'instant. Bravo pour tes efforts !")
                    await broadcast({"type": "edu_hide"})
                    return "L'utilisateur a arrêté l'apprentissage de l'alphabet."
                
                expected = LETTER_MAP.get(letter, [letter.lower()])
                if any(e in words for e in expected) or any(user_input_clean == e for e in expected):
                    await tts_q.put("Bravo !")
                    await tts_q.join()
                    break # Success, go to next letter
                else:
                    errors += 1
                    if errors >= 3:
                        await tts_q.put(f"Ne t'en fais pas, c'était la lettre [{letter}]. Passons à la suite !")
                        await tts_q.join()
                        break
                    else:
                        await tts_q.put(f"Presque ! C'est la lettre [{letter}]. Essaie encore !")
                        await tts_q.join()
                
        await broadcast({"type": "edu_reading_unzoom"})
        await tts_q.put("Félicitations ! Tu connais tout l'alphabet !")
        await broadcast({"type": "edu_hide"})
        return "Apprentissage de l'alphabet terminé."
    except asyncio.CancelledError:
        print("[Alphabet] Apprentissage annulé par l'utilisateur.", flush=True)
        await broadcast({"type": "edu_hide"})
        state_ref[0] = State.SLEEPING
        await broadcast({"type": "state", "state": "sleeping"})
        raise
