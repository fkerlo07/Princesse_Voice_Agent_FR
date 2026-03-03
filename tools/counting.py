import asyncio
import string
from ui_server import broadcast
from stt import State

NUMBER_MAP = {
    1: ["un", "1", "hein", "hun"],
    2: ["deux", "2", "de"],
    3: ["trois", "3", "troi"],
    4: ["quatre", "4"],
    5: ["cinq", "5", "saint", "sain"],
    6: ["six", "6", "si"],
    7: ["sept", "7", "set", "cette"],
    8: ["huit", "8", "uit"],
    9: ["neuf", "9", "oeuf"],
    10: ["dix", "10", "di", "dit"]
}

async def learn_counting(tts_q: asyncio.Queue, text_q: asyncio.Queue, state_ref: list) -> str:
    """Outil interactif d'apprentissage du calcul (compter)."""
    try:
        numbers = list(range(1, 101))
        # Affichage de la grille des chiffres
        await broadcast({
            "type": "edu_grid", 
            "items": [str(n) for n in numbers], 
            "gridClass": "grid-count"
        })
        
        for i, num in enumerate(numbers):
            await broadcast({"type": "edu_reading_zoom", "index": i})
            await tts_q.put(f"{num}.")
            await tts_q.put("À toi !")
            await tts_q.join()
            
            errors = 0
            while True:
                # Force listening state for STT bypass
                state_ref[0] = State.LISTENING
                await broadcast({"type": "state", "state": "listening"})
                print(f"[Compter] Attente du chiffre {num}...", flush=True)
                
                user_input = await text_q.get()
                text_q.task_done()
                
                user_input_lower = user_input.lower().strip()
                user_input_clean = user_input_lower.translate(str.maketrans('', '', string.punctuation)).strip()
                words = user_input_clean.split()
                print(f"[Compter] Mots nettoyés: {words}", flush=True)
                
                # Stop words
                if any(w in words for w in ["arrête", "stop", "quitter", "fin"]):
                    await tts_q.put("D'accord, on arrête de compter pour l'instant.")
                    await broadcast({"type": "edu_hide"})
                    return "L'utilisateur a arrêté l'apprentissage des chiffres."
                
                expected = NUMBER_MAP.get(num, [str(num)])
                if any(e in words for e in expected) or any(user_input_clean == e for e in expected):
                    await tts_q.put("Bravo !")
                    await tts_q.join()
                    break # Success, next number
                else:
                    errors += 1
                    if errors >= 3:
                        await tts_q.put(f"Ne t'en fais pas, c'était le chiffre {num}. Passons à la suite !")
                        await tts_q.join()
                        break
                    else:
                        await tts_q.put(f"Presque ! C'est le chiffre {num}. Essaie encore !")
                        await tts_q.join()
                
        await broadcast({"type": "edu_reading_unzoom"})
        await tts_q.put("Félicitations ! Tu sais compter jusqu'à dix !")
        await broadcast({"type": "edu_hide"})
        return "Apprentissage des chiffres terminé."
    except asyncio.CancelledError:
        print("[Compter] Apprentissage annulé par l'utilisateur.", flush=True)
        await broadcast({"type": "edu_hide"})
        state_ref[0] = State.SLEEPING
        await broadcast({"type": "state", "state": "sleeping"})
        raise
