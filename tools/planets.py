"""
tools/planets.py — Planet guessing quiz: 3 clues per planet, description on find or after all misses.
"""
import asyncio
import random
import re
import string
from difflib import SequenceMatcher

from stt import State
from ui_server import broadcast


PLANETS: dict[str, dict] = {
    "mercure": {
        "name": "Mercure",
        "answers": ["mercure", "mercury"],
        "clues": [
            "Je suis la planète la plus proche du Soleil.",
            "Je suis la plus petite planète du système solaire.",
            "Une année sur moi ne dure que 88 jours terrestres !",
        ],
        "description": (
            "Mercure est la plus petite planète et la plus proche du Soleil ! "
            "Sa surface est couverte de cratères, comme la Lune. "
            "Les températures varient entre moins 180 et plus 430 degrés selon le côté exposé."
        ),
    },
    "venus": {
        "name": "Vénus",
        "answers": ["vénus", "venus"],
        "clues": [
            "Je suis la planète la plus chaude du système solaire, encore plus que Mercure.",
            "Je suis la deuxième planète en partant du Soleil et je brille très fort dans le ciel.",
            "Mon atmosphère est faite de nuages épais de dioxyde de carbone.",
        ],
        "description": (
            "Vénus est la planète la plus chaude avec 465 degrés en permanence ! "
            "C'est aussi l'astre le plus brillant dans le ciel après le Soleil et la Lune. "
            "On l'appelle parfois l'étoile du berger."
        ),
    },
    "terre": {
        "name": "Terre",
        "answers": ["terre", "la terre", "earth"],
        "clues": [
            "Je suis la seule planète où la vie est connue dans l'univers.",
            "Je suis recouverte aux trois quarts d'eau liquide.",
            "Ma lune s'appelle tout simplement... la Lune !",
        ],
        "description": (
            "La Terre est notre planète bleue ! "
            "C'est la seule planète connue où il y a de la vie. "
            "Elle est recouverte d'eau aux trois quarts et a quatre milliards et demi d'années."
        ),
    },
    "mars": {
        "name": "Mars",
        "answers": ["mars"],
        "clues": [
            "Je suis rouge à cause de l'oxyde de fer dans mon sol, comme de la rouille.",
            "J'ai deux petites lunes : Phobos et Deimos.",
            "J'abrite le plus grand volcan du système solaire, Olympus Mons.",
        ],
        "description": (
            "Mars est la planète rouge ! "
            "Son volcan Olympus Mons est trois fois plus haut que l'Everest. "
            "Des robots explorateurs y roulent pour la découvrir de plus près."
        ),
    },
    "jupiter": {
        "name": "Jupiter",
        "answers": ["jupiter"],
        "clues": [
            "Je suis la plus grande planète du système solaire, une géante gazeuse.",
            "J'ai une célèbre grande tache rouge, qui est en fait une tempête gigantesque.",
            "J'ai plus de 90 lunes ! La plus grande s'appelle Ganymède.",
        ],
        "description": (
            "Jupiter est la plus grande planète ! "
            "1 300 Terres pourraient tenir à l'intérieur. "
            "Sa Grande Tache Rouge est une tempête qui dure depuis plus de 400 ans !"
        ),
    },
    "saturne": {
        "name": "Saturne",
        "answers": ["saturne", "saturn"],
        "clues": [
            "Je suis entourée de magnifiques anneaux visibles depuis la Terre.",
            "Je suis si légère que je flotterais sur l'eau si l'océan était assez grand.",
            "Ma lune Titan est plus grande que la planète Mercure.",
        ],
        "description": (
            "Saturne est la planète aux anneaux ! "
            "Ses anneaux sont faits de milliards de morceaux de glace et de roches. "
            "C'est la deuxième plus grande planète du système solaire."
        ),
    },
    "uranus": {
        "name": "Uranus",
        "answers": ["uranus"],
        "clues": [
            "Je suis bleue-verte à cause du méthane dans mon atmosphère.",
            "Je tourne sur le côté ! Mon axe est incliné à 98 degrés.",
            "Je suis la planète la plus froide avec moins 224 degrés.",
        ],
        "description": (
            "Uranus est la planète bleue-verte qui tourne couchée sur le côté ! "
            "C'est la planète la plus froide du système solaire. "
            "Elle possède aussi des anneaux, moins visibles que ceux de Saturne."
        ),
    },
    "neptune": {
        "name": "Neptune",
        "answers": ["neptune"],
        "clues": [
            "Je suis la planète la plus éloignée du Soleil.",
            "J'ai les vents les plus violents du système solaire, à plus de 2 000 km/h.",
            "Un an sur moi correspond à 165 années terrestres !",
        ],
        "description": (
            "Neptune est la planète la plus éloignée du Soleil. "
            "Ses vents peuvent souffler à 2 000 kilomètres par heure ! "
            "Une année sur Neptune correspond à 165 années terrestres."
        ),
    },
}

ROUNDS_PER_GAME = 4
_STOP_WORDS = {"arrête", "stop", "quitter", "quitte", "fin", "retour", "menu"}
_FILLER     = re.compile(r'\b(euh|hum|ben|bah|ah|oh|hein|donc|alors|voilà|voila)\b')
_GET_TIMEOUT = 12.0


# ── Answer matching (same approach as geography) ───────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.replace("’", " ").replace("‘", " ").replace("'", " ")
    text = _FILLER.sub("", text)
    for prefix in ("c est la ", "c est le ", "c est l ", "c est ",
                   "la ", "le ", "l "):
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix):]
            break
    return " ".join(text.split())


def _is_correct(user_input: str, answers: list[str]) -> bool:
    norm = _normalize(user_input)
    if not norm:
        return False
    words = [w for w in norm.split() if len(w) >= 2]
    for ans in answers:
        if ans == norm or ans in norm:
            return True
        thr = 0.76 if len(ans) <= 6 else 0.82
        for word in words:
            if SequenceMatcher(None, word, ans).ratio() >= thr:
                return True
    return False


# ── Game loop ──────────────────────────────────────────────────────────────────

async def learn_planets(
    tts_q:     asyncio.Queue,
    text_q:    asyncio.Queue,
    state_ref: list,
) -> str:
    """Planet quiz — identify planets from clues, then hear a description."""

    async def _say(msg: str) -> None:
        """Speak msg; STT is THINKING so the mic won't capture playback."""
        state_ref[0] = State.THINKING
        await broadcast({"type": "state", "state": "thinking"})
        await tts_q.put(msg)
        try:
            await asyncio.wait_for(tts_q.join(), timeout=20.0)
        except asyncio.TimeoutError:
            print("[Planètes] TTS join timeout", flush=True)
        await asyncio.sleep(0.25)

    async def _get_answer() -> str:
        """Set LISTENING, wait for STT, return text (or "" on timeout)."""
        state_ref[0] = State.LISTENING
        await broadcast({"type": "state", "state": "listening"})
        try:
            ans = await asyncio.wait_for(text_q.get(), timeout=_GET_TIMEOUT)
            text_q.task_done()
        except asyncio.TimeoutError:
            ans = ""
        # Stop recording immediately before any feedback TTS
        state_ref[0] = State.THINKING
        await broadcast({"type": "state", "state": "thinking"})
        return ans

    try:
        await broadcast({"type": "planet_start"})
        await _say(
            "Je vais te donner des indices sur une planète du système solaire. "
            "Écoute bien et essaie de deviner son nom !"
        )

        pool  = random.sample(list(PLANETS.keys()), min(ROUNDS_PER_GAME, len(PLANETS)))
        score = 0

        for round_num, planet_key in enumerate(pool, 1):
            planet  = PLANETS[planet_key]
            name    = planet["name"]
            answers = planet["answers"]
            clues   = planet["clues"]
            desc    = planet["description"]

            found = False

            for clue_idx, clue in enumerate(clues, 1):
                # Show planet (without name) + current clue
                await broadcast({
                    "type":      "planet_show",
                    "planet":    planet_key,
                    "clue":      f"Indice {clue_idx} : {clue}",
                    "show_name": False,
                    "round":     round_num,
                    "total":     ROUNDS_PER_GAME,
                })
                await _say(f"Indice {clue_idx}. {clue}")
                print(f"[Planètes] Round {round_num}, indice {clue_idx} — {name}", flush=True)

                user_input = await _get_answer()
                print(f"[Planètes] Reçu : {user_input!r}", flush=True)

                # Stop command
                if user_input and any(w in user_input.lower().split() for w in _STOP_WORDS):
                    await _say("D'accord, on arrête l'exploration des planètes pour l'instant !")
                    await broadcast({"type": "edu_hide"})
                    return "L'utilisateur a arrêté le quiz planètes."

                if user_input and _is_correct(user_input, answers):
                    found  = True
                    score += 1
                    await broadcast({
                        "type":      "planet_show",
                        "planet":    planet_key,
                        "clue":      desc,
                        "show_name": True,
                        "round":     round_num,
                        "total":     ROUNDS_PER_GAME,
                        "score":     score,
                    })
                    await _say(f"Bravo ! C'est bien {name} ! {desc}")
                    await asyncio.sleep(1.0)
                    break

                # Wrong answer or timeout
                if clue_idx < len(clues):
                    hint = "Pas encore ! Voici un autre indice." if user_input else "Voici un autre indice."
                    await _say(hint)

            if not found:
                # All clues used — reveal + describe
                await broadcast({
                    "type":      "planet_show",
                    "planet":    planet_key,
                    "clue":      desc,
                    "show_name": True,
                    "round":     round_num,
                    "total":     ROUNDS_PER_GAME,
                })
                await _say(f"C'était {name} ! {desc}")
                await asyncio.sleep(1.0)

        # End of game
        if score == len(pool):
            end = f"Parfait ! {score} planètes sur {len(pool)} ! Tu es un vrai astronome !"
        elif score >= len(pool) // 2:
            end = f"Bien joué ! {score} planètes sur {len(pool)} ! L'espace n'a presque plus de secrets pour toi !"
        else:
            end = f"Tu as trouvé {score} planètes sur {len(pool)}. Continue d'explorer le cosmos !"

        await _say(end)
        await broadcast({"type": "edu_hide"})
        return f"Quiz planètes terminé. Score : {score}/{len(pool)}."

    except asyncio.CancelledError:
        print("[Planètes] Quiz annulé.", flush=True)
        await broadcast({"type": "edu_hide"})
        state_ref[0] = State.SLEEPING
        await broadcast({"type": "state", "state": "sleeping"})
        raise
