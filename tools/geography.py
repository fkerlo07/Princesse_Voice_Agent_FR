"""
tools/geography.py — Interactive geography quiz: identify highlighted countries on the world map.
"""
import asyncio
import random
import re
import string
from difflib import SequenceMatcher

from stt import State
from ui_server import broadcast


# ── Country data ── ISO 3166-1 numeric code : French TTS name + accepted answers ──
# Answers are all lowercase, without articles.
# Include phonetic/Whisper transcription variants so fuzzy match catches edge cases.

COUNTRIES: dict[str, dict] = {
    "250": {"name": "la France",          "answers": ["france", "frances"]},
    "276": {"name": "l'Allemagne",        "answers": ["allemagne", "alleman", "almagne"]},
    "724": {"name": "l'Espagne",          "answers": ["espagne", "espagnol"]},
    "380": {"name": "l'Italie",           "answers": ["italie", "italia", "italian"]},
    "826": {"name": "le Royaume-Uni",     "answers": ["royaume-uni", "royaume uni", "angleterre", "grande-bretagne", "grande bretagne", "bretagne"]},
    "643": {"name": "la Russie",          "answers": ["russie", "russia", "russe"]},
    "840": {"name": "les États-Unis",     "answers": ["états-unis", "etats-unis", "états unis", "etats unis", "amérique", "amerique", "usa", "america"]},
    "124": {"name": "le Canada",          "answers": ["canada"]},
    "76":  {"name": "le Brésil",          "answers": ["brésil", "bresil", "brasil", "brezel"]},
    "484": {"name": "le Mexique",         "answers": ["mexique", "mexico", "mexiko"]},
    "32":  {"name": "l'Argentine",        "answers": ["argentine", "argentina"]},
    "818": {"name": "l'Égypte",           "answers": ["égypte", "egypte", "egypte", "gypte", "egypt"]},
    "710": {"name": "l'Afrique du Sud",   "answers": ["afrique du sud", "afrique"]},
    "566": {"name": "le Nigéria",         "answers": ["nigéria", "nigeria", "nigerien", "nigérien"]},
    "504": {"name": "le Maroc",           "answers": ["maroc", "marroc", "marrocco"]},
    "404": {"name": "le Kenya",           "answers": ["kenya", "kenia"]},
    "156": {"name": "la Chine",           "answers": ["chine", "china", "chaina"]},
    "356": {"name": "l'Inde",             "answers": ["inde", "india", "indie", "indi", "hinde", "en de"]},
    "392": {"name": "le Japon",           "answers": ["japon", "japan", "jaapan"]},
    "36":  {"name": "l'Australie",        "answers": ["australie", "australia", "australi"]},
    "682": {"name": "l'Arabie Saoudite",  "answers": ["arabie saoudite", "arabie", "saudi", "saoudite"]},
    "792": {"name": "la Turquie",         "answers": ["turquie", "turkey", "turkie", "turkiye"]},
    "620": {"name": "le Portugal",        "answers": ["portugal"]},
    "752": {"name": "la Suède",           "answers": ["suède", "suede", "sweden"]},
    "578": {"name": "la Norvège",         "answers": ["norvège", "norvege", "norway"]},
}

ROUNDS_PER_GAME = 6
_STOP_WORDS = {"arrête", "stop", "quitter", "quitte", "fin", "retour", "menu"}

# Filler words Whisper sometimes inserts
_FILLER = re.compile(r'\b(euh|hum|ben|bah|ah|oh|hein|donc|alors|voilà|voila)\b')


# ── Answer matching ────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """Lowercase, remove punctuation/articles/fillers, normalise whitespace."""
    text = text.lower()
    # Remove punctuation (keep spaces)
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Normalise apostrophes
    text = text.replace("’", " ").replace("‘", " ").replace("'", " ")
    # Strip filler words
    text = _FILLER.sub("", text)
    # Strip leading articles / "c'est …" prefixes
    for prefix in (
        "c est la ", "c est le ", "c est l ", "c est les ", "c est ",
        "je crois que c est ", "je pense que c est ",
        "la ", "le ", "l ", "les ",
    ):
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix):]
            break
    return " ".join(text.split())   # collapse whitespace


def _fuzzy(a: str, b: str) -> float:
    """SequenceMatcher similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def _is_correct(user_input: str, answers: list[str]) -> bool:
    norm = _normalize(user_input)
    if not norm:
        return False

    words = [w for w in norm.split() if len(w) >= 2]

    for ans in answers:
        # 1. Exact: answer anywhere in the normalised input
        if ans == norm or ans in norm:
            return True

        # 2. Per-word fuzzy match (handles transcription noise)
        threshold = 0.76 if len(ans) <= 5 else (0.80 if len(ans) <= 8 else 0.83)
        for word in words:
            if _fuzzy(word, ans) >= threshold:
                return True

        # 3. Multi-word answers (e.g. "afrique du sud"): enough words fuzzy-match
        if " " in ans:
            ans_parts = ans.split()
            hits = sum(
                1 for ap in ans_parts
                if any(_fuzzy(w, ap) >= 0.80 for w in words)
            )
            if hits >= max(1, round(len(ans_parts) * 0.6)):
                return True

    return False


# ── Game loop ──────────────────────────────────────────────────────────────────

async def learn_geography(
    tts_q: asyncio.Queue,
    text_q: asyncio.Queue,
    state_ref: list,
) -> str:
    """Geography quiz — identify countries highlighted on the world map."""

    async def _think_and_say(msg: str) -> None:
        """Set THINKING state, speak, wait for audio to finish."""
        state_ref[0] = State.THINKING
        await broadcast({"type": "state", "state": "thinking"})
        await tts_q.put(msg)
        try:
            await asyncio.wait_for(tts_q.join(), timeout=20.0)
        except asyncio.TimeoutError:
            print(f"[Géo] TTS join timeout — skipping", flush=True)
        await asyncio.sleep(0.25)   # let room echo settle before recording

    try:
        await broadcast({"type": "geo_start"})
        await _think_and_say(
            "Je vais te montrer des pays sur la carte du monde. "
            "Tu dois dire le nom du pays surligné en jaune !"
        )

        pool  = random.sample(list(COUNTRIES.keys()), min(ROUNDS_PER_GAME, len(COUNTRIES)))
        score = 0

        for round_num, country_id in enumerate(pool, 1):
            country = COUNTRIES[country_id]
            name    = country["name"]
            answers = country["answers"]

            # ── Announce next country ──────────────────────────────────────────
            await broadcast({
                "type":  "geo_highlight",
                "id":    int(country_id),
                "state": "question",
                "round": round_num,
                "total": ROUNDS_PER_GAME,
            })
            await _think_and_say(f"Pays numéro {round_num}. Quel est ce pays ?")

            # ── Wait for answer loop ───────────────────────────────────────────
            # Timeout = LISTEN_TIMEOUT_S (5s) + Whisper inference (~4s) + margin
            _GET_TIMEOUT = 12.0
            errors = 0
            while True:
                # Only set LISTENING right before expecting voice input
                state_ref[0] = State.LISTENING
                await broadcast({"type": "state", "state": "listening"})
                print(f"[Géo] En attente — {name}", flush=True)

                try:
                    user_input = await asyncio.wait_for(text_q.get(), timeout=_GET_TIMEOUT)
                    text_q.task_done()
                except asyncio.TimeoutError:
                    # STT gave nothing (VAD rejected, silence, or device issue)
                    print(f"[Géo] Timeout réponse — aucun texte reçu", flush=True)
                    errors += 1
                    if errors >= 2:
                        await broadcast({
                            "type": "geo_highlight",
                            "id": int(country_id),
                            "state": "reveal",
                        })
                        await _think_and_say(f"Je n'ai pas entendu. C'était {name} !")
                        await asyncio.sleep(0.5)
                        break
                    await _think_and_say("Je n'ai pas entendu ta réponse. Essaie encore !")
                    continue   # loop → sets LISTENING again ✓

                # Immediately stop recording before any feedback TTS
                state_ref[0] = State.THINKING
                await broadcast({"type": "state", "state": "thinking"})

                print(f"[Géo] Reçu : {user_input!r}", flush=True)

                # Stop command
                if any(w in user_input.lower().split() for w in _STOP_WORDS):
                    await broadcast({"type": "geo_highlight", "id": 0, "state": "none"})
                    await _think_and_say("D'accord, on arrête le voyage pour l'instant !")
                    await broadcast({"type": "edu_hide"})
                    return "L'utilisateur a arrêté le quiz géographique."

                if _is_correct(user_input, answers):
                    score += 1
                    await broadcast({
                        "type":  "geo_highlight",
                        "id":    int(country_id),
                        "state": "correct",
                        "score": score,
                    })
                    await _think_and_say(f"Bravo ! C'est bien {name} !")
                    await asyncio.sleep(0.5)
                    break   # next country; state is THINKING ✓

                else:
                    errors += 1
                    if errors >= 2:
                        await broadcast({
                            "type":  "geo_highlight",
                            "id":    int(country_id),
                            "state": "reveal",
                        })
                        await _think_and_say(f"C'est {name} ! Allons voir le prochain pays.")
                        await asyncio.sleep(0.5)
                        break   # next country; state is THINKING ✓
                    await _think_and_say(
                        "Pas tout à fait… Regarde bien la carte et essaie encore !"
                    )
                    # loop → sets LISTENING again ✓

        # ── End of game ────────────────────────────────────────────────────────
        await broadcast({"type": "geo_highlight", "id": 0, "state": "none"})

        if score == len(pool):
            end_msg = f"Parfait ! {score} sur {len(pool)} ! Tu es un vrai explorateur du monde !"
        elif score >= len(pool) // 2:
            end_msg = f"Bien joué ! {score} pays sur {len(pool)} trouvés. Continue d'explorer !"
        else:
            end_msg = f"Tu as trouvé {score} pays sur {len(pool)}. Reviens t'entraîner !"

        await _think_and_say(end_msg)
        await broadcast({"type": "edu_hide"})
        return f"Quiz géographie terminé. Score : {score}/{len(pool)}."

    except asyncio.CancelledError:
        print("[Géo] Quiz annulé.", flush=True)
        await broadcast({"type": "geo_highlight", "id": 0, "state": "none"})
        await broadcast({"type": "edu_hide"})
        state_ref[0] = State.SLEEPING
        await broadcast({"type": "state", "state": "sleeping"})
        raise
