# Jarvis - Assistant Vocal Local et Éducatif

Jarvis est un assistant vocal intelligent interactif fonctionnant **entièrement en local**. Il allie reconnaissance vocale, détection de mot de réveil, synthèse vocale rapide et intelligence artificielle générative pour vous écouter, réfléchir et vous répondre sans dépendre du cloud.

Le projet inclut également des outils éducatifs interactifs (Alphabet, Lecture, Calcul) avec des interfaces web dédiées pour les enfants.

## ✨ Fonctionnalités Principales

- **100% Local & Privé :** Tous les modèles (LLM, reconnaissance vocale, synthèse vocale) tournent sur votre machine.
- **Réveil par la voix :** Écoute passive et déclenchement via le mot clé ("Hey Jarvis") grâce à OpenWakeWord.
- **Reconnaissance Vocale (STT) :** Transcription rapide et hors-ligne du français grâce à Vosk.
- **Synthèse Vocale (TTS) :** Voix française naturelle et réactive avec Piper TTS, synchronisée avec les animations de visage (visèmes).
- **Intelligence Artificielle (LLM) :** Propulsé par Ollama. Utilise un modèle de routage léger (`gemma3:1b`) pour la réflexion rapide et un modèle principal (`ministral-3:3b`) pour converser.
- **Recherche Web :** Capable de rechercher des informations récentes (actualités, météo) grâce à DuckDuckGo Search.
- **Histoires Audio :** Recherche et lecture de contes et d'histoires audios via RSS.
- **Outils Éducatifs (Enfants) :** Interfaces visuelles et vocales pour apprendre, accessibles via l'interface web :
  - L'alphabet (26 lettres avec suivi de progression et zoom)
  - Le calcul (Compter jusqu'à 100)
  - La lecture de mots (épellation avec explication courte du sens générée par l'IA)
- **Interface Web (UI) :** Un visage SVG animé qui respire, écoute et parle, avec console de discussion en direct et boutons pour les outils.

## ⚙️ Prérequis

- **Python** 3.10 à 3.12 (Python 3.13 n'est pas supporté par onnxruntime/openwakeword)
- **Ollama** installé sur votre machine (`https://ollama.com`)
- **PortAudio** pour l'accès au microphone :
  - Sur macOS : `brew install portaudio`
  - Sur Linux : `sudo apt install portaudio19-dev`

## 📥 Installation

**1. Cloner le projet :**
```bash
git clone <URL_DE_VOTRE_DEPOT>
cd <NOM_DU_DOSSIER>
```

**2. Créer un environnement virtuel et installer les dépendances :**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**3. Télécharger les modèles locaux :**
Le projet nécessite des modèles audio locaux. Vous devez ajuster les chemins dans `config.py` selon votre structure.

- **Vosk (STT Français)** :
  Téléchargez `vosk-model-small-fr-X.X` depuis [alphacephei.com/vosk](https://alphacephei.com/vosk/models) et extrayez-le.
- **Piper TTS (Synthèse vocale)** :
  Téléchargez un fichier `.onnx` français (ex: `fr_FR-siwis-medium.onnx`) et son fichier `.json` associé depuis le dépôt HuggingFace de Piper TTS.

Mettez à jour le fichier `config.py` avec vos chemins :
```python
PIPER_MODEL   = "/chemin/vers/fr_FR-siwis-medium.onnx"
VOSK_MODEL_FR = "/chemin/vers/vosk-model-small-fr-0.22"
```

**4. Récupérer les modèles Ollama :**
Assurez-vous qu'Ollama tourne en fond, puis lancez :
```bash
ollama run ministral-3:3b
ollama run gemma3:1b
```

## 🚀 Utilisation

Pour démarrer l'assistant vocal et le serveur Web :

```bash
source venv/bin/activate
python main.py
```

1. Ouvrez votre navigateur et allez sur `http://localhost:8080`. Vous verrez le visage de l'assistant (mode "En veille").
2. Dites **"Hey Jarvis"** à proximité de votre micro. Le visage va passer en vert (mode "J'écoute").
3. Posez votre question en français. L'assistant passera en jaune (réflexion) puis vous répondra à voix haute en synchronisant sa bouche.
4. **Outils éducatifs** : Cliquez sur les boutons "Alphabet", "Lecture" ou "Compter" depuis l'interface web pour lancer les mini-jeux interactifs pour enfants. Utilisez le bouton "Stop" rouge pour interrompre l'outil et repasser en mode assistant classique.

## 🛠️ Configuration (`config.py`)

Vous pouvez personnaliser le comportement dans `config.py` :
- `LLM_MODEL` / `ROUTER_MODEL` : Changez les modèles Ollama.
- `OWW_THRESHOLD = 0.5` : Sensibilité du mot de réveil (baissez si la détection est difficile).
- `LISTEN_TIMEOUT_S = 5.0` : Temps d'écoute maximal après avoir dit le mot de réveil.
- `SYSTEM_PROMPT` : Changez la personnalité de base de l'assistant.

## 📄 Architecture Technique

- **`main.py`** : Point d'entrée, gère les boucles asynchrones et l'orchestration.
- **`stt.py`** : Détecte "Hey Jarvis" avec OpenWakeWord et transcrit l'audio du microphone avec Vosk.
- **`tts.py`** : Synthétise le texte en audio avec Piper TTS et calcule les visèmes pour l'interface web.
- **`llm.py`** : Route la demande (heuristique de rapidité ou `gemma3:1b`), fait des recherches si besoin via DuckDuckGo, et génère la réponse en stream avec `ministral-3:3b`.
- **`ui_server.py` & `ui.html`** : Serveur aiohttp et interface front-end websocket (visage SVG pur CSS, console de logs, overlay éducatif).
- **`/tools/`** : Scripts des outils éducatifs intégrés (lecture, nombres, alphabet).

## 🤝 Contribution
Toute contribution (nouvelles idées d'apprentissage, modèles STT/TTS alternatifs, etc.) est la bienvenue ! N'hésitez pas à ouvrir une *Issue* ou à faire une *Pull Request*.
