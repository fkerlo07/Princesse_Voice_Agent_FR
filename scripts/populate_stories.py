import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CHROMA_DB_DIR, STORIES_DIR
from tools import add_story

# Create a dummy MP3 file for testing
dummy_mp3_path = os.path.join(STORIES_DIR, "dummy_princesse.mp3")

if not os.path.exists(dummy_mp3_path):
    print(f"Creating dummy file at {dummy_mp3_path}...")
    # Just an empty file or you can put a real mp3 there manually later
    with open(dummy_mp3_path, "wb") as f:
        f.write(b"")

print("Ajout de l'histoire de princesse dans ChromaDB...")
add_story(
    story_id="histoire_princesse_01",
    title="La princesse et le dragon",
    text_summary="Une histoire de princesse captivante. Princesse dragon chateau aventure histoire.",
    file_path=dummy_mp3_path
)

print("Terminé !")
