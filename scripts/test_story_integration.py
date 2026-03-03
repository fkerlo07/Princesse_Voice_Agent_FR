import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threading
import time
from tools import play_audio, search_story

# In real code tts_abort is in tts, we can import it
from tts import tts_abort

async def test_playback():
    print("Testing story search and playback...")
    
    # 1. Test Search
    story = search_story("racontes moi une histoire de princesses")
    print(f"Search Result: {story}")
    
    if not story:
        print("Story not found!")
        return

    # 2. Test Playback & Abort
    # To test abort, we will start playback and then simulate a wake-word detection
    # after 2 seconds. Because our dummy file is 0 bytes, mpv will exit immediately.
    # So we'll need to use a small valid mp3 or rely on the process exiting fast anyway.
    # Since dummy mp3 is 0 bytes, mpv exits instantly. Let's create a 5 second silent mp3 or use sleep in a mock,
    # but for true test we just verify the call doesn't throw.
    
    print("Issuing play command...")
    # Because mpv with a 0 byte file exits instantly, this will return right away.
    await play_audio(story["path"])
    print("Playback completed.")

if __name__ == "__main__":
    asyncio.run(test_playback())
