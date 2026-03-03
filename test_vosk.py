"""Test OpenWakeWord directly — say 'hey jarvis' to check if it triggers."""
import time
import numpy as np
import sounddevice as sd
from openwakeword.model import Model as OWWModel

print("Loading OWW model...")
oww = OWWModel(wakeword_models=["hey_jarvis"], inference_framework="onnx")
print("Model keys:", list(oww.models.keys()))
print("Say 'hey jarvis' now!\n")


def callback(indata: bytes, frames: int, t, status):
    audio = np.frombuffer(bytes(indata), dtype=np.int16)
    scores = oww.predict(audio)
    for key, score in scores.items():
        if score > 0.05:   # print anything above noise floor
            marker = " ← TRIGGERED!" if score >= 0.5 else ""
            print(f"  {key}: {score:.3f}{marker}", flush=True)


with sd.RawInputStream(
    samplerate=16000,
    blocksize=1280,    # 80ms — OWW recommended chunk size
    dtype="int16",
    channels=1,
    callback=callback,
):
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nDone.")
