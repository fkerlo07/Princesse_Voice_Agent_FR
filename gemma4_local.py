"""
gemma4_local.py — Gemma 4 E2B via HuggingFace Transformers on Apple Silicon MPS.
Single model instance shared by both STT (audio transcription) and LLM (text generation).
"""
import threading
from typing import Callable

import numpy as np
import torch
from transformers import (
    AutoProcessor,
    Gemma4ForConditionalGeneration,
    TextIteratorStreamer,
)

MODEL_ID = "google/gemma-4-E2B-it-qat-mobile-transformers"  # 2.5 GB QAT — fits on M1

_processor: AutoProcessor | None = None
_model:     Gemma4ForConditionalGeneration | None = None
_load_lock  = threading.Lock()


# ── Model loading ─────────────────────────────────────────────────────────────

def load() -> None:
    """Load model weights onto MPS. Idempotent — safe to call multiple times."""
    global _processor, _model
    with _load_lock:
        if _model is not None:
            return
        print(f"Chargement Gemma 4 E2B (transformers / MPS) — {MODEL_ID} …", flush=True)
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype="auto",            # QAT model handles its own quantization
            device_map="mps",
            attn_implementation="eager",   # MPS stability
        ).eval()
        print("Gemma 4 E2B prêt sur MPS.", flush=True)


def is_ready() -> bool:
    return _model is not None


def _get():
    if _model is None:
        load()
    return _processor, _model


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_mps(batch: dict) -> dict:
    """Move tensors to MPS, casting floats to bfloat16."""
    out = {}
    for k, v in batch.items():
        if not isinstance(v, torch.Tensor):
            out[k] = v
        elif v.dtype.is_floating_point:
            out[k] = v.to("mps", dtype=torch.bfloat16)
        else:
            out[k] = v.to("mps")
    return out


# ── Audio transcription (STT) ─────────────────────────────────────────────────

def transcribe(pcm_bytes: bytes, sample_rate: int = 16_000) -> str:
    """
    Transcribe raw int16 PCM bytes to French text using Gemma 4's audio encoder.
    Blocking — run in a thread executor from asyncio.
    """
    processor, model = _get()

    audio_f32 = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_f32},
            {"type": "text",  "text": (
                "Transcris exactement ce que dit la personne en français. "
                "Réponds UNIQUEMENT avec la transcription brute, sans commentaire."
            )},
        ],
    }]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _to_mps(processor(
        text=text,
        audios=[audio_f32],
        sampling_rate=sample_rate,
        return_tensors="pt",
    ))

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


# ── Text generation with streaming (LLM) ─────────────────────────────────────

def stream_generate(
    messages: list[dict],
    on_token: Callable[[str], None],
    max_new_tokens: int = 512,
) -> str:
    """
    Stream text generation. on_token(token) is called for each generated token.
    Blocking — run in a thread executor from asyncio.
    """
    processor, model = _get()

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = _to_mps(processor(text=text, return_tensors="pt"))

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    # Generation blocks — run in a daemon thread, iterate streamer in this thread
    gen_thread = threading.Thread(target=model.generate, kwargs=gen_kwargs, daemon=True)
    gen_thread.start()

    full = ""
    for token in streamer:
        on_token(token)
        full += token

    gen_thread.join()
    return full
