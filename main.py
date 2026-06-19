"""
main.py — Jarvis entry point
Run: python main.py
"""
import asyncio
from datetime import datetime

import gemma4_local
from config import SYSTEM_PROMPT
from llm import process_command
from stt import State, load_oww, stt_worker, start_microphone
from tts import tts_abort, tts_worker
from tools.alphabet import learn_alphabet
from tools.counting import learn_counting
from tools.geography import learn_geography
from tools.planets import learn_planets
from tools.reading import learn_reading
from ui_server import broadcast, build_app, set_main_loop


async def main() -> None:
    loop = asyncio.get_event_loop()
    set_main_loop(loop)

    # Queues and shared state
    audio_q:   asyncio.Queue = asyncio.Queue()
    text_q:    asyncio.Queue = asyncio.Queue()
    tts_q:     asyncio.Queue = asyncio.Queue()
    event_q:   asyncio.Queue = asyncio.Queue()
    edu_text_q: asyncio.Queue = asyncio.Queue()
    state_ref: list          = [State.SLEEPING]

    # Start web UI server
    runner, _site = await build_app(event_q)

    # Conversation history — system prompt includes today's date
    today = datetime.now().strftime("%A %d %B %Y")
    system = f"Nous sommes le {today}. " + SYSTEM_PROMPT
    messages: list[dict] = [{"role": "system", "content": system}]

    # Start background workers
    tts_task = asyncio.create_task(tts_worker(tts_q))
    stt_task = asyncio.create_task(stt_worker(audio_q, text_q, state_ref))

    # Event handler for UI buttons
    active_edu_task: asyncio.Task | None = None

    async def _drain_queue(q: asyncio.Queue) -> None:
        """Empty a queue and call task_done for each item."""
        while not q.empty():
            try:
                q.get_nowait()
                q.task_done()
            except Exception:
                break

    async def _cancel_edu_task() -> None:
        """Cancel the active edu task and wait for it to finish."""
        nonlocal active_edu_task
        if active_edu_task and not active_edu_task.done():
            active_edu_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(active_edu_task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        active_edu_task = None

    async def event_worker():
        nonlocal active_edu_task
        while True:
            event = await event_q.get()
            event_type = event.get('type')

            if event_type == 'start_edu':
                tool = event.get('tool')
                print(f"\n[UI Event] Lancement de l'outil éducatif: {tool}", flush=True)

                # Stop any currently running tool first
                tts_abort.set()
                await _cancel_edu_task()
                await _drain_queue(tts_q)
                await _drain_queue(text_q)
                await _drain_queue(edu_text_q)
                await asyncio.sleep(0.1)
                tts_abort.clear()

                if tool == 'alphabet':
                    active_edu_task = asyncio.create_task(learn_alphabet(tts_q, edu_text_q, state_ref))
                elif tool == 'reading':
                    active_edu_task = asyncio.create_task(learn_reading(tts_q, edu_text_q, state_ref))
                elif tool == 'counting':
                    active_edu_task = asyncio.create_task(learn_counting(tts_q, edu_text_q, state_ref))
                elif tool == 'geography':
                    active_edu_task = asyncio.create_task(learn_geography(tts_q, edu_text_q, state_ref))
                elif tool == 'planets':
                    active_edu_task = asyncio.create_task(learn_planets(tts_q, edu_text_q, state_ref))

            elif event_type == 'stop_edu':
                print("\n[UI Event] Retour au menu principal", flush=True)
                # 1. Abort current TTS immediately
                tts_abort.set()
                # 2. Cancel edu task and wait for it to clean up
                await _cancel_edu_task()
                # 3. Drain all queues so nothing plays or gets processed
                await _drain_queue(tts_q)
                await _drain_queue(text_q)
                await _drain_queue(edu_text_q)
                # 4. Small pause for audio hardware to release, then clear abort
                await asyncio.sleep(0.25)
                tts_abort.clear()
                # 5. Restore UI to main screen
                await broadcast({"type": "edu_hide"})
                state_ref[0] = State.SLEEPING
                await broadcast({"type": "state", "state": "sleeping"})
                print("💤 Retour en veille.", flush=True)

            elif event_type == 'stop_tts':
                # Interrupt current speech only — edu task keeps running
                print("\n[UI Event] Stop TTS", flush=True)
                tts_abort.set()
                await _drain_queue(tts_q)
                await asyncio.sleep(0.15)
                tts_abort.clear()

            event_q.task_done()

    event_task = asyncio.create_task(event_worker())

    # Preload Gemma 4 E2B (transformers/MPS) in executor — blocks until model is in memory
    await asyncio.get_event_loop().run_in_executor(None, gemma4_local.load)

    # Load OWW in thread executor so it doesn't block the event loop
    oww = await asyncio.get_event_loop().run_in_executor(None, load_oww)
    print("OWW prêt.", flush=True)

    # Open microphone (OWW runs inside the PortAudio callback)
    mic = start_microphone(audio_q, loop, oww, state_ref, tts_abort)

    try:
        while True:
            command = await text_q.get()
            text_q.task_done()

            if active_edu_task and not active_edu_task.done():
                await edu_text_q.put(command)
                continue

            print(f"\n🎤 Vous : {command}")
            await broadcast({"type": "message", "role": "user", "content": command})

            messages.append({"role": "user", "content": command})
            tts_abort.clear()
            reply = await process_command(command, messages, tts_q, text_q, state_ref)
            messages.append({"role": "assistant", "content": reply})

            await broadcast({"type": "message", "role": "assistant", "content": reply})
            await tts_q.join()

            if state_ref[0] == State.LISTENING:
                print("\n🟢 Jarvis écoute votre commande...", flush=True)
            else:
                state_ref[0] = State.SLEEPING
                await broadcast({"type": "state", "state": "sleeping"})
                print("\n💤 En veille... dites 'Hey Jarvis' pour me réveiller.\n", flush=True)

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nArrêt...")

    finally:
        mic.stop()
        mic.close()
        await audio_q.put(None)
        await tts_q.put(None)
        event_task.cancel()
        if active_edu_task and not active_edu_task.done():
            active_edu_task.cancel()
        await asyncio.gather(stt_task, tts_task, return_exceptions=True)
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
