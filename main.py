"""
main.py — Jarvis entry point
Run: python main.py
"""
import asyncio
from datetime import datetime

from config import SYSTEM_PROMPT
from llm import process_command
from stt import State, load_oww, stt_worker, start_microphone
from tts import tts_abort, tts_worker
from tools.alphabet import learn_alphabet
from tools.reading import learn_reading
from tools.counting import learn_counting
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

    async def event_worker():
        nonlocal active_edu_task
        while True:
            event = await event_q.get()
            event_type = event.get('type')
            
            if event_type == 'start_edu':
                tool = event.get('tool')
                print(f"\n[UI Event] Lancement de l'outil éducatif: {tool}", flush=True)
                
                # Stop any currently running tool
                if active_edu_task and not active_edu_task.done():
                    active_edu_task.cancel()
                    tts_abort.set()
                
                # Clear queues
                while not text_q.empty():
                    text_q.get_nowait()
                    text_q.task_done()
                while not edu_text_q.empty():
                    edu_text_q.get_nowait()
                    edu_text_q.task_done()
                tts_abort.clear()
                    
                if tool == 'alphabet':
                    active_edu_task = asyncio.create_task(learn_alphabet(tts_q, edu_text_q, state_ref))
                elif tool == 'reading':
                    active_edu_task = asyncio.create_task(learn_reading(tts_q, edu_text_q, state_ref))
                elif tool == 'counting':
                    active_edu_task = asyncio.create_task(learn_counting(tts_q, edu_text_q, state_ref))
            
            elif event_type == 'stop_edu':
                print("\n[UI Event] Arrêt de l'outil éducatif", flush=True)
                if active_edu_task and not active_edu_task.done():
                    active_edu_task.cancel()
                tts_abort.set()
                # Need to clear it so subsequent normal TTS voices can work
                await asyncio.sleep(0.1) # Let the abort signals process
                tts_abort.clear()
                
                await broadcast({"type": "edu_hide"})
                state_ref[0] = State.SLEEPING
                await broadcast({"type": "state", "state": "sleeping"})
            
            event_q.task_done()

    event_task = asyncio.create_task(event_worker())

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
