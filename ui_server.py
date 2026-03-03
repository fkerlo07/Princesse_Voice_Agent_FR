"""
ui_server.py — aiohttp WebSocket server + broadcast helpers
"""
import asyncio
import json
from pathlib import Path

import aiohttp
import aiohttp.web

from config import UI_PORT, UI_HTML

# ── Globals ───────────────────────────────────────────────────────────────────
_ws_clients: set = set()
_main_loop: asyncio.AbstractEventLoop | None = None


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _main_loop
    _main_loop = loop


async def broadcast(data: dict) -> None:
    """Send JSON event to all connected WebSocket clients."""
    global _ws_clients
    if not _ws_clients:
        return
    msg  = json.dumps(data, ensure_ascii=False)
    dead = set()
    for ws in list(_ws_clients):
        try:
            await ws.send_str(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def broadcast_sync(data: dict) -> None:
    """Thread-safe broadcast — callable from TTS / audio threads."""
    if _main_loop and _main_loop.is_running():
        asyncio.run_coroutine_threadsafe(broadcast(data), _main_loop)


# ── Route handlers ────────────────────────────────────────────────────────────

async def ws_handler(request: aiohttp.web.Request) -> aiohttp.web.WebSocketResponse:
    ws = aiohttp.web.WebSocketResponse()
    await ws.prepare(request)
    _ws_clients.add(ws)
    event_q = request.app['event_q']
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    # Forward events like 'start_edu', 'stop_edu' to main loop
                    if 'type' in data:
                        await event_q.put(data)
                except json.JSONDecodeError:
                    pass
    finally:
        _ws_clients.discard(ws)
    return ws


async def index_handler(request: aiohttp.web.Request) -> aiohttp.web.FileResponse:
    return aiohttp.web.FileResponse(UI_HTML)


async def build_app(event_q: asyncio.Queue) -> tuple[aiohttp.web.AppRunner, aiohttp.web.TCPSite]:
    """Create, configure and start the aiohttp app. Returns (runner, site)."""
    app = aiohttp.web.Application()
    app['event_q'] = event_q
    app.router.add_get("/",   index_handler)
    app.router.add_get("/ws", ws_handler)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, "localhost", UI_PORT)
    await site.start()
    print(f"🌐 UI disponible sur http://localhost:{UI_PORT}", flush=True)
    return runner, site
