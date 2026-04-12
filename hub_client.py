"""DAGO Hub Client — connects this service to the Hub dashboard."""
import asyncio
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

HUB_URL = os.getenv("HUB_URL", "https://backend-production-4311.up.railway.app")
AGENT_ID = "dago-assistant"
AGENT_NAME = "DAGO Assistant"
DESCRIPTION = "Zendesk Carrier Support AI — ticket summaries, auto-replies"
SERVICE_URL = os.getenv("SERVICE_URL", "")

_client: Optional[httpx.AsyncClient] = None
_heartbeat_task: Optional[asyncio.Task] = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=10)
    return _client


async def register():
    """Register this agent with the Hub."""
    try:
        client = await _get_client()
        await client.post(f"{HUB_URL}/api/agents", json={
            "id": AGENT_ID,
            "name": AGENT_NAME,
            "description": DESCRIPTION,
            "url": SERVICE_URL,
        })
        logger.info(f"[hub] Registered as {AGENT_ID} at {HUB_URL}")
    except Exception as e:
        logger.warning(f"[hub] Failed to register: {e}")


async def heartbeat():
    """Send heartbeat to Hub."""
    try:
        client = await _get_client()
        await client.post(f"{HUB_URL}/api/agents/{AGENT_ID}/heartbeat", json={
            "status": "online",
        })
    except Exception:
        pass


async def emit(event_type: str, payload: dict | None = None):
    """Emit an event to the Hub."""
    try:
        client = await _get_client()
        await client.post(f"{HUB_URL}/api/events", json={
            "source": AGENT_ID,
            "type": event_type,
            "payload": payload or {},
        })
    except Exception:
        pass


async def _heartbeat_loop():
    while True:
        await heartbeat()
        await asyncio.sleep(30)


async def start():
    """Start Hub connection — register + heartbeat loop."""
    await register()
    global _heartbeat_task
    _heartbeat_task = asyncio.create_task(_heartbeat_loop())


async def stop():
    """Stop Hub connection."""
    if _heartbeat_task:
        _heartbeat_task.cancel()
    if _client and not _client.is_closed:
        await _client.aclose()
