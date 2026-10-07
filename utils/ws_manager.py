"""utils/ws_manager.py — WebSocket connection manager and event emitters.

Backend-agnostic: no GraphRAG or route-level coupling.  Any FastAPI app
that streams progress over a WS can reuse this directly.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

from fastapi import WebSocket

from utils.logger import logger


class ConnectionManager:
    """Tracks WebSocket connections and emits JSON events to them.

    Emits are best-effort: if a client has disconnected, the failure is
    logged and the connection is dropped from the registry.  Callers never
    need to check liveness.
    """

    def __init__(self) -> None:
        self.active_connections: Dict[str, WebSocket] = {}

    # ---- lifecycle -------------------------------------------------------
    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str) -> None:
        self.active_connections.pop(client_id, None)

    # ---- low-level emit --------------------------------------------------
    async def _send(self, client_id: str, payload: Dict[str, Any]) -> None:
        ws = self.active_connections.get(client_id)
        if ws is None:
            return
        try:
            await ws.send_text(json.dumps(payload))
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            self.disconnect(client_id)

    # ---- public emit API -------------------------------------------------
    async def event(self, client_id: str, payload: Dict[str, Any]) -> None:
        """Send a raw event dict, auto-stamping `timestamp` if missing."""
        payload = dict(payload)
        payload.setdefault("timestamp", datetime.now().isoformat())
        await self._send(client_id, payload)

    async def progress(
        self, client_id: str, stage: str, progress: int, message: str
    ) -> None:
        await self.event(client_id, {
            "type": "progress",
            "stage": stage,
            "progress": progress,
            "message": message,
        })

    async def qa_update(self, client_id: str, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload.setdefault("type", "qa_update")
        await self.event(client_id, payload)

    async def error(self, client_id: str, stage: str, message: str) -> None:
        await self.event(client_id, {
            "type": "error",
            "stage": stage,
            "message": message,
        })