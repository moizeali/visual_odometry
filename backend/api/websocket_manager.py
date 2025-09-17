"""
WebSocket manager for real-time communication
"""

from fastapi import WebSocket
from typing import List
import json
import logging

logger = logging.getLogger(__name__)

class WebSocketManager:
    """Manage WebSocket connections for real-time updates"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept and store WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return

        message_str = json.dumps(message)
        disconnected = []

        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_processing_update(self, frame_number: int, total_frames: int,
                                   trajectory_point: List[float], stats: dict):
        """Send real-time processing update"""
        message = {
            "type": "processing_update",
            "frame": frame_number,
            "total_frames": total_frames,
            "progress": (frame_number / total_frames) * 100 if total_frames > 0 else 0,
            "trajectory_point": trajectory_point,
            "stats": stats
        }
        await self.broadcast(message)

    async def send_error(self, error_message: str):
        """Send error message to all clients"""
        message = {
            "type": "error",
            "message": error_message
        }
        await self.broadcast(message)