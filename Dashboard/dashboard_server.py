from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
import os

logger = logging.getLogger("DashboardServer")

app = FastAPI()

# Store connected clients
class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        # History cache: Key = "SYMBOL_TIMEFRAME" (e.g. "EURUSD_M1"), Value = List of candles
        self.history = {} 

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Send ALL stored history to new client
        if self.history:
            # Flatten all history into a single list for bulk sending
            all_history = []
            for key, items in self.history.items():
                all_history.extend(items)
            
            if all_history:
                # Send in chunks if too large (optional, but good practice)
                await websocket.send_text(json.dumps({"type": "history", "data": all_history}))
                
        logger.info(f"New client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        # Update history logic moved to push_data for cleaner separation
        # Broadcast raw message to all clients
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

# Endpoint for main.py to push data
@app.post("/push")
async def push_data(data: list | dict = Body(...)):
    """Receive data (single or batch) from trading engine and broadcast to web clients"""
    if isinstance(data, list):
        logger.debug(f"Received batch push: {len(data)} items")
        
        # Store in history cache
        for item in data:
            if item.get("type") == "candle":
                symbol = item.get("symbol", "EURUSD")
                tf = item.get("timeframe", "M1")
                key = f"{symbol}_{tf}"
                
                if key not in manager.history:
                    manager.history[key] = []
                
                manager.history[key].append(item)
                # Keep last 1000 candles per key
                if len(manager.history[key]) > 1000:
                    manager.history[key] = manager.history[key][-1000:]
                    
        # Broadcast as a single history batch
        self_data = {"type": "history", "data": data}
        await manager.broadcast(json.dumps(self_data))
    else:
        logger.debug(f"Received single push: {data.get('type')}")
        
        # Store single item if it's a candle
        if data.get("type") == "candle":
            symbol = data.get("symbol", "EURUSD")
            tf = data.get("timeframe", "M1")
            key = f"{symbol}_{tf}"
            
            if key not in manager.history:
                manager.history[key] = []
                
            manager.history[key].append(data)
            if len(manager.history[key]) > 1000:
                manager.history[key] = manager.history[key][-1000:]
        
        # NOTE: Main engine now handles LLM analysis generation and sends it with the signal.
        # Server just relays the 'analysis' field if present.
        
        await manager.broadcast(json.dumps(data))
    return {"status": "ok"}

# WebSocket for frontend
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Just keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Serve Frontend
static_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if not os.path.exists(static_path):
    # Fallback to legacy static if React build not found
    static_path = os.path.join(os.path.dirname(__file__), "static")
    if not os.path.exists(static_path):
        os.makedirs(static_path)

app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
