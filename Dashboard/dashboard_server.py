from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.staticfiles import StaticFiles
import asyncio
import json
import logging
import os

logger = logging.getLogger("DashboardServer")

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.history = {} 

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if self.history:
            all_history = []
            for key, items in self.history.items():
                all_history.extend(items)
            
            if all_history:
                await websocket.send_text(json.dumps({"type": "history", "data": all_history}))
        logger.info(f"New client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.post("/push")
async def push_data(data: list | dict = Body(...)):
    if isinstance(data, list):
        logger.debug(f"Received batch push: {len(data)} items")
        
        for item in data:
            if item.get("type") == "candle":
                symbol = item.get("symbol", "EURUSD")
                tf = item.get("timeframe", "M1")
                key = f"{symbol}_{tf}"
                
                if key not in manager.history:
                    manager.history[key] = []
                
                manager.history[key].append(item)
                if len(manager.history[key]) > 1000:
                    manager.history[key] = manager.history[key][-1000:]
                    
        self_data = {"type": "history", "data": data}
        await manager.broadcast(json.dumps(self_data))
    else:
        logger.debug(f"Received single push: {data.get('type')}")
        
        if data.get("type") == "candle":
            symbol = data.get("symbol", "EURUSD")
            tf = data.get("timeframe", "M1")
            key = f"{symbol}_{tf}"
            
            if key not in manager.history:
                manager.history[key] = []
                
            manager.history[key].append(data)
            if len(manager.history[key]) > 1000:
                manager.history[key] = manager.history[key][-1000:]
        
        await manager.broadcast(json.dumps(data))
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

static_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if not os.path.exists(static_path):
    static_path = os.path.join(os.path.dirname(__file__), "static")
    if not os.path.exists(static_path):
        os.makedirs(static_path)

app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
