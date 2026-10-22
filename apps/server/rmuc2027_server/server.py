"""LAN-only FastAPI surface. No strategy upload, training, or referee routes."""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import socket
import sys
import threading
import time

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from rmuc2027_core.engine import binary_path
from rmuc2027_core.paths import ROOT
from . import PROTOCOL
from .protocol import Input, Join, Operation, Profile, Ready, Seat
from .room import Room


def create_app(room=None):
    room = room or Room()

    @asynccontextmanager
    async def lifespan(app):
        async def sweep():
            while True:
                await asyncio.sleep(1)
                with room.lock:
                    room.expire()
        task = asyncio.create_task(sweep())
        try:
            yield
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.to_thread(room.close)

    app = FastAPI(title="Arena LAN", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.room = room

    def authorize(authorization):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "需要玩家会话")
        token = authorization[7:]
        with room.lock:
            if token not in room.players:
                raise HTTPException(401, "会话已过期")
        return token

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.middleware("http")
    async def boundary(request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in (f"http://{request.headers.get('host')}", f"https://{request.headers.get('host')}"):
            return JSONResponse({"detail": "Invalid origin"}, status_code=403)
        try:
            if int(request.headers.get("content-length", 0)) > 16384:
                return JSONResponse({"detail": "Request too large"}, status_code=413)
        except ValueError:
            return JSONResponse({"detail": "Invalid body size"}, status_code=400)
        return await call_next(request)

    @app.get("/api/lan/v1/health")
    def health():
        return {"protocol": PROTOCOL, "service": "arena-lan", "season": "2027", "simulation_hz": 20,
                "input_hz": 50, "max_players": 10, "manual_sentry": True, "radar_decode": "always_success"}

    @app.get("/api/lan/v1/room")
    def state():
        with room.lock:
            room.expire()
        return room.state()

    @app.post("/api/lan/v1/join")
    def join(body: Join):
        return room.join(body)

    @app.post("/api/lan/v1/seat")
    def seat(body: Seat, authorization: str | None = Header(default=None)):
        return room.select(authorize(authorization), body)

    @app.post("/api/lan/v1/ready")
    def ready(body: Ready, authorization: str | None = Header(default=None)):
        return room.ready(authorize(authorization), body.ready)

    @app.post("/api/lan/v1/profile")
    def profile(body: Profile, authorization: str | None = Header(default=None)):
        return room.performance(authorize(authorization), body)

    @app.post("/api/lan/v1/start")
    def start(authorization: str | None = Header(default=None)):
        return room.start(authorize(authorization))

    @app.post("/api/lan/v1/leave")
    def leave(authorization: str | None = Header(default=None)):
        return room.leave(authorize(authorization))

    @app.websocket("/api/lan/v1/session")
    async def session(websocket: WebSocket):
        origin = websocket.headers.get("origin")
        host = websocket.headers.get("host")
        if origin and origin not in (f"http://{host}", f"https://{host}"):
            await websocket.close(code=1008)
            return
        await websocket.accept()
        token, generation = None, None
        sender = None
        try:
            auth = await asyncio.wait_for(websocket.receive_json(), timeout=5)
            if not isinstance(auth, dict) or auth.get("type") != "auth" or auth.get("protocol") != PROTOCOL or not isinstance(auth.get("token"), str):
                raise ValueError("协议版本不兼容或缺少会话")
            token = auth["token"]
            generation = room.attach(token)

            async def broadcast():
                sent_static = None
                while True:
                    value = await asyncio.to_thread(room.snapshot, token, room.match_id != sent_static)
                    if value.get("static"):
                        sent_static = room.match_id
                    await asyncio.wait_for(websocket.send_json(value), timeout=2)
                    await asyncio.sleep(.05)

            sender = asyncio.create_task(broadcast())
            while not sender.done():
                # A heartbeat is required even when no robot key is being held.
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=3)
                if len(raw) > 16384:
                    raise ValueError("消息过大")
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("消息必须是 JSON 对象")
                if value.get("type") == "ping":
                    room.touch(token)
                    continue
                try:
                    message = Input.model_validate(value) if value.get("type") == "input" else Operation.model_validate(value)
                    receipt = room.submit(token, message)
                    # Receipts travel with snapshots so only one coroutine writes the socket.
                    if isinstance(message, Operation) and receipt["status"] == "duplicate":
                        with room.lock:
                            p = room.player(token)
                            p.receipts = (p.receipts + [receipt])[-8:]
                except (ValueError, ValidationError) as exc:
                    with room.lock:
                        p = room.player(token)
                        p.receipts = (p.receipts + [{"status": "rejected", "message": str(exc)[:300], "seq": value.get("seq")}])[-8:]
                if sender.done():
                    sender.result()
        except ValueError as exc:
            if generation is None:
                try:
                    await asyncio.wait_for(websocket.send_json({"type": "error", "fatal": True, "message": str(exc)}), timeout=1)
                except (RuntimeError, OSError, asyncio.TimeoutError):
                    pass
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError, KeyError, TypeError, OSError):
            pass
        finally:
            if sender:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
            if token and generation is not None:
                room.detach(token, generation)
            try:
                await websocket.close()
            except (RuntimeError, OSError):
                pass

    @app.get("/api/lan/v1/replay")
    def replay(authorization: str | None = Header(default=None)):
        authorize(authorization)
        with room.lock:
            if not room.result:
                raise HTTPException(404, "回放尚未保存")
            path = room.directory / room.result["replay"]
        return FileResponse(path, filename=path.name)

    frontend = ROOT / "build/web"

    @app.get("/play")
    def play():
        page = frontend / "apps/client/web/index.html"
        if not page.is_file():
            raise HTTPException(503, "请先在 Arena 目录执行 npm run build，生成仿真画面")
        return FileResponse(page)

    if (frontend / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=frontend / "assets"), name="assets")
    return app


def serve(host="0.0.0.0", port=8766, *, owner_key=None, parent_stdin=False, directory=None):
    import uvicorn
    binary_path()
    room = Room(owner_key=owner_key, directory=directory)
    app = create_app(room)
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="warning", ws_max_size=16384))
    # QProcess closes stdin when the owning client exits; both platforms then use
    # the ordinary lifespan cleanup instead of leaving AI subprocesses behind.
    if parent_stdin:
        def parent_watch():
            for line in sys.stdin:
                if line.strip() == "shutdown":
                    break
            room.stop_event.set()
            server.should_exit = True
        threading.Thread(target=parent_watch, name="arena-lan-parent", daemon=True).start()

    def readiness():
        while not server.started and not server.should_exit:
            time.sleep(.05)
        if server.started:
            addresses = {"127.0.0.1"}
            try:
                addresses.update(socket.gethostbyname_ex(socket.gethostname())[2])
            except OSError:
                pass
            print(json.dumps({"lan_ready": True, "port": port, "protocol": PROTOCOL,
                              "owner_key": room.owner_key, "addresses": sorted(addresses)}), flush=True)
    threading.Thread(target=readiness, name="arena-lan-ready", daemon=True).start()
    server.run()
