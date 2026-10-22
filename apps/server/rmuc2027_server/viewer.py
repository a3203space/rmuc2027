"""Local read-only playback. This app is separate from the LAN room surface."""
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from rmuc2027_core.paths import ROOT
from rmuc2027_core.replay import ReplayArchive


def create_replay_app(source):
    archive = ReplayArchive(source)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get('/api/replay/header')
    def header():
        return {'metadata': archive.metadata, 'static': archive.static, 'count': len(archive.offsets)}

    @app.get('/api/replay/frames')
    def frames(start: int = Query(ge=0), count: int = Query(default=60, ge=1, le=100)):
        return {'start': start, 'frames': archive.frames(start, count)}

    @app.get('/')
    def page():
        return FileResponse(ROOT / 'build/web/apps/replay-viewer/index.html')

    app.mount('/assets', StaticFiles(directory=ROOT / 'build/web/assets'), name='assets')
    return app
