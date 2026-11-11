"""Bounded-memory playback of canonical old and new RMUC 2027 recordings."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from fastapi.testclient import TestClient
from rmuc2027_core.replay import ReplayArchive
from rmuc2027_server.viewer import create_replay_app


class ReplayArchiveTests(unittest.TestCase):
    def fixture(self, directory):
        frames = [{'tick': i, 'time': i * .05, 'events': [{'message': '重装 {引号\\"} ' + 'a' * 18000}]} for i in range(9)]
        value = {'format': 'rmuc-replay-v4', 'metadata': {'map': 'rmuc2027'}, 'static': {'arena': {'name': 'rmuc2027'}}, 'frames': frames, 'actions': []}
        path = Path(directory) / 'replay.json'
        path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')))
        return path, frames

    def test_utf8_offsets_cross_chunk_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            path, frames = self.fixture(directory)
            archive = ReplayArchive(path)
            self.assertEqual(len(archive.offsets), 9)
            self.assertEqual(archive.frames(0, 2), frames[:2])
            self.assertEqual(archive.frames(7, 100), frames[7:])
            self.assertEqual(archive.frames(99, 1), [])

    def test_viewer_is_read_only_and_bounds_frame_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            path, frames = self.fixture(directory)
            options = {'use_uvloop': True} if importlib.util.find_spec('uvloop') else {}
            with TestClient(create_replay_app(path), backend_options=options) as client:
                header = client.get('/api/replay/header').json()
                self.assertEqual(header['count'], 9)
                self.assertEqual(client.get('/api/replay/frames?start=8&count=1').json()['frames'], frames[8:])
                self.assertEqual(client.get('/api/replay/frames?start=-1').status_code, 422)
                self.assertEqual(client.get('/api/replay/frames?start=0&count=101').status_code, 422)
                self.assertEqual(client.post('/api/lan/v1/join', json={}).status_code, 404)
                self.assertEqual(client.get('/').status_code, 200)

    def test_old_seasons_and_empty_frames_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.fixture(directory)
            value = json.loads(path.read_text())
            value['metadata']['map'] = 'removed-season'
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): ReplayArchive(path)
            value['metadata']['map'] = 'rmuc2027'
            value['frames'] = []
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError): ReplayArchive(path)
