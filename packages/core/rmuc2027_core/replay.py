"""Deterministic replay verification for the matching RMUC 2027 engine build."""
import json
from pathlib import Path
from .engine import Match

def verify_replay(path):
    replay = json.loads(Path(path).read_text(encoding="utf-8"))
    if replay.get("format") not in ("rmuc-replay-v3", "rmuc-replay-v4"):
        raise ValueError("Unsupported 2027 replay format")
    if replay["format"] == "rmuc-replay-v4":
        replay["frames"] = [{**replay["static"], **frame} for frame in replay["frames"]]
    meta = replay["metadata"]
    if meta.get("map") != "rmuc2027":
        raise ValueError("Only RMUC 2027 replays are supported")
    if len(replay["frames"]) != len(replay["actions"]) + 1:
        raise ValueError("Frame/action count mismatch")
    options = {"lineups": meta["lineups"]} if meta.get("lineups") else {}
    if meta.get("battle_settings"):
        options["battle_settings"] = meta["battle_settings"]

    def canonical(frame):
        return {**frame, "events": [event for event in frame["events"] if event["type"] != "policy_error"]}

    with Match(meta["seed"], meta["duration"], meta["map"], **options) as match:
        if meta.get("engine_hash") != match.snapshot().get("engine_hash"):
            raise ValueError("Replay was recorded with a different Rust engine build")
        if not replay["actions"] and replay["result"]["reason"] == "stopped":
            match.finish(None, "stopped")
        if canonical(match.snapshot()) != canonical(replay["frames"][0]):
            raise ValueError("Initial frame mismatch")
        for index, commands in enumerate(replay["actions"], 1):
            frame = match.step(commands)
            if index == len(replay["actions"]) and replay["result"]["reason"] == "stopped":
                match.finish(None, "stopped")
                frame = match.snapshot()
            if canonical(frame) != canonical(replay["frames"][index]):
                raise ValueError(f"Replay diverges at tick {index}")
        if match.digest() != replay["result"]["digest"]:
            raise ValueError("Final digest mismatch")
        return {"verified_ticks": len(replay["actions"]), "digest": match.digest(), "winner": match.winner}


class ReplayArchive:
    """Index canonical 2027 replays with bounded memory, including old full games.

    The recorder writes metadata/static before frames. Original replay bytes stay
    intact; only byte offsets are held. Actions are not loaded for playback.
    """
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.offsets = []
        self.metadata, self.static = {}, {}
        decoder = json.JSONDecoder()
        with self.path.open('r', encoding='utf-8', newline='') as stream:
            buffer, position, eof = '', 0, False

            def fill():
                nonlocal buffer, eof
                chunk = stream.read(65536)
                buffer += chunk
                eof = not chunk

            def consume(count):
                nonlocal buffer, position
                position += len(buffer[:count].encode('utf-8'))
                buffer = buffer[count:]

            def whitespace():
                while True:
                    consume(len(buffer) - len(buffer.lstrip()))
                    if buffer or eof:
                        return
                    fill()

            def punctuation(expected):
                whitespace()
                if not buffer.startswith(expected):
                    raise ValueError(f'Invalid replay: expected {expected}')
                consume(1)

            def value():
                whitespace()
                start = position
                while True:
                    try:
                        item, end = decoder.raw_decode(buffer)
                        length = len(buffer[:end].encode('utf-8'))
                        consume(end)
                        return item, (start, length)
                    except json.JSONDecodeError:
                        if eof:
                            raise ValueError('Incomplete replay') from None
                        fill()

            punctuation('{')
            while True:
                key, _ = value()
                punctuation(':')
                if key == 'frames':
                    if self.metadata.get('map') != 'rmuc2027':
                        raise ValueError('Expected a canonical RMUC 2027 replay')
                    punctuation('[')
                    whitespace()
                    while not buffer.startswith(']'):
                        frame, location = value()
                        if not isinstance(frame, dict):
                            raise ValueError('Invalid replay frame')
                        self.offsets.append(location)
                        whitespace()
                        if buffer.startswith(']'):
                            break
                        punctuation(',')
                    if not self.offsets:
                        raise ValueError('Replay has no frames')
                    break
                item, _ = value()
                if key == 'metadata':
                    self.metadata = item
                elif key == 'static':
                    self.static = item
                elif key == 'format' and item not in ('rmuc-replay-v3', 'rmuc-replay-v4'):
                    raise ValueError('Unsupported 2027 replay format')
                punctuation(',')

    def frames(self, start, count):
        result = []
        with self.path.open('rb') as stream:
            for offset, length in self.offsets[start:start + count]:
                stream.seek(offset)
                result.append(json.loads(stream.read(length)))
        return result
