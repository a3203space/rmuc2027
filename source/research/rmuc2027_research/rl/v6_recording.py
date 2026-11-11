"""Export recorded frames without ending or otherwise mutating the match."""
import hashlib
import json
import os
from pathlib import Path


def save_snapshot_replay(battle, path, *, metadata=None, reason=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if len(battle.frames) != len(battle.actions) + 1:
        raise ValueError('Full replay needs one real frame per recorded command')
    last = battle.frames[-1]
    meta = {**battle.metadata, **(metadata or {})}
    meta.update(test_match=True, snapshot_playback=True,
                recording_end_tick=last['tick'], recording_end_time=last['time'],
                recording_stop_reason=reason)
    same_end = last['tick'] == battle.match.tick
    result = battle.result() if same_end else dict(winner=last.get('winner'), reason='recording_quota',
        time=last['time'], teams=last.get('teams', {}), digest=None, policy_errors=len(battle.diagnostics))
    encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    pending = path.with_suffix('.tmp')
    with pending.open('wb') as stream:
        stream.write(b'{"format":"rmuc-replay-v4","metadata":'); stream.write(encode(meta))
        stream.write(b',"static":'); stream.write(encode(battle.frames.static))
        stream.write(b',"frames":[')
        for index, raw in enumerate(battle.frames.encoded()):
            if index: stream.write(b',')
            stream.write(raw)
        stream.write(b'],"actions":'); stream.write(encode(battle.actions))
        stream.write(b',"diagnostics":'); stream.write(encode(battle.diagnostics))
        stream.write(b',"result":'); stream.write(encode(result)); stream.write(b'}')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                bytes=path.stat().st_size, frames=len(battle.frames),
                simulation_seconds=last['time'], metadata=meta)
