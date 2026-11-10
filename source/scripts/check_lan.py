"""Exercise 10 player intents against a real host loop and verify its replay.

This is an in-process load and deterministic-replay check, not physical LAN or
desktop rendering evidence. Use the documented two-machine checklist separately.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rmuc2027_core.replay import verify_replay
from rmuc2027_server import SLOTS
from rmuc2027_server.protocol import Input, Join, Seat
from rmuc2027_server.room import Room


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=int, default=420)
    parser.add_argument("--players", type=int, default=10, choices=range(1, 11))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fast", action="store_true", help="Run without real-time pacing; does not prove 20 Hz wall-clock scheduling")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory for each run")
    args.output.mkdir(parents=True)
    room = Room(owner_key="local-validation", directory=args.output, duration=args.duration, realtime=not args.fast)
    tokens, generations, pending, delays, seen = [], {}, {}, [], set()
    for index, (team, slot) in enumerate((team, slot) for team in ("red", "blue") for slot in SLOTS):
        if index >= args.players:
            break
        result = room.join(Join(nickname=f"load-{index + 1}", host_key="local-validation" if index == 0 else None))
        token = result["token"]
        tokens.append(token)
        generations[token] = room.attach(token)
        room.select(token, Seat(team=team, slot=slot))
        room.ready(token, True)
    wall_start = time.monotonic()
    room.start(tokens[0])
    last_report = wall_start
    try:
        while room.status in ("starting", "running", "settling"):
            now = time.monotonic()
            if now - wall_start > max(60, args.duration * 1.5 + 30):
                raise RuntimeError("Match exceeded its wall-clock bound")
            if room.status == "running":
                for token in tokens:
                    snapshot = room.snapshot(token)
                    view = snapshot.get("player")
                    if not view:
                        continue
                    ack = snapshot["ack_seq"]
                    # Acknowledge the newest input executed by the Rust step.
                    if ack >= 0 and (token, ack) in pending:
                        delays.append((now - pending[(token, ack)]) * 1000)
                        for key in [key for key in pending if key[0] == token and key[1] <= ack]:
                            del pending[key]
                    robot = view["self"]
                    if "x" not in robot:
                        continue
                    target = (24 if view["team"] == "red" else 4, 7.5)
                    angle = math.atan2(target[1] - robot["y"], target[0] - robot["x"])
                    seq = room.player(token).seq + 1
                    command = Input(match_id=room.match_id, seq=seq, forward=.3 if view["slot"] in (3, 4, 7) else 0,
                                    yaw=angle, pitch=.03, fire=view["slot"] in (3, 4, 7))
                    room.submit(token, command)
                    pending[(token, seq)] = now
                    scene = view.get("scene", {})
                    seen.update(scene.get("events", []))
            if now - last_report >= 30:
                print(json.dumps({"elapsed_s": round(now - wall_start, 1), "tick": room.runner.tick, "players": len(tokens)}), flush=True)
                last_report = now
            time.sleep(.02 if not args.fast else .001)
        room.runner.thread.join(timeout=15)
        if room.error or not room.result:
            raise RuntimeError(room.error or "No final match result")
        verified = verify_replay(args.output / room.result["replay"])
        samples = sorted(delays)
        report = {"transport": "in-process", "desktop_rendering": "not_checked", "physical_lan": "not_checked",
                  "players": len(tokens), "realtime": not args.fast, "duration": args.duration,
                  "wall_seconds": time.monotonic() - wall_start, "result": room.result, "verification": verified,
                  "input_execution_p95_ms": samples[min(len(samples) - 1, int(len(samples) * .95))] if samples else None,
                  "input_execution_samples": len(samples)}
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False), flush=True)
    finally:
        room.close()


if __name__ == "__main__":
    main()
