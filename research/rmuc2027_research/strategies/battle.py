from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json, time, threading, hashlib
from rmuc2027_core.engine import Match, POLICY_INTERVAL, DT, SLOT_IDS, RULES_PROFILE, STATIC_FRAME_KEYS
from rmuc2027_core.policy import FrameLog, Policy, EncodedObservation, POLICY_TIMEOUT_LIMIT
from rmuc2027_core.referee import validate_referee
from rmuc2027_core.submissions import load_submission
from rmuc2027_core.communication import TeamCommunication, validate_messages
from rmuc2027_core import seasons as season_rules
from .decks import prepare_decks
from .profiles import load_profile

class Battle:
    def __init__(self, red, blue, seed=42, duration=420, map_name=None, sandbox="docker", timeout=0.2, deadline=None, stop_event=None, decks=None, team_lineup=None, practice_lineup=False, free_team_strategies=None, battle_settings=None, controller_overrides=None, record_replay=True):
        # Overrides implement Policy.decide/close and replace single-slot entries.
        # They receive exactly the same slot-scoped observation as that entry.
        overrides = dict(controller_overrides or {})
        self.record_replay = record_replay
        # The field follows the strategies' rules season unless the caller names it.
        if map_name is None:
            from rmuc2027_core.submissions import manifest_season
            sources = [s for s in (red, blue) if s is not None] or [cards[0] for cards in (decks or {}).values() if cards]
            found = {manifest_season(load_submission(s)[0]) for s in sources}
            map_name = season_rules.map_of(found.pop() if len(found) == 1 else season_rules.DEFAULT_SEASON)
        season = season_rules.season_of_map(map_name)
        self.season, rules_profile, slot_ids = season, season_rules.rules_profile(season), season_rules.slots(season)
        lineups, deck_policies, deck_metadata = prepare_decks(decks, seed, free_team_strategies, season) if decks is not None else (None, None, None)
        if decks is None:
            if free_team_strategies:raise ValueError("Free controllers require free battle")
            from rmuc2027_core.models import normalize_team_lineup,mirrored_lineups
            manifests=[load_submission(source)[0] for source in (red,blue)]
            if any(m.get("rules_profile") != rules_profile for m in manifests):
                raise ValueError(f"Both strategies must be written for the {season} season to play on {map_name}")
            selected=[normalize_team_lineup(m.get('team_lineup'),season) for m in manifests]
            chosen=normalize_team_lineup(team_lineup,season) if team_lineup is not None else selected[0]
            if not practice_lineup and (selected[0]!=selected[1] or any(s!=chosen for s in selected)):
                raise ValueError('Team matches require identical registered robot lineups; develop both strategies on the same lineup')
            if chosen!=season_rules.default_lineup(season) or season!=season_rules.DEFAULT_SEASON:lineups=mirrored_lineups(chosen,season)
        elif team_lineup is not None:raise ValueError('Decks and shared team lineups cannot be combined')
        from rmuc2027_core.battle_settings import normalize
        battle_settings = normalize(battle_settings or {}, season)
        self.match, self.sandbox, self.policies, self.closed = Match(seed, duration, map_name, lineups=lineups, battle_settings=battle_settings), sandbox, [], False
        self.deck_mode = decks is not None
        self.deadline, self.stop_event = deadline, stop_event
        self.pool = ThreadPoolExecutor(max_workers=24 if self.deck_mode else 12)
        self.metadata = {"api_version": 3, "engine": "rust", "rules": rules_profile, "season": season, "decision_timeout_s":timeout, "policy_timeout_limit":POLICY_TIMEOUT_LIMIT, "roster":list(slot_ids), "test_match":duration!=420, "dt": DT, "policy_interval": POLICY_INTERVAL, "seed": seed, "duration": duration,
                         "map": map_name, "sandbox": sandbox, "match_mode":"deck" if self.deck_mode else "standard", "teams": {}}
        if battle_settings:
            self.metadata.update({"battle_settings": battle_settings, "test_match": True})
        self.frames, self.actions, self.diagnostics = FrameLog(self.match.snapshot()), [], []
        self._latest_frame = self.match.snapshot()
        self.metadata["rules_hash"] = self.frames[0]["rules_hash"]
        self.metadata["engine_hash"] = self.frames[0]["engine_hash"]
        self.held_commands = {"red": {}, "blue": {}}
        self.communication = TeamCommunication({team:[d["slot"] for d in items] for team, items in lineups.items()} if lineups else None, slot_ids)
        if lineups:
            self.metadata.update({"lineups":lineups,"test_match":True,"roster":{team:[d["slot"] for d in items] for team,items in lineups.items()}})
        if not self.deck_mode:
            self.metadata['team_lineup']=chosen
            self.metadata['test_match']=duration!=420 or practice_lineup or bool(battle_settings)
        self.pending_referee = []
        self.metadata["agents"] = {}
        try:
            for team, source in (("red", red), ("blue", blue)):
                if self.deck_mode:
                    self.metadata["teams"][team] = deck_metadata[team]
                    controller = (free_team_strategies or {}).get(team)
                    cards = deck_metadata[team]["cards"]
                    self.metadata["agents"][team] = {str(d["slot"]): ({**deck_metadata[team]["agent"], "strategy": deck_metadata[team].get("display_name") or Path(controller).name}
                        if controller else {**card["agent"], "strategy": card.get("display_name") or card["name"]}) for d, card in zip(lineups[team], cards)}
                    for slot, path, digest in deck_policies[team]:
                        if self.cancelled():
                            raise TimeoutError("Practice development window ended")
                        policy = Policy(path, seed*100+(0 if team == "red" else 10000)+(slot or 0), sandbox, timeout, expected_digest=digest,
                                        deadline=deadline,stop_event=stop_event,dynamic_messages=True)
                        self.policies.append((team,None if slot is None else [slot],policy))
                    continue
                manifest, entries = load_submission(source)
                if manifest.get("api_version") != 3 or manifest.get("rules_profile") != rules_profile:
                    raise ValueError(f"This submission does not use the {season} rules; create and submit a {season} strategy workspace")
                if manifest["mode"] not in ("team", "squad", "distributed"):
                    raise ValueError("Assemble robot submissions into a squad before matching")
                profile = load_profile(source)
                self.metadata["teams"][team] = {"name": Path(source).name, "version": manifest["version"], "mode": manifest["mode"], "agent": profile["agent"],
                                                **({"display_name": profile["display_name"]} if profile.get("display_name") else {})}
                self.metadata["agents"][team] = {str(slot): {**profile["agent"], "strategy": profile.get("display_name") or Path(source).name} for slot in slot_ids}
                # Each process sees exactly one strategy. Robot policy seed is stable per slot.
                for key, path in sorted(entries.items()):
                    if self.cancelled():
                        raise TimeoutError("Practice development window ended")
                    slots = None if manifest["mode"] == "team" else [int(key)]
                    override_key = (team, int(key)) if slots else None
                    policy = overrides.pop(override_key) if override_key in overrides else Policy(
                        path, seed * 100 + (0 if team == "red" else 10) + (int(key) if slots else 0), sandbox, timeout,
                        expected_digest=manifest["hashes"][manifest["entries"][key]],deadline=deadline,stop_event=stop_event)
                    self.policies.append((team, slots, policy))
            if overrides:
                raise ValueError("Controller overrides must replace existing single-slot entries")
        except BaseException:
            try:
                self.close()
            finally:
                for controller in overrides.values():
                    controller.close()
            raise

    def cancelled(self):
        return (self.deadline is not None and time.monotonic() >= self.deadline) or bool(self.stop_event and self.stop_event.is_set())

    def queue_referee(self, command):
        value = validate_referee(command, [robot["id"] for robot in self.match.robots])
        if self.match.status == "finished":
            raise ValueError("The match has ended")
        if len(self.pending_referee) >= 16:
            raise ValueError("Too many pending referee decisions")
        self.pending_referee.append(value)

    def observation(self, team, slots):
        observation = self.match.observation(team, slots)
        inboxes={slot:self.communication.inboxes[team][slot] for slot in observation["controlled_ids"]}
        observation["message_inboxes"] = inboxes
        observation["messages"] = [message for inbox in inboxes.values() for message in inbox]
        return observation

    def encoded_observations(self, policies):
        """The same observations as `observation`, fetched together and left encoded."""
        controlled, lines = self.match.observations([(team, slots) for team, slots, _ in policies])
        encoded = []
        for (team, _, _), ids, line in zip(policies, controlled, lines):
            inboxes = {slot: self.communication.inboxes[team][slot] for slot in ids}
            messages = [message for inbox in inboxes.values() for message in inbox]
            # Append both keys last, in the order `observation` adds them.
            tail = ',"message_inboxes":' + json.dumps(inboxes, allow_nan=False, separators=(",", ":")) + ',"messages":' + json.dumps(messages, allow_nan=False, separators=(",", ":"))
            encoded.append(EncodedObservation(line.rstrip(b"\n")[:-1] + tail.encode() + b"}\n", ids, True))
        return encoded

    def step(self):
        if self.match.status == "finished":
            return self._latest_frame
        if self.cancelled():
            self.match.finish(None, "stopped")
            self._latest_frame = self.match.snapshot()
            if self.record_replay:
                self.frames[-1] = self._latest_frame
            return self._latest_frame
        policy_tick = bool(self.policies) and self.match.tick % POLICY_INTERVAL == 0
        # Disabled policies return no actions, so they need no observation.
        deciding = [item for item in self.policies if not item[2].disabled] if policy_tick else []
        futures = []
        if deciding:
            for (team, slots, p), observation in zip(deciding, self.encoded_observations(deciding)):
                futures.append((team, slots, p, self.pool.submit(p.decide, observation)))
        commands, failures = ({"red": {}, "blue": {}} if policy_tick else self.held_commands), []
        outgoing = []
        for team, slots, p, future in futures:
            actions, error = future.result()
            owned = ({str(slot) for slot in slots} if slots is not None else
                     {str(robot['slot']) for robot in self.match.robots if robot['team'] == team})
            if not set(actions) <= owned or set(actions) & set(commands[team]):
                raise ValueError("Controller returned an unowned or multiply controlled robot")
            for slot, action in actions.items():
                action = dict(action)
                messages = action.pop("messages", [])
                if messages:
                    outgoing.append((team, slot, messages))
                commands[team][slot] = action
            if error:
                scope = "team" if slots is None else "-".join(map(str, slots))
                detail = {"type": "policy_error", "id": f"policy:{team}:{scope}:{self.match.tick}",
                          "tick": self.match.tick, "time": self.match.time, "team": team, "slots": slots,
                          "disabled": p.disabled, "consecutive_timeouts": p.consecutive_timeouts,
                          "message": error, "logs": list(p.logs)[-8:]}
                self.diagnostics.append(detail)
                failures.append(detail)
        self.held_commands = commands
        communications = self.communication.route(outgoing, self.match.tick, self.match.tick + POLICY_INTERVAL) if policy_tick else []
        if self.pending_referee:
            commands = {**commands, "referee": self.pending_referee}
            self.pending_referee = []
        # Communication is recorded alongside actions, but never sent to physics
        # or repeated during the held-action tick. Replay verification uses the
        # recorded robot commands; strategies are not re-executed.
        if self.record_replay:
            self.actions.append({**commands, "communication": communications} if communications else commands)
        frame = self.match.step(commands)
        frame["events"].extend(failures)
        # Reuse the engine's encoding unless policy errors were added to the events.
        if self.record_replay:
            self.frames.append(frame, None if failures else self.match.raw_frame)
        self._latest_frame = frame
        return frame

    def run(self):
        while self.match.status != "finished":
            self.step()
        return self.result()

    def result(self):
        return {"winner": self.match.winner, "reason": self.match.reason, "time": self.match.time,
                "teams": self.match.snapshot()["teams"], "digest": getattr(self, "_digest_cache", None) or self.match.digest(), "policy_errors": len(self.diagnostics)}

    def replay(self):
        if not self.record_replay:
            raise RuntimeError("Replay recording was disabled")
        return {"format": "rmuc-replay-v4", "metadata": self.metadata, "static":self.frames.static,"frames": [json.loads(raw) for raw in self.frames.encoded()],
                "actions": self.actions, "diagnostics": self.diagnostics, "result": self.result()}

    def save(self, path):
        if not self.record_replay:
            raise RuntimeError("Replay recording was disabled")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encode = lambda value: json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
        with path.open("wb") as stream:
            stream.write(b'{"format":"rmuc-replay-v4","metadata":')
            stream.write(encode(self.metadata))
            stream.write(b',"static":');stream.write(encode(self.frames.static))
            stream.write(b',"frames":[')
            for index, raw in enumerate(self.frames.encoded()):
                if index:
                    stream.write(b",")
                stream.write(raw)
            stream.write(b'],"actions":')
            stream.write(encode(self.actions))
            stream.write(b',"diagnostics":')
            stream.write(encode(self.diagnostics))
            stream.write(b',"result":')
            stream.write(encode(self.result()))
            stream.write(b"}")

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                self._digest_cache = self.match.digest()
            except (RuntimeError, OSError, ValueError, KeyError, TypeError):
                self._digest_cache = None
            self.pool.shutdown(wait=True, cancel_futures=True)
            if self.policies:
                with ThreadPoolExecutor(max_workers=12) as pool:
                    # Retrieve every result even if one cleanup fails.
                    futures = [pool.submit(item[2].close) for item in self.policies]
                    for future in futures:
                        try:
                            future.result()
                        except (OSError, RuntimeError):
                            pass
        finally:
            self.match.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
