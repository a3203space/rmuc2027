"""Rulebook 5.7.4 telemetry is distinct from the robot's optical scene.

The broadcaster deliberately never serializes the authoritative frame wholesale.
Enemy pose fields are an exterior-only allowlist; radar adds its eight specified
information categories without exposing opponents' weapon heat or controllers.
"""

from .hud import actions, rules, weapon_status

EXTERIOR = frozenset("id slot team role x y z vx vy vz quaternion angular_velocity heading aim pitch radius alive hp max_hp powered model_slot model_asset chassis_size_m chassis_type design_name muzzle_height_m muzzle_reach_m armor_center_z_m armor_side_center_z_m armor_hit_face_offset_m armor_inset_m armor_mount_offset_m armor_hit_times steering_angles wheel_angular_velocity suspension legs hip_angle self_righting grounded tipped upright crouched deployed transform_allowed transform_progress arm_count arm_joints arm2_joints arm_tip arm2_tip arm_mode arm2_mode arm_extension carried_units launcher".split())
BUFFS = ("attack_multiplier", "defense_buff", "vulnerability", "cooling_multiplier",
         "cooling_bonus", "invulnerable", "crossing_buffs", "posture", "posture_enhanced")
PUBLIC_FRAME = ("api_version", "dt", "duration", "remaining", "season", "status", "tick", "time",
                "winner", "reason", "bases", "outposts", "positions")


class PublicEventLog:
    """Repeat safe kill/revival events across network snapshots without losing a tick."""
    def __init__(self):
        self.events = []

    def update(self, frame):
        self.events = [e for e in self.events if e['time'] >= frame['time'] - 15]
        seen = {e['id'] for e in self.events}
        for event in frame.get('events', []):
            if event.get('type') not in ('destroy', 'respawn'):
                continue
            key = event.get('id') or f"{frame['tick']}:{event['type']}:{event.get('robot')}:{event.get('killer')}"
            if key in seen:
                continue
            safe = {k: event[k] for k in ('type', 'tick', 'time', 'team', 'robot', 'killer') if k in event}
            safe.update(id=key, tick=event.get('tick', frame['tick']), time=event.get('time', frame['time']))
            self.events.append(safe)
            seen.add(key)
        self.events.sort(key=lambda e: (e['tick'], e['time']))
        return list(self.events)


def project(frame, observation, team, slot, match_id, *, include_static=False, public_events=()):
    own = next(r for r in frame["robots"] if r["team"] == team and r["slot"] == slot)
    now = frame["time"]
    blocked = frame["status"] != "finished" and own.get("fpv_blocked_until", 0) > now
    no_camera = own["role"] == "aerial"
    red_card = own.get("red_card", False)
    heat_cut = own.get("overheated", False) and own["heat"] > 1e-8
    camera_off = blocked or no_camera or red_card or heat_cut
    camera = observation.get("vision", {}).get(str(slot), {}).get("camera")
    visible = observation.get("vision", {}).get(str(slot), {})
    seen = set(visible.get("enemies", [])) | set(visible.get("allies", []))
    base = frame["bases"][team]
    logistics = max(.05, min(1, 1 - max(0, base["max_hp"] - base["hp"]) / 20000))
    result = {"protocol": 2, "match_id": match_id, "tick": frame["tick"],
              "team": team, "slot": slot, "self": own,
              "display": {"ui_blocked": blocked, "blocked_until": own.get("fpv_blocked_until", 0),
                          "camera_off": camera_off, "reason": "penalty" if blocked else "red_card" if red_card else
                          "hangar" if no_camera else "overheat" if heat_cut else "destroyed" if not own["alive"] else "",
                          "grayscale": not own['alive'] and not camera_off,
                          "logistics": logistics, "resolution_scale": logistics,
                          "camera": camera}}
    # No tactical payload can continue feeding hidden HUDs during a yellow card.
    if blocked:
        result["self"] = {"id": own["id"], "slot": slot, "team": team, "role": own["role"]}
        return result
    radar = []
    for robot in frame["robots"]:
        radar.append({**{k: robot[k] for k in ("id", "team", "slot", "role", "x", "y", "hp", "max_hp", "alive", "ammo")},
                      "kills": robot.get('kills', 0),
                      "buffs": {k: robot[k] for k in BUFFS if k in robot},
                      "armor_hit_times": robot.get("armor_hit_times", []),
                      **({"heading": robot["heading"]} if robot["id"] in seen or robot["id"] == own["id"] else {})})
    result["tactical"] = {
        "robots": radar,
        "teams": {side: {k: values.get(k, 0) for k in ("resources", "total_resources", "control", "score", "attack_damage")}
                  for side, values in frame["teams"].items()},
        "zones": {key: {"active": bool(value.get("occupants")), "owner": value.get("owner"),
                         "progress": value.get("progress", 0), "contested": value.get("contested", False),
                         "active_teams": sorted({r.split("-")[0] for r in value.get("occupants", [])})}
                  for key, value in frame.get("zones", {}).items()},
        "positions": frame.get("positions", {}), "bases": frame["bases"], "outposts": frame["outposts"],
        "decoded": True, "sample_time": now,
        "control_rules": {k: rules()["control"][k] for k in ("capture_s", "release_s")},
    }
    # Rust transport frames are immutable after publication. Share public values
    # between these projections instead of copying five full allies per player.
    scene = {key: frame[key] for key in PUBLIC_FRAME if key in frame}
    # A blind camera receives no moving scene, including projectiles and drones.
    scene["robots"] = [] if camera_off else [
        {k: v for k, v in r.items() if k in EXTERIOR}
        for r in frame["robots"] if r["id"] in seen or r["id"] == own["id"]]
    scene["projectiles"] = [] if camera_off else observation.get("projectiles", [])
    optical_drones = set(visible.get("drones", []))
    scene["visible_drones"] = [] if camera_off else [
        {**{k: d[k] for k in ("id", "x", "y", "z", "vx", "vy", "vz", "quaternion", "state")},
         "team": r["team"], "asset": r["model_asset"].replace("/robot-", "/drone-")}
        for r in frame["robots"] for d in r.get("drones", []) if d["id"] in optical_drones]
    # Only the owner's confirmed shots drive first-person recoil/muzzle effects.
    # Do not forward global shot positions or combat events from unseen robots.
    scene["events"] = [] if camera_off else [
        {k: v for k, v in e.items() if k in (
            "type", "id", "tick", "time", "robot", "team", "role", "x", "y", "z", "angle", "pitch", "muzzle_origin")}
        for e in frame.get("events", []) if e.get("type") == "shot" and e.get("robot") == own["id"]]
    scene["runes"] = observation.get("runes", {})
    scene["assemblies"] = {team: observation.get("assemblies", {}).get(team, {})}
    scene["zones"] = result["tactical"]["zones"]
    # Renderer models need exterior animation fields, never the private enemy drone list.
    scene["teams"] = result["tactical"]["teams"]
    result["scene"] = scene
    result["actions"] = actions(frame, own)
    result["weapon"] = weapon_status(frame, own)
    result["economy"] = {k: frame["teams"][team].get(k) for k in ("completed_difficulties", "energy_units_left", "launcher_unlock_s", "level_cap")}
    result["events"] = [e for e in frame.get("events", []) if e.get("robot") == own["id"] and e.get("type") in ("ammo_purchase", "ammo_delivered", "remote_heal", "respawn", "rpc_error")]
    result['public_events'] = list(public_events)
    result["assembly"] = observation.get("assemblies", {}).get(team, {})
    result["air"] = own.get("drones", []) if no_camera else []
    if include_static:
        result["static"] = {key: frame[key] for key in ("arena", "robot_models", "simulation_assumptions")}
    return result
