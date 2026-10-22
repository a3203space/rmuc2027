"""Rulebook 5.7.4 telemetry is distinct from the robot's optical scene.

The broadcaster deliberately never serializes the authoritative frame wholesale.
Enemy pose fields are an exterior-only allowlist; radar adds its eight specified
information categories without exposing opponents' weapon heat or controllers.
"""


EXTERIOR = frozenset("id slot team role x y z vx vy vz quaternion angular_velocity heading aim pitch radius alive hp max_hp powered model_slot model_asset chassis_size_m chassis_type design_name muzzle_height_m muzzle_reach_m armor_center_z_m armor_side_center_z_m armor_hit_face_offset_m armor_inset_m armor_mount_offset_m armor_hit_times steering_angles wheel_angular_velocity suspension legs hip_angle self_righting grounded tipped upright crouched deployed transform_allowed transform_progress arm_count arm_joints arm2_joints arm_tip arm2_tip arm_mode arm2_mode arm_extension carried_units launcher".split())
BUFFS = ("attack_multiplier", "defense_buff", "vulnerability", "cooling_multiplier",
         "cooling_bonus", "invulnerable", "crossing_buffs", "posture", "posture_enhanced")
PUBLIC_FRAME = ("api_version", "dt", "duration", "remaining", "season", "status", "tick", "time",
                "winner", "reason", "bases", "outposts", "positions")


def project(frame, observation, team, slot, match_id, *, include_static=False):
    own = next(r for r in frame["robots"] if r["team"] == team and r["slot"] == slot)
    now = frame["time"]
    blocked = frame["status"] != "finished" and own.get("fpv_blocked_until", 0) > now
    no_camera = own["role"] == "aerial"
    red_card = own.get("red_card", False)
    heat_cut = own.get("overheated", False) and own["heat"] > 1e-8
    camera_off = blocked or no_camera or red_card or not own["alive"] or heat_cut
    camera = observation.get("vision", {}).get(str(slot), {}).get("camera")
    visible = observation.get("vision", {}).get(str(slot), {})
    seen = set(visible.get("enemies", [])) | set(visible.get("allies", []))
    base = frame["bases"][team]
    logistics = max(.05, min(1, 1 - max(0, base["max_hp"] - base["hp"]) / 20000))
    result = {"protocol": 1, "match_id": match_id, "tick": frame["tick"],
              "team": team, "slot": slot, "self": own,
              "display": {"ui_blocked": blocked, "blocked_until": own.get("fpv_blocked_until", 0),
                          "camera_off": camera_off, "reason": "penalty" if blocked else "red_card" if red_card else
                          "hangar" if no_camera else "destroyed" if not own["alive"] else "overheat" if heat_cut else "",
                          "logistics": logistics, "resolution_scale": logistics,
                          "camera": camera}}
    # No tactical payload can continue feeding hidden HUDs during a yellow card.
    if blocked:
        result["self"] = {"id": own["id"], "slot": slot, "team": team, "role": own["role"]}
        return result
    radar = []
    for robot in frame["robots"]:
        radar.append({**{k: robot[k] for k in ("id", "team", "slot", "role", "x", "y", "hp", "max_hp", "alive", "ammo")},
                      "buffs": {k: robot[k] for k in BUFFS if k in robot},
                      "armor_hit_times": robot.get("armor_hit_times", [])})
    result["tactical"] = {
        "robots": radar,
        "teams": {side: {k: values.get(k, 0) for k in ("resources", "total_resources", "control", "score")}
                  for side, values in frame["teams"].items()},
        "zones": {key: {"active": bool(value.get("occupants")), "owner": value.get("owner")}
                  for key, value in frame.get("zones", {}).items()},
        "positions": frame.get("positions", {}), "bases": frame["bases"], "outposts": frame["outposts"],
        "decoded": True, "sample_time": now,
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
    scene["events"] = []
    scene["runes"] = observation.get("runes", {})
    scene["assemblies"] = {team: observation.get("assemblies", {}).get(team, {})}
    scene["zones"] = {key: {"owner": value.get("owner"), "progress": 0, "contested": False}
                      for key, value in frame.get("zones", {}).items()}
    # Renderer models need exterior animation fields, never the private enemy drone list.
    scene["teams"] = result["tactical"]["teams"]
    result["scene"] = scene
    result["assembly"] = observation.get("assemblies", {}).get(team, {})
    result["air"] = own.get("drones", []) if no_camera else []
    if include_static:
        result["static"] = {key: frame[key] for key in ("arena", "robot_models", "simulation_assumptions")}
    return result
