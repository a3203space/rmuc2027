"""RMUC 2027 rule and construction catalog."""
import copy
import json
from functools import lru_cache

from .paths import ROOT

DEFAULT_SEASON = "2027"



@lru_cache(maxsize=1)
def _config():
    return json.loads((ROOT / "configs" / "rules.json").read_text(encoding="utf-8"))


def config():
    return _config()


def seasons():
    return tuple(sorted(_config()["seasons"]))


def season(key):
    try:
        return _config()["seasons"][str(key)]
    except KeyError:
        raise ValueError(f"Unknown rules season {key}") from None


def season_of_map(name):
    try:
        return _config()["maps"][name].get("season", DEFAULT_SEASON)
    except KeyError:
        raise ValueError("Unknown map") from None


def map_of(key):
    return season(key)["map"]


def slots(key):
    return tuple(item["slot"] for item in season(key)["roster"])


def roles(key):
    return {item["slot"]: item["role"] for item in season(key)["roster"]}


def rules_profile(key):
    return season(key)["rules_profile"]


def profiles():
    return {rules_profile(key): key for key in seasons()}


def season_of_profile(profile):
    return profiles().get(profile)


def rulebook(key):
    return _config()[season(key)["rules_key"]]


def specs(key):
    value = season(key).get("specs")
    return copy.deepcopy(value if value is not None else _config()["specs"])


def simulation(key):
    value = copy.deepcopy(_config()["simulation_assumptions"])
    value.update(copy.deepcopy(season(key).get("simulation_overrides") or {}))
    return value


def default_lineup(key):
    return {str(slot): int(model) for slot, model in season(key)["default_lineup"].items()}


def legal(model, key):
    design = _config()["robot_designs"].get(str(model))
    return bool(design and design.get("seasons", {}).get(str(key), {}).get("legal", False))


def legal_models(key, role=None):
    return tuple(sorted(int(k) for k, d in _config()["robot_designs"].items()
                        if legal(k, key) and (role is None or d["role"] == role)))


def infer_season(slot_keys):
    """The season whose roster matches a set of team lineup slots."""
    wanted = {int(s) for s in slot_keys}
    for key in seasons():
        if set(slots(key)) == wanted:
            return key
    raise ValueError("RMUC 2027 requires slots 1,3,4,6,7")


def summary():
    """Public description of every season for clients."""
    return {key: {"label": season(key)["label"], "rules_profile": rules_profile(key), "map": map_of(key),
                  "slots": list(slots(key)), "roles": {str(s): r for s, r in roles(key).items()},
                  "team_size": len(slots(key)), "default_lineup": default_lineup(key),
                  "legal_models": list(legal_models(key)),
                  "excluded_systems": rulebook(key).get("adaptations", {}).get("excluded_systems", [])}
            for key in seasons()}


def request_season(season=None, mode=None, robot=None, lineup=None, entries=None, default=DEFAULT_SEASON):
    """The season a request means: an explicit season, else the one its lineup, distributed
    entries or robot pin down, else `default` (RMUC 2027)."""
    if season:
        if str(season) not in seasons():
            raise ValueError(f"Unknown rules season {season}")
        return str(season)
    if lineup:
        return infer_season(lineup.keys())
    if mode == "distributed" and entries:
        return infer_season(entries)
    if robot is not None and mode == "adaptive":
        fits = [key for key in seasons() if legal(robot, key)]
    elif robot is not None and mode == "robot":
        fits = [key for key in seasons() if int(robot) in slots(key)]
    else:
        fits = []
    if len(fits) == 1:
        return fits[0]
    return default if not fits or default in fits else fits[0]
