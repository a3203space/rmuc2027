"""Local immutable RMUC 2027 policy bundles. No upload or organizer platform."""
import hashlib, json, os, secrets
from pathlib import Path
from . import seasons as season_rules
from .models import model_ids, normalize_team_lineup
from .worker import SOURCE_LIMIT
SLOT_IDS = (1,3,4,6,7)
RULES_PROFILE = RULES_PROFILE_2027 = 'rmuc-2027-v1.0.0-arena-v1'
def manifest_season(manifest):
    if manifest.get('rules_profile') != RULES_PROFILE:
        raise ValueError('Only RMUC 2027 policy bundles are supported')
    return '2027'

def submission_keys(mode, robot=None, slots=SLOT_IDS, season=None):
    if season is not None:
        slots = season_rules.slots(season)
    if mode == "team":
        return {"team"}
    if mode in ("squad", "distributed"):
        return {str(slot) for slot in slots}
    if (mode == "robot" and robot in slots) or (mode == "adaptive" and robot in model_ids()
                                                and (season is None or season_rules.legal(robot, season))):
        return {str(robot)}
    raise ValueError("Invalid submission mode or robot slot")



def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + secrets.token_hex(6) + ".tmp")
    try:
        temporary.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)



def save_submission(output, mode, entries, robot=None, team_lineup=None, showcase=None, profile=None, season=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Submission exists; choose a new version name")
    if season is None:
        season = season_rules.infer_season(team_lineup.keys()) if mode == "team" and team_lineup else \
            season_rules.infer_season(entries.keys()) if mode in ("squad", "distributed") and len(entries) == 5 else season_rules.DEFAULT_SEASON
    lineup=normalize_team_lineup(team_lineup,season) if mode=='team' else None
    if mode!='team' and team_lineup is not None:raise ValueError('A shared team lineup is only for team mode')
    if set(entries) != submission_keys(mode, robot, season=season):
        raise ValueError("Entries do not match the selected control mode")
    if any(p.is_symlink() or not p.is_file() or p.stat().st_size > SOURCE_LIMIT for p in entries.values()):
        raise ValueError(f"Strategy must be a regular file of at most {SOURCE_LIMIT // 1024} KiB")
    sources = {key: p.read_bytes() for key, p in entries.items()}
    for source in sources.values():
        compile(source, "strategy.py", "exec")
    sidecars = {}
    if showcase is not None:
        sidecars["showcase.json"] = json.loads(showcase) if isinstance(showcase, (str, bytes)) else showcase
    if profile is not None:
        sidecars["profile.json"] = profile
    output.mkdir(parents=True)
    manifest = {"api_version": 3, "rules_profile":season_rules.rules_profile(season), "mode": mode, "robot": robot, "entries": {}, "hashes": {}}
    if mode=='team':manifest['team_lineup']=lineup
    for slot, source in sources.items():
        name = "strategy.py" if mode in ("team", "robot", "adaptive") else f"robot_{slot}.py"
        (output / name).write_bytes(source)
        manifest["entries"][slot] = name
        manifest["hashes"][name] = hashlib.sha256(source).hexdigest()
    manifest["version"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    write_json(output / "manifest.json", manifest)
    # Display sidecars are not hashed: the code alone is the version.
    for name, value in sidecars.items():
        write_json(output / name, value)
    return manifest



def load_submission(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    mode = manifest.get("mode")
    season = manifest_season(manifest)
    slots = season_rules.slots(season)
    expected = submission_keys(mode, manifest.get("robot"), slots)
    if manifest.get("api_version") != 3 or (mode in ("distributed", "adaptive") and manifest.get("api_version") != 3) or set(manifest["entries"]) != expected:
        raise ValueError("Invalid submission manifest")
    if mode == "robot" and manifest.get("robot") not in slots:
        raise ValueError("Invalid robot slot")
    if mode=='team':normalize_team_lineup(manifest.get('team_lineup'),season)
    # Refuse nested paths and symlinks before resolving or mounting anything.
    for name in manifest["entries"].values():
        p = path / name
        if Path(name).name != name or p.is_symlink() or not p.is_file() or p.stat().st_size > SOURCE_LIMIT:
            raise ValueError("Invalid strategy path")
        if sha(p) != manifest["hashes"].get(name):
            raise ValueError(f"Submitted strategy changed: {name}")
    version_body = {k: v for k, v in manifest.items() if k != "version"}
    version = hashlib.sha256(json.dumps(version_body, sort_keys=True).encode()).hexdigest()[:16]
    if version != manifest.get("version"):
        raise ValueError("Submission manifest changed")
    return manifest, {key: path / name for key, name in manifest["entries"].items()}


