"""Resolve exported RL match requirements for CLI and Arena web alike."""
import json
from pathlib import Path
from .submissions import load_submission
from .models import normalize_team_lineup
from .battle_settings import normalize


def deployment_settings(sources, settings=None, *, explicit=True, season='2027'):
    requested = dict(settings or {})
    required = {}
    for source in sources:
        path = Path(source)/'rl-metadata.json'
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text())
        spec = metadata.get('deployment')
        if spec is None:  # Older exports retain their original behavior.
            continue
        manifest, _ = load_submission(source)
        if metadata.get('deployment_version') != manifest['version']:
            raise ValueError('RL deployment metadata does not match the selected strategy')
        if normalize_team_lineup(spec['team_lineup'], season) != normalize_team_lineup(manifest.get('team_lineup'), season):
            raise ValueError('RL deployment lineup differs from its manifest')
        normalize(spec['battle_settings'], season)
        for key, value in spec['battle_settings'].items():
            if key in required and required[key] != value:
                raise ValueError(f'RL strategies require conflicting battle setting: {key}')
            required[key] = value
    for key, value in required.items():
        if explicit and key in requested and requested[key] != value:
            raise ValueError(f'Battle setting conflicts with RL training: {key}')
        requested[key] = value
    return normalize(requested, season)
