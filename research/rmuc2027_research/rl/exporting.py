"""Portable policy bundling without widening the existing policy sandbox."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import zlib

from rmuc2027_core.submissions import load_submission, save_submission
from . import portable
from .session import provenance, resource


def export_layers(layers, metadata, output):
    """Bundle shared code + actor mean as a normal hashed five-entry squad.

    Public for testing the portable runtime without installing neural libraries.
    Production export obtains layers from PPO's Linear/Tanh actor.
    """
    config = copy.deepcopy(metadata['config'])
    if metadata.get('obs_version', portable.OBS_VERSION) != portable.OBS_VERSION:
        raise ValueError('Observation version mismatch')
    current = provenance(config)
    if current['navigation']!=metadata.get('navigation'):
        raise ValueError('Training runtime changed: navigation')
    if current['sources'] != metadata['sources'] or current['sdk_hash'] != metadata['sdk_hash']:
        raise ValueError('Fixed strategy or SDK changed since training; export refused')
    for key in ('runtime_hash', 'engine_binary_sha256', 'rules_sha256'):
        if metadata.get(key, current[key]) != current[key]:
            raise ValueError(f'Training runtime changed: {key}')
    # The deployment side follows the observation and therefore supports swaps.
    runtime_config = {key: config[key] for key in ('slot', 'rl_seconds', 'grid_m', 'recover_enter', 'recover_exit')}
    runtime_config['learning_control_hz'] = config.get('learning_control_hz', 10)
    runtime_config['navigation_mode'] = config.get('navigation_mode', 'legacy')
    source = Path(portable.__file__).read_text()
    base_manifest, entries = load_submission(resource(config['friendly']))
    from rmuc2027_core.models import normalize_team_lineup
    lineup = normalize_team_lineup(config.get('lineups', {}).get(config['team'], base_manifest.get('team_lineup')), '2027')
    baseline = entries[str(config['slot'])].read_text()
    packed = base64.b85encode(zlib.compress(json.dumps(layers, allow_nan=False).encode())).decode()
    code = source + '\nimport base64, json, zlib\n'
    code += '_baseline_namespace = {}\nexec(' + repr(baseline) + ', _baseline_namespace)\n'
    code += 'from types import SimpleNamespace\n_BASELINE = SimpleNamespace(**_baseline_namespace)\n'
    code += '_CONFIG = ' + repr(runtime_config) + '\n'
    code += '_LAYERS = json.loads(zlib.decompress(base64.b85decode(' + repr(packed) + ')))\n'
    code += '_NAVIGATOR = SdkNavigator(_CONFIG["grid_m"])\n'
    code += '''
def act(obs, state):
    state = dict(state or {})
    opening = fixed_opening(obs, state, _CONFIG, _BASELINE)
    if opening is not None:
        return opening, state
    # Integer scheduling agrees with the Gym session's exact game-tick count.
    if 'goal_time' not in state or obs['tick'] - state.get('goal_tick', 0) >= round(_CONFIG['rl_seconds']/.05):
        features = encode_observation(obs, state, _CONFIG['slot'], _CONFIG['rl_seconds'])
        select_goal(obs, state, mlp_forward(features, _LAYERS), _NAVIGATOR, _CONFIG)
        state['goal_tick'] = obs['tick']
    if ('control_tick' not in state or obs['tick'] < state['control_tick']
            or obs['tick'] - state['control_tick'] >= round(1 / (_CONFIG['learning_control_hz'] * .05))):
        control_obs = dict(obs, policy_dt=1 / _CONFIG['learning_control_hz'], navigation_mode=_CONFIG['navigation_mode'])
        state['held_actions'] = execute_goal(control_obs, state, _NAVIGATOR, _CONFIG, _BASELINE)
        state['control_tick'] = obs['tick']
    return state['held_actions'], state
'''
    with tempfile.TemporaryDirectory(prefix='arena-rl-export-') as directory:
        policy = Path(directory)/'learned_role.py'
        policy.write_text(code)
        files = {**entries, str(config['slot']): policy}
        manifest = save_submission(Path(output), 'squad', files, season='2027', team_lineup=lineup)
    from rmuc2027_core.battle_settings import defaults
    assembly_settings = {k:v for k,v in defaults('2027').items() if k.startswith('timed_assembly_profile.')}
    metadata = {**metadata, 'portable_source_hash': hashlib.sha256(source.encode()).hexdigest(),
        'actor_layers_hash': hashlib.sha256(json.dumps(layers, sort_keys=True).encode()).hexdigest(),
        'sentry_source_bytes': len(code.encode()), 'learned_slots': [config['slot']],
        'deployment': {'team_lineup': lineup, 'battle_settings': {**assembly_settings, 'timed_assembly': config.get('timed_assembly', False), 'drones_enabled': config.get('drones_enabled', True)}}, 'deployment_version': manifest['version']}
    (Path(output)/'rl-metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def export_model(model_path, output):
    from stable_baselines3 import PPO
    from torch import nn
    model_path = Path(model_path)
    metadata = json.loads((model_path.parent/'experiment.json').read_text())
    model = PPO.load(model_path, device='cpu')
    modules = list(model.policy.mlp_extractor.policy_net.children())
    if not all(isinstance(layer, (nn.Linear, nn.Tanh)) for layer in modules):
        raise ValueError('Only the documented Linear/Tanh actor can be exported')
    layers = [{'weight': layer.weight.detach().cpu().tolist(), 'bias': layer.bias.detach().cpu().tolist()}
              for layer in [*modules, model.policy.action_net] if isinstance(layer, nn.Linear)]
    # Validate actual weights against torch, rather than testing only a mock net.
    import numpy as np
    import torch
    rng = np.random.default_rng(191)
    inputs = rng.uniform(-1, 1, (32, portable.OBS_SIZE)).astype(np.float32)
    with torch.no_grad():
        expected = model.policy.action_net(model.policy.mlp_extractor.policy_net(torch.as_tensor(inputs))).cpu().numpy()
    actual = np.asarray([portable.mlp_forward(row, layers) for row in inputs])
    error = float(np.max(np.abs(expected-actual)))
    if error > 1e-5:
        raise ValueError(f'Portable actor parity failed: {error}')
    metadata['actor_parity_max_error'] = error
    metadata['model_sha256'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return export_layers(layers, metadata, output)


def compose_exports(sources, output):
    """Combine independently learned roles; never silently overwrite a role."""
    from rmuc2027_core.models import normalize_team_lineup
    files, learned, members, deployment = {}, set(), [], None
    for source in sources:
        source = Path(source)
        manifest, entries = load_submission(source)
        metadata = json.loads((source/'rl-metadata.json').read_text())
        if metadata.get('deployment_version') != manifest['version']:
            raise ValueError('Export metadata does not match its submission')
        spec = metadata.get('deployment')
        if spec is None:
            raise ValueError('Re-export this model to include its deployment settings')
        normalize_team_lineup(spec['team_lineup'], '2027')
        if deployment is not None and deployment != spec:
            raise ValueError('Cannot compose models with different lineups or battle settings')
        deployment = spec
        slots = {str(slot) for slot in metadata['learned_slots']}
        if learned & slots:
            raise ValueError('Multiple learned models claim the same slot')
        if not files:
            files.update(entries)
        for slot in slots:
            files[slot] = entries[slot]
        learned.update(slots)
        members.append({'version': manifest['version'], 'learned_slots': sorted(slots)})
    if not members:
        raise ValueError('At least one exported model is required')
    manifest = save_submission(Path(output), 'squad', files, season='2027', team_lineup=deployment['team_lineup'])
    result = {'deployment_version': manifest['version'], 'deployment': deployment,
              'learned_slots': sorted(map(int, learned)), 'members': members}
    (Path(output)/'rl-metadata.json').write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
