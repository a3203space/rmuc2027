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
    if current['sources'] != metadata['sources'] or current['sdk_hash'] != metadata['sdk_hash']:
        raise ValueError('Fixed strategy or SDK changed since training; export refused')
    for key in ('runtime_hash', 'engine_binary_sha256', 'rules_sha256'):
        if metadata.get(key, current[key]) != current[key]:
            raise ValueError(f'Training runtime changed: {key}')
    # The deployment side follows the observation and therefore supports swaps.
    runtime_config = {key: config[key] for key in ('slot', 'rl_seconds', 'grid_m', 'recover_enter', 'recover_exit')}
    source = Path(portable.__file__).read_text()
    _, entries = load_submission(resource(config['friendly']))
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
    # Integer scheduling agrees with the Gym session's exact game-tick count.
    if 'goal_time' not in state or obs['tick'] - state.get('goal_tick', 0) >= round(_CONFIG['rl_seconds']/.05):
        features = encode_observation(obs, state, _CONFIG['slot'], _CONFIG['rl_seconds'])
        select_goal(obs, state, mlp_forward(features, _LAYERS), _NAVIGATOR, _CONFIG)
        state['goal_tick'] = obs['tick']
    actions = execute_goal(obs, state, _NAVIGATOR, _CONFIG, _BASELINE)
    return actions, state
'''
    with tempfile.TemporaryDirectory(prefix='arena-rl-export-') as directory:
        policy = Path(directory)/'sentry.py'
        policy.write_text(code)
        files = {**entries, str(config['slot']): policy}
        manifest = save_submission(Path(output), 'squad', files, season='2027')
    metadata = {**metadata, 'portable_source_hash': hashlib.sha256(source.encode()).hexdigest(),
        'actor_layers_hash': hashlib.sha256(json.dumps(layers, sort_keys=True).encode()).hexdigest(),
        'sentry_source_bytes': len(code.encode()), 'deployment_version': manifest['version']}
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
