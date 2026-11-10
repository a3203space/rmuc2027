"""Portable policy bundling without widening the existing policy sandbox."""
import base64
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import tempfile
import zlib

from rmuc2027_core.submissions import load_submission, save_submission
from . import portable
from .session import provenance, resource


def _pack_v4_actor(actor):
    """Lossless float32 byte planes, avoiding JSON's decimal weight overhead."""
    branches=('tactical','skill','revive') if actor.get('version')==6 else ('tactical','skill')
    header={key:copy.deepcopy(value) for key,value in actor.items() if key not in branches}
    header['shapes']={}; values=[]
    for branch in branches:
        header['shapes'][branch]=[]
        for layer in actor[branch]:
            rows=len(layer['weight']);columns=len(layer['weight'][0])
            if len(layer['bias']) != rows or any(len(row) != columns for row in layer['weight']):
                raise ValueError('Inconsistent v4 actor layer dimensions')
            header['shapes'][branch].append([rows,columns])
            values.extend(value for row in layer['weight'] for value in row)
            values.extend(layer['bias'])
    if not all(math.isfinite(value) for value in values):
        raise ValueError('V4 actor weights must be finite')
    weights=struct.pack('<%df'%len(values),*values)
    header=json.dumps(header,separators=(',',':'),allow_nan=False).encode()
    # Group equal byte positions so repeated exponent bits compress together.
    payload=struct.pack('<I',len(header))+header+b''.join(weights[i::4] for i in range(4))
    return base64.b85encode(zlib.compress(payload,9)).decode(),hashlib.sha256(weights).hexdigest()


_UNPACK_V4_ACTOR = '''
def _decode_v4_actor(encoded):
    import struct
    raw=zlib.decompress(base64.b85decode(encoded))
    header_size=struct.unpack_from('<I',raw)[0]
    actor=json.loads(raw[4:4+header_size])
    planes=raw[4+header_size:]
    if len(planes)%4:raise ValueError('Invalid v4 float32 payload')
    count=len(planes)//4
    weights=bytearray(len(planes))
    for lane in range(4):weights[lane::4]=planes[lane*count:(lane+1)*count]
    values=struct.unpack('<%df'%count,weights);offset=0
    for branch,shapes in actor.pop('shapes').items():
        actor[branch]=[]
        for rows,columns in shapes:
            weight=[list(values[offset+i*columns:offset+(i+1)*columns]) for i in range(rows)]
            offset+=rows*columns
            bias=list(values[offset:offset+rows]);offset+=rows
            actor[branch].append(dict(weight=weight,bias=bias))
    if offset!=count:raise ValueError('Inconsistent v4 float32 layer shapes')
    return actor
'''


def export_layers(layers, metadata, output, *, allow_sdk_transfer=False):
    """Bundle shared code + actor mean as a normal hashed five-entry squad.

    Public for testing the portable runtime without installing neural libraries.
    Production export obtains layers from PPO's Linear/Tanh actor.
    """
    config = copy.deepcopy(metadata['config'])
    if metadata.get('obs_version', portable.OBS_VERSION) != portable.policy_schema(config)['version']:
        raise ValueError('Observation version mismatch')
    current = provenance(config)
    if config.get('policy_version') in (3,4,6):
        for key in ('sentry_sources','sentry_schema'):
            if metadata.get(key)!=current[key]:raise ValueError(f'Training runtime changed: {key}')
    if current['navigation']!=metadata.get('navigation'):
        raise ValueError('Training runtime changed: navigation')
    if current['sources'] != metadata['sources'] or (current['sdk_hash'] != metadata['sdk_hash'] and not allow_sdk_transfer):
        raise ValueError('Fixed strategy or SDK changed since training; export refused')
    for key in ('runtime_hash', 'engine_binary_sha256', 'rules_sha256'):
        if metadata.get(key, current[key]) != current[key]:
            raise ValueError(f'Training runtime changed: {key}')
    metadata = copy.deepcopy(metadata)
    metadata['deployment_sdk_hash'] = current['sdk_hash']
    if current['sdk_hash'] != metadata['sdk_hash']:
        metadata['sdk_transfer'] = dict(training=metadata['sdk_hash'], deployment=current['sdk_hash'],
                                       weights_changed=False, requires_validation=True)
    # The deployment side follows the observation and therefore supports swaps.
    runtime_config = {key: config[key] for key in ('slot', 'rl_seconds', 'grid_m', 'recover_enter', 'recover_exit')}
    runtime_config['learning_control_hz'] = config.get('learning_control_hz', 10)
    runtime_config['navigation_mode'] = config.get('navigation_mode', 'legacy')
    runtime_config['policy_version'] = config.get('policy_version', 1)
    source = Path(portable.__file__).read_text()
    base_manifest, entries = load_submission(resource(config['friendly']))
    from rmuc2027_core.models import normalize_team_lineup
    lineup = normalize_team_lineup(config.get('lineups', {}).get(config['team'], base_manifest.get('team_lineup')), '2027')
    baseline = entries[str(config['slot'])].read_text()
    v4=config.get('policy_version') in (4,6)
    if v4:
        packed,weights_hash=_pack_v4_actor(layers)
    else:
        packed = base64.b85encode(zlib.compress(json.dumps(layers, allow_nan=False).encode())).decode()
    code = source + '\nimport base64, json, zlib\n'
    if config.get('policy_version') in (3,4,6):
        from . import sentry_v3
        code += 'from types import SimpleNamespace\n_sentry_namespace = {}\nexec('+repr(Path(sentry_v3.__file__).read_text())+', _sentry_namespace)\nsentry_v3 = SimpleNamespace(**_sentry_namespace)\n'
        if config.get('policy_version') in (4,6):
            from . import sentry_v4
            code += '_v4_namespace = {"__name__": "embedded_sentry_v4", "__package__": "", "sentry_v3": sentry_v3}\nexec('+repr(Path(sentry_v4.__file__).read_text())+', _v4_namespace)\nsentry_v4 = SimpleNamespace(**_v4_namespace)\n'
        if config.get('policy_version')==6:
            from . import recovery_belief,sentry_v6
            code += '_belief_namespace = {"__name__": "embedded_recovery_belief", "__package__": ""}\nexec('+repr(Path(recovery_belief.__file__).read_text())+', _belief_namespace)\nrecovery_belief = SimpleNamespace(**_belief_namespace)\n'
            code += '_v6_namespace = {"__name__": "embedded_sentry_v6", "__package__": "", "sentry_v3": sentry_v3, "sentry_v4": sentry_v4, "recovery_belief": recovery_belief}\nexec('+repr(Path(sentry_v6.__file__).read_text())+', _v6_namespace)\nsentry_v6 = SimpleNamespace(**_v6_namespace)\n'
    if v4:
        if config['slot'] != 7:raise ValueError('The v4 sentry actor requires slot 7')
        # Slot 7 has no scripted opening; its v4 executor never calls baseline.
        # Other squad roles retain their ordinary exported entry files.
        code += '_BASELINE = None\n'
    else:
        code += '_baseline_namespace = {}\nexec(' + repr(baseline) + ', _baseline_namespace)\n'
        code += 'from types import SimpleNamespace\n_BASELINE = SimpleNamespace(**_baseline_namespace)\n'
    code += '_CONFIG = ' + repr(runtime_config) + '\n'
    if v4:
        code += _UNPACK_V4_ACTOR+'\n_LAYERS = _decode_v4_actor(_PACKED_V4_ACTOR)\n'
    else:
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
        features = encode_observation(obs, state, _CONFIG['slot'], _CONFIG['rl_seconds'], _CONFIG)
        select_goal(obs, state, mlp_forward(features, _LAYERS), _NAVIGATOR, _CONFIG)
        state['goal_tick'] = obs['tick']
    if ('control_tick' not in state or obs['tick'] < state['control_tick']
            or obs['tick'] - state['control_tick'] >= round(1 / (_CONFIG['learning_control_hz'] * .05))):
        control_obs = dict(obs, policy_dt=1 / _CONFIG['learning_control_hz'], navigation_mode=_CONFIG['navigation_mode'])
        state['held_actions'] = execute_goal(control_obs, state, _NAVIGATOR, _CONFIG, _BASELINE)
        state['control_tick'] = obs['tick']
    return state['held_actions'], state
'''
    if v4:
        # Compress source separately: recompressing a large base85 weight literal
        # wastes the strict 512 KiB submission budget. Both payloads are lossless
        # and use only the standard library available in the policy sandbox.
        source_hash=hashlib.sha256(code.encode()).hexdigest()
        source_bytes=len(code.encode())
        source_payload=base64.b85encode(zlib.compress(code.encode(),9)).decode()
        code=('"""Lossless v4 float32 actor and versioned public runtime."""\n'
              'import base64, zlib\n_PACKED_V4_ACTOR = '+repr(packed)+'\n'
              'exec(zlib.decompress(base64.b85decode('+repr(source_payload)+')), globals())\n')
        metadata.update(actor_weight_encoding='float32-le-byteplanes-zlib-base85-v1',
            actor_float32_sha256=weights_hash, embedded_runtime_sha256=source_hash,
            embedded_runtime_bytes=source_bytes, embedded_runtime_encoding='zlib-base85-v1',
            actor_architecture=copy.deepcopy(layers.get('architecture',{})))
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


def export_model(model_path, output, *, allow_sdk_transfer=False):
    from stable_baselines3 import PPO
    from torch import nn
    model_path = Path(model_path)
    metadata = json.loads((model_path.parent/'experiment.json').read_text())
    if metadata['config'].get('policy_version')==6:
        from .v6_ppo import HybridPPO
        from .sentry_v6_policy import export_actor
        from .sentry_v6 import v6_feature_masks
        import numpy as np
        import torch
        model=HybridPPO.load(model_path,device='cpu')
        layers=export_actor(model.policy)
        inputs=np.random.default_rng(191).uniform(-1,1,(32,775)).astype(np.float32)
        masks=np.asarray([v6_feature_masks(row) for row in inputs],dtype=np.float32)
        observations=dict(actor=inputs,critic=np.zeros((32,454),np.float32),revive_mask=masks)
        expected=model.predict(observations,deterministic=True)[0]
        actual=np.asarray([portable.mlp_forward(row,layers) for row in inputs])
        actual[:,:6]=actual[:,:6].clip(-1,1)
        error=float(np.max(np.abs(expected-actual)))
        if error>1e-5:raise ValueError(f'Portable v6 actor parity failed: {error}')
        metadata['actor_parity_max_error']=error
        metadata['model_sha256']=hashlib.sha256(model_path.read_bytes()).hexdigest()
        return export_layers(layers,metadata,output,allow_sdk_transfer=allow_sdk_transfer)
    if metadata['config'].get('policy_version')==4:
        from .asymmetric_ppo import AsymmetricPPO
        from .sentry_v4_policy import export_actor
        import numpy as np
        import torch
        model=AsymmetricPPO.load(model_path,device='cpu')
        layers=export_actor(model.policy)
        inputs=np.random.default_rng(191).uniform(-1,1,(32,portable.policy_schema(metadata['config'])['obs_size'])).astype(np.float32)
        with torch.no_grad():expected=model.policy.mlp_extractor.policy_net(torch.as_tensor(inputs)).cpu().numpy()
        actual=np.asarray([portable.mlp_forward(row,layers) for row in inputs])
        error=float(np.max(np.abs(expected-actual)))
        if error>1e-5:raise ValueError(f'Portable v4 actor parity failed: {error}')
        metadata['actor_parity_max_error']=error
        metadata['model_sha256']=hashlib.sha256(model_path.read_bytes()).hexdigest()
        return export_layers(layers,metadata,output,allow_sdk_transfer=allow_sdk_transfer)
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
    inputs = rng.uniform(-1, 1, (32, portable.policy_schema(metadata['config'])['obs_size'])).astype(np.float32)
    with torch.no_grad():
        expected = model.policy.action_net(model.policy.mlp_extractor.policy_net(torch.as_tensor(inputs))).cpu().numpy()
    actual = np.asarray([portable.mlp_forward(row, layers) for row in inputs])
    error = float(np.max(np.abs(expected-actual)))
    if error > 1e-5:
        raise ValueError(f'Portable actor parity failed: {error}')
    metadata['actor_parity_max_error'] = error
    metadata['model_sha256'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return export_layers(layers, metadata, output, allow_sdk_transfer=allow_sdk_transfer)


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
