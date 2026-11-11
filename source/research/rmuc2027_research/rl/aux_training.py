"""Explicitly invoked auxiliary initialization and training stages.

No training runs at import time. The caller supplies the shared monotonic wall
deadline; collection, imitation and PPO all charge the same training budget.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from . import privileged_aux as contract


def command_to_action(obs, command):
    """Convert a native 2027 teacher command to actions and per-head labels.

    Missing human actions must NOT enter here as empty commands: this conversion
    is for a native teacher whose omitted commands really mean no request.
    """
    r = contract.own_robot(obs); d = contract.sdk._design(r)
    action = contract.neutral_action(); supervised = [0.]*len(action)
    def put(name, value):
        i = contract.ACTION_INDEX[name]; action[i] = float(value); supervised[i] = 1.
    def cat(name, value):
        try: put(name, contract.CATEGORIES[name].index(value))
        except ValueError: pass
    # A complete script command specifies absence/no-op for categorical heads.
    for name in contract.CAT_NAMES: put(name, 0)
    flip = obs['team'] == 'blue'; sign = -1 if flip else 1
    if 'move' in command:
        put('move_x', float(command['move'][0])*sign); put('move_y', float(command['move'][1])*sign)
    if command.get('aim') is not None:
        cat('target', 'manual')
        angle = (command['aim']-(math.pi if flip else 0)+math.pi) % math.tau-math.pi
        put('aim_yaw', angle/math.pi); put('aim_pitch', command.get('pitch', 0)/(math.pi/2))
    if 'yaw_rate' in command:
        cat('turn', 'yaw_rate'); put('yaw_rate', command['yaw_rate']/12)
    elif 'heading' in command:
        cat('turn', 'heading'); put('heading', ((command['heading']-(math.pi if flip else 0)+math.pi) % math.tau-math.pi)/math.pi)
    if 'power' in command: cat('power_mode', 'explicit'); put('power', command['power']/100-1)
    cat('brake', 'brake' if command.get('brake', True) else 'coast')
    if 'transform' in command or 'crouch' in command: cat('transform', 'fold' if command.get('transform', command.get('crouch')) else 'unfold')
    if command.get('jump'):
        cat('jump', 'on'); put('jump_strength', 2*float(command['jump'])-1)
    if 'leg_height' in command:
        cat('leg_mode', 'height'); lo, hi = d.get('leg_min_m', .1), d.get('leg_max_m', .5)
        put('leg_height', 2*(command['leg_height']-lo)/max(1e-9, hi-lo)-1)
    if 'lean' in command: put('lean', command['lean']/.35)
    for name in ('fire', 'remote_heal', 'revive', 'rune_start', 'install_launcher', 'assembly_insert', 'assembly_lift', 'assembly_flip', 'assembly_confirm'):
        cat(name, 'on' if command.get(name, command.get('assembly_translate', False) if name == 'assembly_lift' else False) else 'off')
    profile = command.get('profile') or {}
    if isinstance(profile, dict):
        if 'chassis' in profile: cat('chassis_profile', profile['chassis'])
        if 'weapon' in profile: cat('weapon_profile', profile['weapon'])
    if command.get('buy_ammo'):
        order = command['buy_ammo']; remote = bool(order.get('remote'))
        cat('ammo', 'remote' if remote else 'local')
        kind = '42mm' if r['role'] == 'heavy' else '17mm'
        batch = contract.sdk.rules(obs)['economy']['ammo'][kind]['remote_batch' if remote else 'local_batch']
        amount = order.get('amount', batch)/max(1, batch)
        if amount in contract.CATEGORIES['ammo_batches']: cat('ammo_batches', amount)
        else:
            # Coarser purchase grid is explicit. Unknown amount is unsupervised.
            put('ammo_batches', min(range(4), key=lambda k: abs(contract.CATEGORIES['ammo_batches'][k]-amount)))
            supervised[contract.ACTION_INDEX['ammo_batches']] = 0.
    if command.get('pick_unit'):
        arm = command.get('pickup_arm', r.get('carried_units', 0) % 2 if r.get('arm_count') == 2 else 0)
        cat('pickup', 'arm2' if arm else 'arm')
    if command.get('rescue') is not None:
        target = command['rescue']
        if isinstance(target, str):
            target = next((p['slot'] for p in obs['robots'] if p['id'] == target), None)
        cat('rescue', target)
    if 'assembly' in command: cat('assembly', command['assembly'])
    if 'assembly_core' in command: cat('assembly_core', command['assembly_core'])
    for name in ('assembly_rotate', 'assembly_rotate2'):
        if name in command: put(name, command[name]/180)
    for arm in ('arm', 'arm2'):
        a = command.get(arm)
        if not a: continue
        if a.get('stow'): cat(arm+'_mode', 'stow')
        elif 'joints' in a:
            cat(arm+'_mode', 'joints')
            limits = d.get(arm, {}).get('limits', [[-math.pi, math.pi]]*7)
            for i, (value, (lo, hi)) in enumerate(zip(a['joints'], limits)):
                put(f'{arm}_joint{i}', 2*(value-lo)/max(1e-9, hi-lo)-1)
        elif 'tip' in a:
            cat(arm+'_mode', 'tip'); xyz = list(a['tip']); direction = list(a.get('dir', [1, 0, 0]))
            if a.get('frame') != 'robot':
                c, s = math.cos(r['heading']), math.sin(r['heading'])
                x, y = xyz[0]-r['x'], xyz[1]-r['y']; xyz = [c*x+s*y, -s*x+c*y, xyz[2]-r.get('z', 0)]
                x, y = direction[:2]; direction[:2] = [c*x+s*y, -s*x+c*y]
            for k, value in zip(('x', 'y', 'z'), (xyz[0]/1.5, xyz[1]/1.5, xyz[2]/.8-1)): put(arm+'_'+k, value)
            put(arm+'_dir_yaw', math.atan2(direction[1], direction[0])/math.pi)
            put(arm+'_dir_pitch', math.atan2(direction[2], math.hypot(*direction[:2]))/(math.pi/2))
            put(arm+'_roll', a.get('roll', 0)/math.pi); put(arm+'_grip', a.get('grip', 0)*2-1)
    continuous, allowed = contract.split_mask(contract.action_mask(obs)); choices = {}
    for name in contract.CAT_NAMES:
        i = contract.ACTION_INDEX[name]; value = int(action[i])
        conditional = contract.conditional_allowed(name, choices)
        if not allowed[name][value] or conditional is not None and value not in conditional:
            action[i] = 0; supervised[i] = 0
        choices[name] = int(action[i])
        # A forced choice carries no imitation information.
        if sum(allowed[name]) <= 1: supervised[i] = 0
    for i in range(contract.N_CONT):
        action[i] = max(-1., min(1., action[i]))
        supervised[i] *= continuous[i]
    active = contract.active_heads(action)
    supervised = [a*b for a, b in zip(supervised, active)]
    # Scripts assert several assembly operations at once. Teach only the engine's
    # current operation, never label ignored flags as evidence of a learned skill.
    core = obs.get('assemblies', {}).get(obs['team'], {})
    operations = [c.get('operation') for c in core.get('cores', [])] or [core.get('operation')]
    if any(operations):
        names = {'assembly_insert': 'inserting', 'assembly_lift': 'lifting',
                 'assembly_flip': 'flipping', 'assembly_confirm': 'verifying',
                 'assembly_rotate': 'rotating', 'assembly_rotate2': 'rotating'}
        for name, op in names.items():
            if op not in operations: supervised[contract.ACTION_INDEX[name]] = 0
    return action, supervised


def make_teacher(config):
    """Isolated native script teacher; no fixed-opening takeover in execution."""
    from .session import baseline_module
    module = baseline_module(config)
    class Teacher:
        def __init__(self): self.state = {}
        def reset(self): self.state.clear()
        def __call__(self, obs):
            if hasattr(module, 'Context') and hasattr(module, 'decide'):
                # Bypass the baseline's catch-all no-op fallback so data defects
                # cannot silently masquerade as successful demonstrations.
                fields = module.decide(contract.own_robot(obs), module.Context(obs, self.state))
                return contract.sdk.action(**fields)
            actions, self.state = module.act(obs, self.state)
            return actions.get(obs['controlled_ids'][0], {})
    return Teacher()


def collect_demonstrations(env, *, deadline, episodes=2, episode_seconds=30., teacher=None, seed=42):
    from .radar_calibration import RadarCalibrationRecorder
    recorder = RadarCalibrationRecorder()
    teacher = teacher or make_teacher(env.session.config)
    records = []; started = time.monotonic(); completed = 0
    for episode in range(episodes):
        if time.monotonic() >= deadline: break
        env.reset(seed=seed+episode); teacher.reset()
        episode_key = f'aux-{env.session.config["slot"]}-{seed+episode}'
        recorder.observe(env.session.battle.match, episode_key)
        start = env.session.obs.get('time', 0)
        while time.monotonic() < deadline:
            obs = env.session.obs
            command = teacher(obs)
            action, head_mask = command_to_action(obs, command)
            encoded = contract.encode(obs)
            records.append((encoded, action, head_mask))
            _, _, terminated, truncated, _ = env.step(np.asarray(action, np.float32))
            recorder.observe(env.session.battle.match, episode_key)
            if terminated or truncated or env.session.obs.get('time', 0)-start >= episode_seconds: break
        completed += 1
    return {'observations': {key: np.asarray([r[0][key] for r in records], np.float32) for key in ('actor', 'action_mask')},
            'actions': np.asarray([r[1] for r in records], np.float32),
            'head_mask': np.asarray([r[2] for r in records], np.float32),
            'radar_calibration_rows': recorder.rows(),
            'provenance': {'source': 'native_2027_script', 'schema_hash': contract.SCHEMA_HASH,
                           'episodes': completed, 'rows': len(records), 'seconds': time.monotonic()-started}}


def imitation_loss(policy, obs, actions, head_mask):
    """Bounded expert controls are regression targets, including exact endpoints.

    PPO continues to use its exact transformed density. Expert values +/-1 do
    not imply infinite latent Gaussian observations, so imitation instead fits
    deterministic means and categorical choices using their explicit masks.
    """
    import torch
    from .aux_policy import _active
    dist = policy.get_distribution(obs)
    mask = head_mask * _active(actions)
    continuous_mask = mask[:, :contract.N_CONT] * obs['action_mask'][:, :contract.N_CONT]
    errors = torch.tanh(dist.mean)-actions[:, :contract.N_CONT]
    columns = []
    for index in range(contract.N_CONT):
        error = errors[:, index]
        if index in (contract.ACTION_INDEX['heading'], contract.ACTION_INDEX['aim_yaw']):
            angle = error*math.pi
            circular = torch.atan2(torch.sin(angle), torch.cos(angle))/math.pi
            # Auto-target aim is a small offset, not a full-circle heading.
            error = torch.where(actions[:, contract.ACTION_INDEX['target']] == 1, circular, error) if index == contract.ACTION_INDEX['aim_yaw'] else circular
        columns.append(error.square())
    continuous = (torch.stack(columns, dim=1)*continuous_mask).sum()/continuous_mask.sum().clamp_min(1)
    losses = []; choices = {}; offset = 0
    for name in contract.CAT_NAMES:
        value = actions[:, contract.ACTION_INDEX[name]].long()
        losses.append(-dist._categorical(name, offset, choices).log_prob(value))
        choices[name] = value; offset += len(contract.CATEGORIES[name])
    discrete_mask = mask[:, contract.N_CONT:]
    categorical = (torch.stack(losses, dim=1)*discrete_mask).sum()/discrete_mask.sum().clamp_min(1)
    return continuous+categorical, continuous, categorical


def behavior_clone(model_or_policy, demonstrations, *, deadline, epochs=4, batch_size=64, seed=42, learning_rate=5e-5):
    import torch
    policy = getattr(model_or_policy, 'policy', model_or_policy)
    data = demonstrations; count = len(data['actions']); rng = np.random.default_rng(seed)
    losses = []; seen = 0; started = time.monotonic()
    parameters = list(policy.actor_net.parameters()) + list(policy.mean_net.parameters()) + list(policy.logits_net.parameters())
    # Imitation must not populate PPO Adam moments or update critic/log_std.
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    policy.set_training_mode(True)
    for _ in range(epochs):
        order = rng.permutation(count)
        for start in range(0, count, batch_size):
            if time.monotonic() >= deadline: break
            ids = order[start:start+batch_size]
            obs = {key: torch.as_tensor(value[ids], device=policy.device) for key, value in data['observations'].items()}
            actions = torch.as_tensor(data['actions'][ids], device=policy.device)
            mask = torch.as_tensor(data['head_mask'][ids], device=policy.device)
            loss, _, _ = imitation_loss(policy, obs, actions, mask)
            if not torch.isfinite(loss): raise ValueError('Non-finite auxiliary imitation loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, .5); optimizer.step()
            losses.append(float(loss.detach())); seen += len(ids)
        if time.monotonic() >= deadline: break
    return dict(source=data.get('provenance', {}), rows=count, sampled_rows=seen,
                objective='bounded_mean_mse_and_masked_categorical_ce', optimizer_scope='actor_only_separate_adam',
                supervised_heads=np.asarray(data['head_mask']).sum(axis=0).tolist() if count else [],
                loss=losses[-1] if losses else None, updates=len(losses), seconds=time.monotonic()-started)


def initialize_model(env, *, device='cpu', seed=42, ppo=None):
    from stable_baselines3 import PPO
    from .aux_policy import AuxLayerNormPolicy
    settings = dict(n_steps=64, batch_size=64, n_epochs=4, learning_rate=5e-5,
                    gamma=.999, gae_lambda=.99, ent_coef=.003, target_kl=.012)
    settings.update(ppo or {})
    return PPO(AuxLayerNormPolicy, env, seed=seed, device=device, verbose=0, **settings)


def train_stage(model, *, total_timesteps, deadline, callback=None, reset_num_timesteps=False):
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    class Budget(BaseCallback):
        def _on_step(self): return time.monotonic() < deadline
    if time.monotonic() >= deadline: return model
    callbacks = [Budget()] + ([] if callback is None else [callback])
    return model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks),
                       reset_num_timesteps=reset_num_timesteps)


def human_representation_batch(data, indices, *, family):
    """Origin/rotation invariant human signals, with explicit input/target masks.

    These are telemetry prediction targets, never fabricated 2027 actions.
    Absolute XY, gun angle, chassis model, visibility and absent actions are not
    supplied. Unobserved native features have valid=0, unlike measured zeros.
    """
    if family not in ('infantry', 'heavy'): raise ValueError('Human pretraining is auxiliary-only')
    ids = np.asarray(indices, dtype=np.int64); state = data['state']; valid = data['state_valid']
    features = np.zeros((len(ids), len(contract.FEATURES)), np.float32)
    labels = np.zeros((len(ids), 7), np.float32); masks = np.zeros_like(labels)
    lookup = {name: (i, scale) for i, (name, scale) in enumerate(contract.FEATURES)}
    for n, i in enumerate(ids):
        source_role = int(data['role_id'][i])
        if family == 'infantry' and source_role not in (2, 3) or family == 'heavy' and source_role not in (0, 1):
            raise ValueError('Human source role does not match auxiliary family')
        slot = 3+(source_role == 3) if family == 'infantry' else 1
        prefix = f'ally.{slot}.'
        def set_(name, value):
            j, scale = lookup[name]; features[n, j] = float(value)/scale
            features[n, lookup['valid.'+name][0]] = 1
        set_('self_slot', slot); set_('time', data['time_s'][i]); set_(prefix+'present', 1)
        set_(prefix+'alive', data['alive'][i])
        # Source hero/engineer both initialize the 2027 heavy family; no chassis
        # identity, arm stage, weapon installation or legal action is invented.
        for key, column in (('power_w', 4), ('hp', 5), ('max_hp', 6)):
            if valid[i, column]: set_(prefix+key, state[i, column])
        heat_col, limit_col = (9, 10) if source_role == 0 else (7, 8)
        if source_role != 1:
            if valid[i, heat_col]: set_(prefix+'heat', state[i, heat_col])
            if valid[i, limit_col]: set_(prefix+'max_heat', state[i, limit_col])
            if valid[i, heat_col] and valid[i, limit_col] and state[i, limit_col] > 0:
                set_(prefix+'heat_fraction', state[i, heat_col]/state[i, limit_col])
        if valid[i, 5] and valid[i, 6] and state[i, 6] > 0:
            set_(prefix+'hp_fraction', state[i, 5]/state[i, 6])
        if valid[i, 14]: set_('ally.team.resources', state[i, 14])
        previous = i-1
        same_previous = previous >= 0 and data['game_id'][previous] == data['game_id'][i] and data['robot_id'][previous] == data['robot_id'][i] and data['time_s'][i]-data['time_s'][previous] == 1
        if same_previous and data['source_position_valid'][i] and data['source_position_valid'][previous] and not data['discontinuity'][i]:
            speed = np.linalg.norm(data['source_xy'][i]-data['source_xy'][previous])
            set_(prefix+'planar_speed', speed)
        for k, horizon in enumerate((1, 3, 5)):
            if data['future_source_mask'][i, k]:
                speed = np.linalg.norm(data['future_source_xy'][i, k]-data['source_xy'][i])/horizon
                labels[n, k] = np.log1p(speed); masks[n, k] = 1
                j = i+horizon
                if j < len(state) and data['game_id'][j] == data['game_id'][i] and data['robot_id'][j] == data['robot_id'][i] and data['time_s'][j]-data['time_s'][i] == horizon:
                    displacement = np.linalg.norm(data['source_xy'][j]-data['source_xy'][j-1])
                    labels[n, 3+k] = displacement > .2; masks[n, 3+k] = 1
        j = i+5; shots = 12 if source_role == 0 else 11
        if source_role != 1 and j < len(state) and data['future_source_mask'][i, 2] and data['game_id'][j] == data['game_id'][i] and data['robot_id'][j] == data['robot_id'][i] and data['time_s'][j]-data['time_s'][i] == 5 and valid[i, shots] and valid[j, shots]:
            delta = state[j, shots]-state[i, shots]
            if delta >= 0:
                labels[n, 6] = np.log1p(delta/5); masks[n, 6] = .1
    return features, labels, masks


def pretrain_human_representation(model_or_policy, directory, *, family, deadline,
                                  updates=128, batch_size=128, seed=42, learning_rate=1e-4):
    """Train-only offline encoder initialization; prediction head is discarded.

    The returned provenance explicitly counts real human rows/targets. This
    cannot initialize a restricted main actor and never uses semantic XY labels.
    """
    import torch
    from torch import nn
    from .aux_policy import AuxPolicy
    policy = getattr(model_or_policy, 'policy', model_or_policy)
    if not isinstance(policy, AuxPolicy): raise ValueError('2026 omniscient data is auxiliary-only')
    directory = Path(directory); path = directory/'train-trajectories.npz'
    manifest = json.loads((directory/'manifest.json').read_text())
    split = json.loads((directory/'split.json').read_text())
    started = time.monotonic(); rng = np.random.default_rng(seed)
    wanted = (2, 3) if family == 'infantry' else (0, 1) if family == 'heavy' else ()
    if not wanted: raise ValueError('Unknown auxiliary family')
    with np.load(path, allow_pickle=False) as source:
        keys = ('state', 'state_valid', 'role_id', 'time_s', 'game_id', 'robot_id', 'alive',
                'discontinuity', 'source_xy', 'source_position_valid', 'future_source_xy', 'future_source_mask')
        data = {key: source[key] for key in keys}
    game_split = split.get('game_split', {})
    if any(game_split.get(str(int(game))) != 'train' for game in np.unique(data['game_id'])):
        raise ValueError('Human representation input contains a non-training game')
    if manifest.get('split_hash') != split.get('split_hash'):
        raise ValueError('Human split manifest mismatch')
    ids = np.flatnonzero(np.isin(data['role_id'], wanted) & data['alive'] & data['future_source_mask'].any(axis=1))
    prediction = nn.Linear(policy.mean_net.in_features, 7).to(policy.device)
    # Separate temporary optimizer: only the actor encoder and disposable head.
    # Critic, action heads, log_std, main actor and pool weights are untouched.
    params = list(policy.actor_net.parameters())+list(prediction.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    losses = []; counts = np.zeros(7, np.int64); seen = set()
    policy.set_training_mode(True)
    for _ in range(updates):
        if not len(ids) or time.monotonic() >= deadline: break
        selected = rng.choice(ids, min(batch_size, len(ids)), replace=False)
        x, y, mask = human_representation_batch(data, selected, family=family)
        x = torch.as_tensor(x, device=policy.device); y = torch.as_tensor(y, device=policy.device)
        m = torch.as_tensor(mask, device=policy.device)
        output = prediction(policy.actor_net(x))
        per_head = torch.cat((nn.functional.smooth_l1_loss(output[:, :3], y[:, :3], reduction='none'),
            nn.functional.binary_cross_entropy_with_logits(output[:, 3:6], y[:, 3:6], reduction='none'),
            nn.functional.smooth_l1_loss(output[:, 6:], y[:, 6:], reduction='none')), dim=1)
        loss = (per_head*m).sum()/m.sum().clamp_min(1)
        if not torch.isfinite(loss): raise ValueError('Non-finite human prediction loss')
        optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, .5); optimizer.step()
        losses.append(float(loss.detach())); counts += (mask > 0).sum(axis=0); seen.update(map(int, selected))
    return {'source': 'human_2026_train_telemetry', 'family': family, 'available_rows': len(ids),
            'human_rows': len(seen), 'updates': len(losses), 'loss': losses[-1] if losses else None,
            'head_names': ['displacement_speed_1s', 'displacement_speed_3s', 'displacement_speed_5s',
                           'future_moving_1s', 'future_moving_3s', 'future_moving_5s', 'coarse_shot_rate_5s'],
            'supervised_heads': counts.tolist(), 'seconds': time.monotonic()-started,
            'source_sha256': manifest.get('source', {}).get('sha256'),
            'offline_schema_hash': manifest.get('schema_hash'),
            'split_sha256': hashlib.sha256(json.dumps(split, sort_keys=True).encode()).hexdigest(),
            'auxiliary_schema_hash': contract.SCHEMA_HASH, 'absolute_goal_supervised': False,
            'gun_angle_supervised': False, 'original_actions_supervised': False,
            'prediction_head_exported': False}
