"""Frozen actor validation on identical public courses or full matches.

This module never updates weights. Course success is not a match win rate;
snapshot validation on CUDA backends is not deterministic Rapier replay.
"""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO

from .campaign import write_json
from .env import SentryEnv
from .exporting import export_model
from .session import provenance


def replay_evidence(path, backend):
    from rmuc2027_core.replay import ReplayArchive, verify_replay
    if backend == 'rapier':
        return dict(kind='deterministic', **verify_replay(path))
    archive = ReplayArchive(path)
    if archive.metadata.get('physics_backend') != backend:
        raise ValueError('Replay backend mismatch')
    first = archive.frames(0, 1)[0]
    last = archive.frames(len(archive.offsets)-1, 1)[0]
    if last['tick']-first['tick'] != len(archive.offsets)-1:
        raise ValueError('Replay frame count mismatch')
    return dict(kind='snapshot_structure', deterministic=False,
                frames=len(archive.offsets), final_tick=last['tick'],
                sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def frozen_model(path, *, allow_sdk_transfer=False):
    path = Path(path)
    metadata = json.loads((path.parent/'experiment.json').read_text())
    current = provenance(metadata['config'])
    if metadata['config'].get('policy_version') in (3,4):
        for key in ('sentry_schema','sentry_sources'):
            if metadata.get(key)!=current[key]:raise ValueError(f'Frozen policy runtime changed: {key}')
    for key in ('sources', 'sdk_hash', 'runtime_hash', 'engine_binary_sha256',
                'rules_sha256', 'navigation', 'simulation'):
        if key == 'sdk_hash' and allow_sdk_transfer:
            continue
        if current[key] != metadata[key]:
            raise ValueError(f'Frozen policy runtime changed: {key}')
    torch.set_num_threads(1)
    if metadata['config'].get('policy_version')==4:
        from .asymmetric_ppo import AsymmetricPPO
        return AsymmetricPPO.load(path,device='cpu'),metadata
    return PPO.load(path, device='cpu'), metadata


def trial(model, metadata, backend, stage, seed, side, directory, record_replay=False, guided_courses=False):
    config = copy.deepcopy(metadata['config'])
    original_side = config['team']
    if config.get('lineups') and original_side!=side:
        config['lineups']={t:config['lineups']['blue' if t=='red' else 'red'] for t in ('red','blue')}
    config.update(team=side, sim_backend=backend, episode_seconds=420,
                  curriculum_seconds=1 if stage == 'course' else 0,
                  alternate_sides=False, navigation_precompute=True)
    config.pop('_training_deadline', None)
    if guided_courses and stage=='course':config['guided_locomotion']=True
    config['controllers'] = {
        f"{side if key.split(':')[0] == original_side else ('blue' if side == 'red' else 'red')}:{key.split(':')[1]}": value
        for key, value in config['controllers'].items()}
    started = time.perf_counter()
    env = SentryEnv(config, record_replay=record_replay)
    total_reward = 0.
    name = f'{stage}-{seed}-{side}'
    try:
        observation, _ = env.reset(seed=seed)
        with gzip.open(directory/f'{name}-actions.jsonl.gz', 'wt') as stream:
            while True:
                action, _ = model.predict(observation, deterministic=True)
                if not np.isfinite(action).all():
                    raise ValueError('Non-finite frozen actor action')
                observation, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                stream.write(json.dumps(info, allow_nan=False)+'\n')
                if terminated or truncated:
                    break
        result = dict(seed=seed, side=side, stage=stage, backend=backend,
                      terminated=terminated, truncated=truncated,
                      reward=total_reward, final=info,
                      evaluation_sources=provenance(config))
        if stage == 'match':
            result['outcome'] = ('win' if info['winner'] == side else
                                 'draw' if info['winner'] is None else 'loss') if terminated else 'truncated'
        else:
            result['success'] = info['course']['success']
        if record_replay:
            path = directory/f'{name}.json'
            env.session.save_replay(path)
            result['replay'] = str(path)
            result['replay_evidence'] = replay_evidence(path, backend)
        result['wall_seconds'] = time.perf_counter()-started
        write_json(directory/f'{name}-result.json', result)
        return result
    finally:
        env.close()


def validate(path, output, backend, stage, seeds, sides, record_replay=False, allow_sdk_transfer=False, guided_courses=False):
    path, output = Path(path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    model, metadata = frozen_model(path, allow_sdk_transfer=allow_sdk_transfer)
    report = dict(model_sha256=before, backend=backend, stage=stage,
                  deterministic_actions=True, gradient_updates=0, results=[])
    report['training_sdk_hash'] = metadata['sdk_hash']
    report['evaluation_sdk_hash'] = provenance(metadata['config'])['sdk_hash']
    report['sdk_transfer'] = report['training_sdk_hash'] != report['evaluation_sdk_hash']
    report['guided_courses'] = guided_courses or bool(metadata['config'].get('guided_locomotion'))
    started = time.perf_counter()
    for seed in seeds:
        for side in sides:
            row = trial(model, metadata, backend, stage, seed, side, output, record_replay, guided_courses)
            report['results'].append(row)
            report['wall_seconds'] = time.perf_counter()-started
            write_json(output/'validation.json', report)
            print(json.dumps(dict(model=str(path.parent.name), backend=backend,
                                  stage=stage, seed=seed, side=side,
                                  success=row.get('success'), outcome=row.get('outcome'),
                                  wall_seconds=row['wall_seconds'])), flush=True)
    report['weights_unchanged'] = hashlib.sha256(path.read_bytes()).hexdigest() == before
    if not report['weights_unchanged']:
        raise RuntimeError('Checkpoint changed during frozen evaluation')
    write_json(output/'validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--model', type=Path)
    source.add_argument('--campaign', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['rapier', 'warp_kinematic', 'mjwarp'], required=True)
    parser.add_argument('--stage', choices=['course', 'match'], default='course')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--sides', choices=['red', 'blue'], nargs='+', default=['red', 'blue'])
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--watch', action='store_true', help='Evaluate completed checkpoints while a campaign runs')
    parser.add_argument('--export', action='store_true', help='Also export each checkpoint as a portable squad')
    parser.add_argument('--allow-sdk-transfer', action='store_true', help='Explicitly evaluate unchanged weights with a changed SDK; retain both source hashes')
    parser.add_argument('--guided-courses', action='store_true', help='Use the corrected travel/spawn-jump courses and require unfolding after tunnel exit')
    args = parser.parse_args()
    if args.watch and (not args.campaign or args.backend != 'rapier'):
        parser.error('--watch requires a campaign and Rapier to leave the training GPU free')
    if args.model:
        validate(args.model, args.output, args.backend, args.stage, args.seeds, args.sides, args.replay, args.allow_sdk_transfer, args.guided_courses)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    done = set()
    report = dict(backend=args.backend, stage=args.stage, results=[], errors=[])
    while True:
        campaign = json.loads((args.campaign/'campaign.json').read_text())
        for row in campaign['results']:
            name = str(row['model'])
            if name in done or row['status'] != 'trained':
                continue
            try:
                checkpoint = Path(row['checkpoint'])
                if args.export:
                    export_model(checkpoint, args.output/f'{name}-export', allow_sdk_transfer=args.allow_sdk_transfer)
                result = validate(checkpoint, args.output/name, args.backend,
                                  args.stage, args.seeds, args.sides, args.replay, args.allow_sdk_transfer, args.guided_courses)
                report['results'].append(dict(model=row['model'], **result))
            except Exception as error:
                report['errors'].append(dict(model=row['model'], type=type(error).__name__, message=str(error)))
                print(json.dumps(report['errors'][-1]), flush=True)
            done.add(name)
            write_json(args.output/'summary.json', report)
        if not args.watch or campaign['status'] != 'running':
            break
        time.sleep(10)


if __name__ == '__main__':
    main()
