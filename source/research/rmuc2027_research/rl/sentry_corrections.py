"""Bounded public-observation corrections on learner-induced trajectories.

The expert supervises the six actor outputs during collection only. Frozen
evaluation and export use the actor without an expert or destination override.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .campaign import write_json
from .checkpoints import save_checkpoint
from .env import SentryEnv
from .learning import TimedPPO
from .sentry_training import initialize_actor, teacher_action


def correct(checkpoint, output, deadline_unix):
    import torch
    torch.set_num_threads(1)
    checkpoint, output = Path(checkpoint), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads((checkpoint.parent/'experiment.json').read_text())
    metadata['correction_training'] = dict(
        parent=str(checkpoint), parent_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        method='public teacher labels on mixed learner/teacher destinations; supervised skill execution',
        training_only=True, deadline_unix=deadline_unix,
        teacher_source_sha256=hashlib.sha256(Path(__file__).with_name('sentry_training.py').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(output/'experiment.json', metadata)
    model = TimedPPO.load(checkpoint, device='cpu')
    with np.load(checkpoint.parent/'guidance/demonstrations.npz') as data:
        observations = list(data['observations']); targets = list(data['targets'])
    start = time.time(); reports = []
    for iteration, teacher_fraction in enumerate((.8, .5, .2)):
        if time.time() >= deadline_unix-30: break
        folder = output/f'iteration-{iteration+1}'; folder.mkdir()
        write_json(folder/'experiment.json', metadata)
        episodes = []
        for side, seed in (('red', 1201+2*iteration), ('blue', 1202+2*iteration)):
            if time.time() >= deadline_unix-30: break
            cfg = copy.deepcopy(metadata['config'])
            cfg.update(team=side, alternate_sides=False, episode_seconds=45,
                       navigation_precompute=False, controllers={side+':7':dict(kind='learning')})
            env = SentryEnv(cfg)
            try:
                features, info = env.reset(seed=seed)
                while time.time() < deadline_unix-25:
                    expert = teacher_action(env.session.obs, env.session.controller.state)
                    predicted, _ = model.predict(features, deterministic=True)
                    observations.append(features.copy()); targets.append(expert.copy())
                    mixed = expert.copy()
                    mixed[:2] = teacher_fraction*expert[:2]+(1-teacher_fraction)*predicted[:2]
                    features, _, done, truncated, info = env.step(mixed)
                    if done or truncated: break
                episodes.append(dict(side=side, seed=seed, simulation_seconds=info['time'], metrics=info['metrics']))
            finally: env.close()
        data_path = folder/'demonstrations.npz'
        np.savez_compressed(data_path, observations=np.asarray(observations), targets=np.asarray(targets))
        result = initialize_actor(model, data_path, time.monotonic()+max(0., min(25.,deadline_unix-time.time()-5)), epochs=600)
        save_checkpoint(model, folder/'model.zip')
        result.update(iteration=iteration+1, teacher_destination_fraction=teacher_fraction, episodes=episodes,
                      dataset_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(), checkpoint=str(folder/'model.zip'))
        reports.append(result); write_json(folder/'training.json',result)
        write_json(output/'training.json',dict(method='supervised correction, no PPO samples added',
            iterations=reports, wall_seconds=time.time()-start, deadline_unix=deadline_unix, finished_unix=time.time()))
        print(json.dumps(result), flush=True)
    return reports


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--deadline-unix',required=True,type=float);a=p.parse_args()
    correct(a.checkpoint,a.output,a.deadline_unix)
