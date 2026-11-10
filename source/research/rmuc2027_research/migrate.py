"""Explicit model migration, preserving weights and original runtime provenance."""
import hashlib, json, shutil
from pathlib import Path
from rmuc2027_core.paths import ROOT

def migrate_model(source, output):
    from stable_baselines3 import PPO
    from .rl.learning import experiment
    from .rl.portable import OBS_SIZE, OBS_VERSION
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('Choose a new output directory')
    old = json.loads((source.parent/'experiment.json').read_text())
    known = json.loads((ROOT/'configs/migrations/layout-v1.json').read_text())
    keys = ('sdk_hash','runtime_hash','engine_binary_sha256','rules_sha256')
    if {key:old.get(key) for key in keys} not in known['source_runtimes']:
        raise ValueError('Unrecognized source runtime; manual compatibility audit required')
    if old.get('obs_version') != OBS_VERSION or old.get('obs_size') != OBS_SIZE:
        raise ValueError('Actor input contract changed')
    current = experiment(old['config'])
    if old['sources'] != current['sources']: raise ValueError('Fixed policies changed')
    model = PPO.load(source, device='cpu')
    if model.observation_space.shape != (OBS_SIZE,) or model.action_space.shape != (2,):
        raise ValueError('Unexpected model spaces')
    import numpy as np
    prediction, _ = model.predict(np.zeros(OBS_SIZE,dtype=np.float32),deterministic=True)
    if not np.isfinite(prediction).all(): raise ValueError('Model produced a non-finite action')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    output.mkdir(parents=True)
    shutil.copy2(source, output/'model.zip')
    current['migration'] = {'id':known['migration'],'source_model_sha256':source_hash,
        'original_experiment':old,'note':'Weights and optimizer preserved. Runtime is now the 2027-only engine; old engine reproducibility is not claimed.'}
    (output/'experiment.json').write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
    assert hashlib.sha256((output/'model.zip').read_bytes()).hexdigest() == source_hash
    return {'model':str(output/'model.zip'),'sha256':source_hash,'obs_size':OBS_SIZE}
