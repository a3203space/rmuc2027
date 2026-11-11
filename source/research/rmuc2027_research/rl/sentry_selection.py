"""Compare immutable v3 candidates on paired short, public full-arena trials.

Selection is local to these probes, never a claim about full-match win rates.
Training artifacts and original checkpoints are retained unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

from .campaign import write_json
from .sentry_campaign import evaluate_worker


def select(output, model_id, candidates, deadline_unix):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    reports=[]
    for name,checkpoint in candidates.items():
        checkpoint=Path(checkpoint).resolve();trials=[]
        checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        metadata_path=checkpoint.parent/'experiment.json'
        expected_sdk=json.loads(metadata_path.read_text()).get('sdk_hash') if metadata_path.exists() else None
        for side,seed in (('red',1001),('blue',1002)):
            folder=output/'trials'/name/side
            result=folder/'result.json'
            if not result.exists():
                if time.time()>=deadline_unix-25:break
                evaluate_worker(checkpoint,folder,'warp_kinematic',side,seed,min(deadline_unix-15,time.time()+130),60)
            trial=json.loads(result.read_text())
            if Path(trial['checkpoint']).resolve()!=checkpoint:raise ValueError('Cached selection checkpoint mismatch')
            if expected_sdk is not None and (trial.get('evaluation_sdk_hash')!=expected_sdk
                    or trial.get('checkpoint_sha256')!=checkpoint_hash):
                raise ValueError('Cached selection runtime or checkpoint bytes changed; rerun frozen trials')
            trials.append(trial)
        valid=(len(trials)==2 and all(t['horizon_completed'] and not t['policy_errors'] and
            t['tipped_seconds']==0 and t['tunnel_spin_violations']==0 for t in trials))
        captures=[t['final']['metrics']['capture_seconds'] for t in trials]
        gaps=[t['final']['metrics'].get('objective_gap_m',0.) for t in trials]
        # Capture dominates. When no candidate captures, prefer getting closer
        # to the real card over an immobile robot's artificially short path.
        score=(min(captures),sum(captures)/2,-max(gaps),-sum(t['path_m'] for t in trials)) if valid else None
        reports.append(dict(name=name,checkpoint=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            qualified_short_trials=valid,score=score,trials=[dict(side=t['side'],capture_seconds=t['final']['metrics']['capture_seconds'],
                simulation_seconds=t['simulation_seconds'],tipped_seconds=t['tipped_seconds'],policy_errors=t['policy_errors'],
                path_m=t['path_m'],objective_gap_m=t['final']['metrics'].get('objective_gap_m'),
                tunnel_spin_violations=t['tunnel_spin_violations']) for t in trials]))
        write_json(output/'candidates.json',reports)
    accepted=[r for r in reports if r['qualified_short_trials']]
    if not accepted:
        write_json(output/'selection.json',dict(model=model_id,selected=None,reason='No candidate completed both safety-gated short trials',candidates=reports))
        return None
    best=max(accepted,key=lambda r:r['score']);source=Path(best['checkpoint'])
    shutil.copy2(source,output/'model.zip')
    shutil.copy2(source.parent/'experiment.json',output/'experiment.json')
    record=dict(model=model_id,selected=best['name'],source_checkpoint=best['checkpoint'],sha256=best['sha256'],
        criterion='two fixed 60 s Warp trials: no errors/tips/tunnel-spin violations; max worst-side capture, then mean capture, then smaller worst-side card gap, then shorter path',
        full_match_strength_qualified=False,candidates=reports)
    write_json(output/'selection.json',record)
    print(json.dumps(dict(model=model_id,selected=best['name'],score=best['score'])),flush=True)
    return record


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--model',required=True,type=int)
    p.add_argument('--candidates',required=True,help='JSON object mapping candidate labels to checkpoints')
    p.add_argument('--deadline-unix',required=True,type=float);a=p.parse_args()
    select(a.output,a.model,json.loads(Path(a.candidates).read_text()),a.deadline_unix)


if __name__=='__main__':main()
