"""Select frozen candidates on matched courses, retaining every evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .campaign import write_json
from .exporting import export_model
from .validation import validate
from .intervention import correct
from .session import provenance


def score(reports):
    rows=[row for report in reports for row in report['results']]
    if not rows:raise ValueError('Cannot select an unevaluated checkpoint')
    return (sum(row['success'] and row['final']['metrics']['tipped_seconds']==0 for row in rows),
            -sum(row['final']['metrics']['tipped_seconds'] for row in rows),
            -sum(row['final']['metrics']['alive_seconds'] for row in rows))


def evaluate_candidate(checkpoint,directory,backend):
    checkpoint,directory=Path(checkpoint),Path(directory)
    path=directory/'validation.json'
    if path.exists():
        cached=json.loads(path.read_text())
        metadata=json.loads((checkpoint.parent/'experiment.json').read_text())
        pairs={(r['seed'],r['side']) for r in cached.get('results',[])}
        expected={(seed,side) for seed in (42,43,44) for side in ('red','blue')}
        if (cached.get('weights_unchanged') and cached.get('gradient_updates')==0
                and cached.get('model_sha256')==hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                and cached.get('backend')==backend and cached.get('stage')=='course'
                and cached.get('guided_courses') and pairs==expected and len(cached['results'])==6
                and cached.get('evaluation_sdk_hash')==provenance(metadata['config'])['sdk_hash']):
            return cached
    if directory.exists():
        directory.rename(directory.with_name(directory.name+f'-incomplete-{time.time_ns()}'))
    return validate(checkpoint,directory,backend,'course',[42,43,44],['red','blue'],
                    allow_sdk_transfer=True,guided_courses=True)


def finish_gpu(result,prepared,output):
    if result.get('export'):return
    model=result['model'];folder=output/str(model)
    candidates=result['candidates']
    latest=next(r for r in json.loads((prepared/'preparation.json').read_text())['results'] if r['model']==model)
    if latest['checkpoint'] not in {c['checkpoint'] for c in candidates}:
        validation=evaluate_candidate(latest['checkpoint'],folder/'updated-reference-rapier','rapier')
        candidates.append(dict(kind='updated-reference',checkpoint=latest['checkpoint'],evaluations=[validation]))
    preferred=max(candidates,key=lambda c:score([v for v in c['evaluations'] if v['backend']=='rapier']))
    order=[preferred]+[c for c in candidates if c is not preferred]
    evaluated=[]
    for candidate in order:
        # Each remaining backend has six trials. A candidate that cannot
        # match the best stable-success count need not consume GPU time.
        upper_bound=score([v for v in candidate['evaluations'] if v['backend']=='rapier'])[0]+12
        if evaluated and upper_bound<max(score(c['evaluations'])[0] for c in evaluated):
            result.setdefault('pruned_candidates',[]).append(dict(kind=candidate['kind'],
                maximum_stable_success=upper_bound,reason='Cannot match an already verified candidate'))
            continue
        try:
            for backend in ('warp_kinematic','mjwarp'):
                validation=evaluate_candidate(candidate['checkpoint'],folder/f"{candidate['kind']}-{backend}",backend)
                candidate['evaluations']=[v for v in candidate['evaluations'] if v['backend']!=backend]+[validation]
            evaluated.append(candidate)
            if score(candidate['evaluations'])[0]==18:break
        except Exception as error:
            result.setdefault('candidate_errors',[]).append(dict(kind=candidate['kind'],error=f'{type(error).__name__}: {error}'))
    if not evaluated:raise RuntimeError('No candidate completed all three backend evaluations')
    chosen=max(evaluated,key=lambda c:score(c['evaluations']))
    result.update(checkpoint=chosen['checkpoint'],kind=chosen['kind'],combined_score=score(chosen['evaluations']))
    export_model(chosen['checkpoint'],folder/'export',allow_sdk_transfer=True)
    result['export']=str(folder/'export')
    write_json(folder/'selection.json',result)
    print(json.dumps(dict(model=model,export=result['export'],combined_score=result['combined_score'])),flush=True)


def select(campaign,prepared,output,deadline_file=None,resume=False):
    campaign,prepared,output=Path(campaign),Path(prepared),Path(output)
    output.mkdir(parents=True,exist_ok=resume)
    report=dict(status='watching',gradient_updates=0,results=[],errors=[],
                seeds=[42,43,44],sides=['red','blue'],guided_courses=True)
    if resume:report=json.loads((output/'selection.json').read_text())
    report['status']='watching'
    seen={r['model'] for r in report['results']};write_json(output/'selection.json',report)
    training_deadline=(json.loads(Path(deadline_file).read_text())['deadline_unix']
                       if deadline_file is not None else 0.)
    while True:
        progress=json.loads((campaign/'campaign.json').read_text())
        warm={row['model']:row for row in json.loads((prepared/'preparation.json').read_text())['results']}
        for row in progress['results']:
            model=row['model']
            if model in seen:continue
            seen.add(model)
            try:
                folder=output/str(model);folder.mkdir(exist_ok=resume)
                candidates=[]
                # PPO is preferred on an exact tie, but never on a worse score.
                reference_kind=warm[model].get('checkpoint_role','guided')
                for kind,checkpoint in [('ppo',row.get('checkpoint')),(reference_kind,warm[model]['checkpoint'])]:
                    if checkpoint is None:continue
                    results=[evaluate_candidate(checkpoint,folder/f'{kind}-rapier','rapier')]
                    candidates.append(dict(kind=kind,checkpoint=checkpoint,evaluations=results))
                if len(candidates)==2 and score(candidates[0]['evaluations'])[0]<score(candidates[1]['evaluations'])[0]:
                    metadata=json.loads((Path(candidates[0]['checkpoint']).parent/'experiment.json').read_text())
                    paths=metadata['config'].get('skill_guidance_datasets',[])
                    remaining=training_deadline-time.time()-90.
                    if paths and remaining>20.:
                        corrected=folder/'restored'
                        if (corrected/'correction.json').exists():
                            fit=json.loads((corrected/'correction.json').read_text())
                        else:
                            fit=correct(candidates[0]['checkpoint'],corrected,paths,
                                        max_seconds=min(15.,remaining),fit_steps=4000,fit_learning_rate=.0003)
                        result=evaluate_candidate(corrected/'model.zip',folder/'restored-rapier','rapier')
                        candidates.append(dict(kind='restored',checkpoint=str(corrected/'model.zip'),
                                               evaluations=[result],correction=fit))
                        report['gradient_updates']+=fit['steps']
                chosen=max(candidates,key=lambda c:score(c['evaluations']))
                result=dict(model=model,checkpoint=chosen['checkpoint'],kind=chosen['kind'],
                            rapier_score=score(chosen['evaluations']),candidates=candidates)
                report['results'].append(result)
                write_json(folder/'selection.json',result)
                print(json.dumps(dict(model=model,selected=chosen['kind'],score=result['rapier_score'])),flush=True)
            except Exception as error:
                report['errors'].append(dict(model=model,error=f'{type(error).__name__}: {error}'))
                print(json.dumps(report['errors'][-1]),flush=True)
            write_json(output/'selection.json',report)
        # Evaluate completed actors while the next chassis trains. The GPU
        # comparisons never update weights or extend the training deadline.
        for result in sorted(report['results'],key=lambda r:(r['model']!=2773,r['model'])):
            if result.get('export'):continue
            try:finish_gpu(result,prepared,output)
            except Exception as error:
                report['errors'].append(dict(model=result['model'],error=f'{type(error).__name__}: {error}'))
            write_json(output/'selection.json',report)
        if progress['status']=='finished':break
        if progress['status'] not in ('waiting','running'):
            raise RuntimeError('Training stopped unexpectedly; selection remains incomplete')
        time.sleep(10)
    report.update(status='finished',all_models_evaluated=(not report['errors'] and
                  set(seen)==set(progress['models']) and len(report['results'])==len(progress['models'])
                  and all(r.get('export') for r in report['results'])))
    write_json(output/'selection.json',report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deadline-file',type=Path,help='Allow corrective fitting before this already-approved deadline only')
    parser.add_argument('--resume',action='store_true',help='Continue evaluation, reusing only complete matching frozen reports')
    args=parser.parse_args()
    select(args.campaign,args.prepared,args.output,args.deadline_file,args.resume)


if __name__=='__main__':main()
