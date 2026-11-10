"""Backfill display-only replay scenario IDs after campaign catalog publication.

No environment/model imports. The live scenario/capability registry and original
replay bytes are never modified. This is safe for the running directory service:
it reads run-manifest.json afresh on each catalog request.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .scenarios import ScenarioSpec
from .v6_web import resource


def metadata_header(path, max_characters=8*1024*1024):
    """Read the canonical metadata header without decoding any frame array."""
    decoder=json.JSONDecoder()
    with Path(path).open(encoding='utf-8') as stream:
        buffer='';position=0
        def fill():
            nonlocal buffer
            chunk=stream.read(min(65536,max_characters-len(buffer)))
            if not chunk:raise ValueError('Replay metadata is missing, incomplete or exceeds header limit')
            buffer+=chunk
        def whitespace():
            nonlocal position
            while True:
                while position<len(buffer) and buffer[position].isspace():position+=1
                if position<len(buffer):return
                fill()
        def punctuation(value):
            nonlocal position
            whitespace()
            if buffer[position]!=value:raise ValueError('Noncanonical replay header')
            position+=1
        def item():
            nonlocal position
            whitespace()
            while True:
                try:value,position=decoder.raw_decode(buffer,position);return value
                except json.JSONDecodeError:fill()
        punctuation('{')
        found_format=False
        while True:
            key=item();punctuation(':')
            if key=='frames':raise ValueError('Replay metadata must precede frames')
            value=item()
            if key=='format':
                if value not in ('rmuc-replay-v3','rmuc-replay-v4'):raise ValueError('Unsupported canonical replay format')
                found_format=True
            if key=='metadata':
                if not found_format or not isinstance(value,dict):raise ValueError('Invalid replay metadata')
                return value
            punctuation(',')


def recorded_scenario(metadata):
    """Recover exact hardware labels from the recorded side/slot assignment."""
    config=metadata.get('rl',{}).get('config',{})
    explicit=config.get('scenario')
    if explicit is not None:return ScenarioSpec.from_dict(explicit).scenario_id
    side=config.get('team')
    if side not in ('red','blue'):raise ValueError('Recorded learner side is unavailable')
    other='blue' if side=='red' else 'red'
    lineups=config.get('lineups',{})
    try:
        friendly,enemy=lineups[side],lineups[other]
        if friendly['7']!=2773:raise ValueError('Recorded main is not the protected 2773')
        case=ScenarioSpec((friendly['3'],friendly['4']),(enemy['3'],enemy['4']),enemy['7'],side)
    except (KeyError,TypeError) as error:
        raise ValueError('Recorded canonical side lineups are unavailable') from error
    return case.scenario_id


def enrich_replay_labels(run):
    root=Path(run).resolve()
    campaign=json.loads((root/'campaign.json').read_text())
    if campaign.get('status')!='web_acceptance':
        raise ValueError('Backfill runs only after the campaign publishes its web_acceptance catalog')
    path=root/'run-manifest.json';before=path.read_bytes();manifest=json.loads(before)
    original_registry=copy.deepcopy({key:manifest[key] for key in ('scenarios','capability_profiles','checkpoints','helpers')})
    changed=[];skipped=[]
    for replay_id,row in manifest.get('replays',{}).items():
        if row.get('scenario_id'):continue
        try:
            replay=resource(root,row)
            metadata=metadata_header(replay)
            row['scenario_id']=recorded_scenario(metadata)
            row['scenario_label_source']='recorded_runtime_side_lineups'
            changed.append(dict(replay_id=replay_id,scenario_id=row['scenario_id']))
        except (ValueError,OSError,TypeError,KeyError) as error:
            skipped.append(dict(replay_id=replay_id,reason=str(error)))
    assert original_registry=={key:manifest[key] for key in original_registry}
    report=dict(format='rmuc-replay-label-backfill-v1',display_only=True,
        original_manifest_sha256=hashlib.sha256(before).hexdigest(),updated=changed,skipped=skipped,
        live_registry_changed=False,replay_bytes_changed=False)
    if changed:
        if path.read_bytes()!=before:raise RuntimeError('Manifest changed concurrently; rerun after publication completes')
        pending=path.with_suffix('.labels.pending')
        pending.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n');pending.replace(path)
    report['manifest_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    report_path=root/'replay-label-backfill.json'
    pending=report_path.with_suffix('.pending');pending.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');pending.replace(report_path)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    print(json.dumps(enrich_replay_labels(parser.parse_args().run),ensure_ascii=False))


if __name__=='__main__':main()
