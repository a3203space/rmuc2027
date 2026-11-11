"""Pure CPU/filesystem checks for published rollback accounting and its watcher."""
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest
import zipfile

from rmuc2027_research.rl import v6_longrun as driver

DEADLINE_UNIX=1794373033.0
MEMBERS=('model.zip','resume-state.pt','samplers.json','actor.json','experiment.json','coordinator-state.json')


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')


def generation(root,name,counters=None,copy_model=None):
    folder=root/'main/checkpoints'/name;folder.mkdir(parents=True)
    if copy_model is None:
        with zipfile.ZipFile(folder/'model.zip','w') as archive:archive.writestr('data',json.dumps(counters))
    else:
        shutil.copy2(copy_model,folder/'model.zip')
        with zipfile.ZipFile(copy_model) as archive:counters=json.loads(archive.read('data'))
    for member in MEMBERS[1:]:(folder/member).write_bytes(b'{}\n')
    ready=dict(directory=folder.relative_to(root).as_posix(),samples=counters['num_timesteps'],
        optimizer_steps=counters['optimizer_steps'],update_id=counters['completed_updates'],
        skill_sha256='a'*64,sha256={member:driver.sha(folder/member) for member in MEMBERS})
    write(folder/'complete.json',ready)
    return ready,folder/'model.zip'


def lineage_fixture(parent):
    master=parent/'master';master.mkdir();previous=parent/'previous'
    _,anchor=generation(previous,'baseline',dict(num_timesteps=100,optimizer_steps=20,completed_updates=5))
    rows=[];accepted_model=anchor
    specs=((True,10,8,2,1,5),(False,100,96,12,7,9),(True,8,8,2,1,7))
    for index,(accepted,observed,published,steps,updates,rejected) in enumerate(specs,1):
        root=parent/f'round-{index}';baseline,start=generation(root,'baseline',copy_model=accepted_model)
        with zipfile.ZipFile(start) as archive:data=json.loads(archive.read('data'))
        values=dict(num_timesteps=data['num_timesteps']+published,optimizer_steps=data['optimizer_steps']+steps,
            completed_updates=data['completed_updates']+updates,rejected_actor_steps=rejected)
        ready,candidate=generation(root,f'update-{index}',values)
        write(root/'main/checkpoint-ready.json',ready)
        shutil.copy2(candidate,root/'main/model.zip');shutil.copy2(start,root/'main/initialized.zip')
        rows.append(dict(round=index,path=str(root),accepted_for_continuation=accepted,checkpoint=ready,
            training=dict(actual_samples=observed,published_samples=published,published_accepted_actor_steps=steps,
                published_completed_updates=updates,published_skill_unchanged=True,rejected_actor_steps=rejected)))
        if accepted:accepted_model=candidate
    final=driver.run_directory(master,'final');ready,model=generation(final,'baseline',copy_model=accepted_model)
    write(final/'main/checkpoint-ready.json',ready)
    shutil.copy2(model,final/'main/model.zip');shutil.copy2(anchor,final/'main/initialized.zip')
    write(final/'main/evaluation-anchor.json',dict(source_run=str(previous),source_generation='main/checkpoints/baseline',
        model_sha256=driver.sha(anchor)))
    write(final/'main/training.json',dict(status='accepted_lineage_frozen_for_evaluation',publication_verified=True,
        published_checkpoint_sha256=driver.sha(model),rejected_actor_steps=12,published_samples=16,
        unrelated={'nested':[1,2,3]}))
    write(final/'campaign.json',dict(status='evaluating'))
    write(master/'campaign.json',dict(status='evaluating',active_run=str(final),previous_run=str(previous),rounds=rows))
    write(master/'budget.json',dict(deadline_unix=DEADLINE_UNIX))
    write(master/'authorization.json',dict(deadline_unix=DEADLINE_UNIX))
    write(master/'orchestrator-launch.json',dict(pid=123,source_sha256={'research/rmuc2027_research/rl/v6_longrun.py':'f'*64}))
    return master,final,anchor,rows


class PublishedCounterTests(unittest.TestCase):
    def test_complete_lineage_deltas_replace_model_lifetime_sum(self):
        with TemporaryDirectory() as temp:
            _,final,anchor,rows=lineage_fixture(Path(temp))
            result=driver.final_training_summary(final,rows,anchor)
            self.assertEqual(result['rejected_actor_steps'],7)
            self.assertEqual([r['published_delta'] for r in result['rejected_actor_steps_provenance']['rounds']],[5,2])
            self.assertEqual(result['actual_samples'],18);self.assertEqual(result['observed_samples_all_branches'],118)
            self.assertEqual(result['published_samples'],16)
            # A lazy baseline counter really is absent, rather than a forged zero.
            with zipfile.ZipFile(anchor) as archive:self.assertNotIn('rejected_actor_steps',json.loads(archive.read('data')))
            for row in rows:row['training']['rejected_actor_steps']=99999
            self.assertEqual(driver.final_training_summary(final,rows,anchor)['rejected_actor_steps'],7)

    def test_all_six_generation_hashes_checked(self):
        for member in MEMBERS:
            with self.subTest(member=member),TemporaryDirectory() as temp:
                _,final,anchor,rows=lineage_fixture(Path(temp))
                (Path(rows[0]['path'])/rows[0]['checkpoint']['directory']/member).write_text('changed')
                with self.assertRaisesRegex(ValueError,'SHA mismatch'):driver.final_training_summary(final,rows,anchor)

    def test_zero_accepted_rounds_retain_anchor_with_zero_new_rollbacks(self):
        with TemporaryDirectory() as temp:
            _,final,anchor,rows=lineage_fixture(Path(temp))
            for row in rows:row['accepted_for_continuation']=False
            shutil.rmtree(final/'main/checkpoints/baseline')
            ready,model=generation(final,'baseline',copy_model=anchor)
            write(final/'main/checkpoint-ready.json',ready)
            shutil.copy2(model,final/'main/model.zip')
            result=driver.final_training_summary(final,rows,anchor)
            self.assertEqual(result['rejected_actor_steps'],0)
            self.assertEqual(result['published_samples'],0)
            self.assertEqual(result['rejected_actor_steps_provenance']['rounds'],[])
            self.assertEqual(result['rejected_branch_observed_samples'],118)

    def test_same_counter_different_model_hash_is_not_continuous_lineage(self):
        with TemporaryDirectory() as temp:
            _,final,anchor,rows=lineage_fixture(Path(temp))
            root=Path(rows[2]['path']);folder=root/'main/checkpoints/baseline'
            with zipfile.ZipFile(folder/'model.zip','a') as archive:
                archive.writestr('different-policy.pth',b'different weights; unchanged counters')
            ready=json.loads((folder/'complete.json').read_text())
            ready['sha256']['model.zip']=driver.sha(folder/'model.zip')
            write(folder/'complete.json',ready)
            with self.assertRaisesRegex(ValueError,'lineage is discontinuous'):
                driver.final_training_summary(final,rows,anchor)

    def test_trained_published_generation_must_store_rollback_counter(self):
        with TemporaryDirectory() as temp:
            _,final,anchor,rows=lineage_fixture(Path(temp))
            row=rows[0];folder=Path(row['path'])/row['checkpoint']['directory']
            with zipfile.ZipFile(folder/'model.zip') as archive:data=json.loads(archive.read('data'))
            del data['rejected_actor_steps']
            with zipfile.ZipFile(folder/'model.zip','w') as archive:archive.writestr('data',json.dumps(data))
            row['checkpoint']['sha256']['model.zip']=driver.sha(folder/'model.zip')
            write(folder/'complete.json',row['checkpoint'])
            with self.assertRaisesRegex(ValueError,'lacks rollback counter'):
                driver.final_training_summary(final,rows,anchor)

    def test_rejects_complete_record_or_continuity_mismatch(self):
        with TemporaryDirectory() as temp:
            _,final,anchor,rows=lineage_fixture(Path(temp))
            rows[0]['checkpoint']['samples']+=1
            with self.assertRaisesRegex(ValueError,'completely published'):driver.final_training_summary(final,rows,anchor)
        with TemporaryDirectory() as temp:
            _,final,anchor,rows=lineage_fixture(Path(temp))
            # Replace the final alias with an ancestor while leaving its ready proof intact.
            shutil.copy2(anchor,final/'main/model.zip')
            with self.assertRaises(ValueError):driver.final_training_summary(final,rows,anchor)


if __name__=='__main__':unittest.main()
