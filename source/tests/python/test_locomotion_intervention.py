"""Corrective guidance must preserve tasks, rare pulses and actor ownership."""
import copy
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from rmuc2027_research.rl.campaign import configuration, models
from rmuc2027_research.rl.env import SentryEnv
from rmuc2027_research.rl.guided_campaign import probe_passed, probe_seeds, select_probe
from rmuc2027_research.rl.intervention import data_tensors, teacher_action
from rmuc2027_research.rl import guided_campaign
from rmuc2027_research.rl.learning import experiment


class GuidanceTests(unittest.TestCase):
    def test_repair_probes_keep_actual_ability_and_jump_courses(self):
        for model in (2714,2772,2773):
            config=configuration(model,500)
            config.update(sim_backend='rapier',guided_locomotion=True,alternate_sides=False)
            env=SentryEnv(config)
            try:
                for round_id in range(3):
                    seeds=probe_seeds(models()[model],round_id)
                    kinds=[]
                    for seed in seeds:
                        env.reset(seed=seed)
                        kinds.append(env.session.controller.state['course_kind'])
                    self.assertEqual(kinds,['ability','flat_jump'] if model!=2773 else ['ability'])
            finally:env.close()

    def test_teacher_does_not_mutate_actor_navigation_or_observation(self):
        config=configuration(2773,500)
        config.update(sim_backend='rapier',guided_locomotion=True,alternate_sides=False)
        env=SentryEnv(config)
        try:
            env.reset(seed=47)
            observation=copy.deepcopy(env.session.obs)
            state=copy.deepcopy(env.session.controller.state)
            teacher_action(env.session.obs,env.session.controller.state,config)
            self.assertEqual(env.session.obs,observation)
            self.assertEqual(env.session.controller.state,state)
        finally:env.close()

    def test_rare_takeoff_is_not_outweighed_by_held_jump_labels(self):
        obs=np.zeros((111,152),dtype=np.float32)
        obs[1:101,137]=1.
        targets=np.zeros((111,3),dtype=np.float32)
        targets[:101,2]=1.;targets[101:,2]=-1.
        masks=np.zeros((111,3),dtype=np.float32);masks[:,2]=1.
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.npz'
            np.savez(path,observations=obs,targets=targets,masks=masks,groups=np.full(111,2))
            _,_,weights=data_tensors([path])
        self.assertAlmostEqual(float(weights[0,2]),float(weights[1:101,2].sum()),places=4)
        self.assertAlmostEqual(float(weights[0,2]),float(weights[101:,2].sum()),places=4)
        self.assertEqual(float(weights[:,:2].sum()),0.)

    def test_incomplete_deadline_probe_cannot_pass(self):
        rows=[dict(seed=47,side=side,finished=True,course={'success':True},
                   metrics={'tipped_seconds':0.}) for side in ('red','blue')]
        self.assertTrue(probe_passed(rows,[47]))
        self.assertFalse(probe_passed(rows[:1],[47]))
        self.assertFalse(probe_passed([],[47]))
        self.assertFalse(probe_passed(rows,[47,48]))
        rows[0]['finished']=False
        self.assertFalse(probe_passed(rows,[47]))

    def test_regressed_correction_does_not_replace_better_checkpoint(self):
        def probe(name,success,tips):
            return dict(kind='frozen-probe',checkpoint=name,episodes=[dict(finished=True,
                course={'success':success},metrics={'tipped_seconds':tips,'alive_seconds':10.})])
        good=probe('earlier.zip',True,0.)
        self.assertIs(select_probe([good,probe('later.zip',False,0.)]),good)
        self.assertIs(select_probe([good,probe('unsafe.zip',True,.1)]),good)

    def test_continuation_records_schedule_change_and_keeps_absolute_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            initial=root/'initial';initial.mkdir()
            prepared=root/'prepared';prepared.mkdir()
            warm=root/'warm';warm.mkdir()
            checkpoint=warm/'model.zip';checkpoint.write_bytes(b'unchanged weights')
            config=configuration(4,500);config['curriculum_seconds']=0.
            (warm/'experiment.json').write_text(json.dumps(experiment(config)))
            (initial/'campaign.json').write_text(json.dumps(dict(status='finished',models=[4],results=[])))
            (prepared/'preparation.json').write_text(json.dumps(dict(status='finished',
                results=[dict(model=4,checkpoint=str(checkpoint))])))
            deadline=root/'deadline.json';deadline.write_text('{"deadline_unix":400.0}')
            commands=[]
            def launch(command,**kwargs):
                commands.append(command)
                cfg=json.loads(Path(command[command.index('--config')+1]).read_text())
                resumed=Path(command[command.index('--resume')+1])
                metadata=json.loads((resumed.parent/'experiment.json').read_text())
                self.assertEqual(cfg,metadata['config'])
                self.assertEqual(resumed.read_bytes(),checkpoint.read_bytes())
                self.assertEqual(cfg['curriculum_seconds'],97.5)
                output=Path(command[command.index('--run-dir')+1]);output.mkdir()
                (output/'training.json').write_text('{"actor_updated":true}')
                (output/'model.zip').write_bytes(b'learned')
                return SimpleNamespace(returncode=0,wait=lambda **kw:0)
            with patch.object(guided_campaign.time,'time',return_value=100.), \
                 patch.object(guided_campaign.time,'monotonic',return_value=100.), \
                 patch.object(guided_campaign.subprocess,'Popen',side_effect=launch), \
                 patch.object(guided_campaign,'stop_worker'):
                guided_campaign.continue_training(initial,prepared,root/'out',deadline)
            self.assertEqual(float(commands[0][commands[0].index('--wall-seconds')+1]),195.)


if __name__=='__main__':unittest.main()
