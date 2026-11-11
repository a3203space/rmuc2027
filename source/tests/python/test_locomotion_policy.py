"""Policy ownership, public features and bounded all-model training contracts."""
import copy
import importlib.util
import math
import unittest
from unittest.mock import patch
from rmuc2027_core import sdk
from rmuc2027_research.rl.portable import policy_schema, select_goal, encode_observation
from rmuc2027_research.rl.campaign import allocations, configuration, models, weight


def robot(model=2731, x=3., y=7.):
    d=sdk.DESIGNS[str(model)]
    return dict(id='red-3',model_slot=model,slot=3,role=d['role'],team='red',alive=True,powered=True,
        x=x,y=y,z=0.,vx=0.,vy=0.,vz=0.,heading=0.,grounded=True,upright=1.,radius=.28,
        transform_progress=0.,speed_reference_mps=d.get('speed_reference_mps',3.))


def observation(r, now=1.):
    return dict(time=now,arena=sdk._REFERENCE_ARENA,navigation_mode='adaptive',policy_dt=.1,
                team='red',robots=[r],enemies=[],controlled_ids=['3'])


class SkillTests(unittest.TestCase):
    def test_policy_jump_is_edge_triggered_and_never_automatically_reissued(self):
        r=robot();obs=observation(r);state={'stable_since':0.}
        intent=dict(transform=0.,leg_height=-1.,jump=1.)
        a=sdk.navigate_action(r,(4.,7.),obs,state,locomotion_intent=intent)
        self.assertEqual(a['jump'],1.)
        obs['time']=3.
        a=sdk.navigate_action(r,(4.,7.),obs,state,locomotion_intent=intent)
        self.assertEqual(a['jump'],0.)
        sdk.navigate_action(r,(4.,7.),obs,state,locomotion_intent={**intent,'jump':-1.})
        obs['time']=3.1
        self.assertEqual(sdk.navigate_action(r,(4.,7.),obs,state,locomotion_intent=intent)['jump'],1.)

    def test_no_jump_capability_and_no_ground_contact_reject_request(self):
        for model,grounded in ((303,True),(2731,False)):
            r=robot(model);r['grounded']=grounded
            state={'stable_since':0.}
            a=sdk.navigate_action(r,(4.,7.),observation(r),state,
                locomotion_intent=dict(transform=1.,leg_height=1.,jump=1.))
            self.assertEqual(a['jump'],0.)
            self.assertIn('jump_unavailable',state['rejected'])
            if model==303:
                self.assertNotIn('leg_height',a)
                self.assertNotIn('transform',a)

    def test_rotated_ceiling_blocks_unfold_and_reports_guard(self):
        r=robot(2732,9.32,12.735);r.update(z=.16,transform_progress=1.)
        state={}
        a=sdk.navigate_action(r,(9.32,14.),observation(r),state,
            locomotion_intent=dict(transform=-1.,leg_height=0.,jump=-1.))
        self.assertTrue(a['transform'])
        self.assertIn('ceiling',state['rejected'])

    def test_fold_hysteresis_and_death_clear_private_state(self):
        r=robot(2732);obs=observation(r);state={}
        for value,expected in ((1.,True),(0.,True),(-1.,False),(0.,False)):
            a=sdk.navigate_action(r,(4.,7.),obs,state,
                locomotion_intent=dict(transform=value,leg_height=0.,jump=-1.))
            self.assertEqual(a['transform'],expected)
            obs['time']+=1.
        r['alive']=False
        sdk.navigate_action(r,(4.,7.),obs,state,locomotion_intent={})
        self.assertEqual(state,{})

    def test_transform_request_cannot_reverse_an_unfinished_transition(self):
        r=robot(2732);obs=observation(r);state={}
        intent=dict(transform=1.,leg_height=0.,jump=-1.)
        sdk.navigate_action(r,(4.,7.),obs,state,locomotion_intent=intent)
        obs['time']+=.1
        result=sdk.navigate_action(r,(4.,7.),obs,state,locomotion_intent={**intent,'transform':-1.})
        self.assertTrue(result['transform'])
        self.assertIn('transform_dwell',state['rejected'])
        obs['time']+=1.
        self.assertFalse(sdk.navigate_action(r,(4.,7.),obs,state,locomotion_intent={**intent,'transform':-1.})['transform'])

    def test_goal_rate_and_skill_rate_are_independent(self):
        from rmuc2027_research.rl.portable import SdkNavigator
        r=robot();obs=observation(r);nav=SdkNavigator();state={};cfg=configuration(2731,500)
        with patch.object(nav,'resolve_goal',side_effect=lambda r,p,a:p):
            select_goal(obs,state,[0.,0.,1.,-1.,-1.],nav,cfg)
            first=state['goal'][:]
            obs['time']+=.1
            select_goal(obs,state,[1.,1.,-1.,1.,1.],nav,cfg)
            self.assertEqual(state['goal'],first)
            self.assertEqual(state['locomotion_intent']['jump'],1.)
            obs['time']+=.4
            select_goal(obs,state,[1.,1.,-1.,1.,-1.],nav,cfg)
            self.assertNotEqual(state['goal'],first)


class CampaignTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3'),'Optional PPO runtime unavailable')
    def test_v2_actor_exports_and_runs_both_sides_in_policy_worker(self):
        import json,tempfile
        from pathlib import Path
        from stable_baselines3 import PPO
        from rmuc2027_research.rl.env import SentryEnv
        from rmuc2027_research.rl.learning import experiment
        from rmuc2027_research.rl.exporting import export_model
        from rmuc2027_research.rl.session import resource
        from rmuc2027_research.strategies.battle import Battle
        cfg=configuration(2731,500)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'run';run.mkdir()
            env=SentryEnv(cfg)
            try:
                actor=PPO('MlpPolicy',env,n_steps=8,batch_size=8,device='cpu',seed=42)
                actor.save(run/'model.zip')
            finally:env.close()
            (run/'experiment.json').write_text(json.dumps(experiment(cfg)))
            exported=root/'policy';meta=export_model(run/'model.zip',exported)
            self.assertEqual(meta['obs_size'],152)
            self.assertEqual(meta['action_size'],5)
            self.assertLess(meta['actor_parity_max_error'],1e-5)
            for side in ('red','blue'):
                sources=(exported,resource(cfg['opponent'])) if side=='red' else (resource(cfg['opponent']),exported)
                with Battle(*sources,duration=1,sandbox='trusted-local',timeout=5,navigation_mode='adaptive') as battle:
                    self.assertEqual(battle.run()['policy_errors'],0)

    def test_process_group_stop_closes_owned_worker(self):
        import subprocess,sys,time
        from rmuc2027_research.rl.campaign import stop_worker
        worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True,
                                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        started=time.monotonic()
        try:
            stop_worker(worker,.2)
            self.assertIsNotNone(worker.poll())
            self.assertLess(time.monotonic()-started,1.)
        finally:stop_worker(worker,0.)

    def test_real_public_observations_encode_every_chassis_without_hidden_state(self):
        from rmuc2027_core.engine import Match
        for model in models():
            cfg=configuration(model,500)
            with Match(lineups={t:[dict(slot=int(s),model=m) for s,m in row.items()]
                               for t,row in cfg['lineups'].items()}) as match:
                obs=match.observation('red',[cfg['slot']])
                a=encode_observation(obs,{},cfg['slot'],.1,cfg)
                obs['referee']={'reward':float('nan')}
                self.assertEqual(a,encode_observation(obs,{},cfg['slot'],.1,cfg))
                self.assertEqual(len(a),152)
                self.assertTrue(all(math.isfinite(v) and -1<=v<=1 for v in a))

    def test_global_budget_counts_all_models_and_reserves_cleanup(self):
        catalog=models();budgets=allocations(list(catalog),7200)
        self.assertEqual(len(budgets),17)
        self.assertAlmostEqual(sum(budgets.values()),7080.)
        self.assertAlmostEqual(budgets[2731],2*budgets[303])
        for model in catalog:
            cfg=configuration(model,budgets[model])
            self.assertEqual(cfg['lineups']['red'][str(cfg['slot'])],model)
            self.assertEqual(policy_schema(cfg),dict(version=2,obs_size=152,action_size=5))
            self.assertIsNone(cfg['github_backup_repo'])

    def test_invalid_budget_and_duplicate_models_rejected(self):
        for args in (([2731],7201),([2731],120),([2731,2731],7200),([],7200)):
            with self.assertRaises(ValueError):allocations(*args)
        self.assertEqual(policy_schema(),dict(version=1,obs_size=116,action_size=2))


if __name__=='__main__':unittest.main()
