import copy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from rmuc2027_core.engine import Match
from rmuc2027_core import sdk
from rmuc2027_research.rl import sentry_v3
from rmuc2027_research.rl.portable import SdkNavigator, select_goal, execute_goal, encode_observation, own_robot
from rmuc2027_research.rl.session import ArenaSession,load_config


class SentryV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Match(observation_profile=sentry_v3.PROFILE) as m:cls.obs=m.observation('red',[7])
        cls.config=load_config(dict(policy_version=3,navigation_mode='adaptive',rl_seconds=.1,navigation_precompute=False))

    def test_named_schema_complete_finite_unclipped_and_no_privileged_labels(self):
        obs=copy.deepcopy(self.obs);state={}
        self.assertEqual(len(sentry_v3.FEATURE_INDEX),len(sentry_v3.FEATURES))
        values=encode_observation(obs,state,config=self.config)
        self.assertEqual(len(values),sentry_v3.schema()['obs_size'])
        self.assertTrue(all(math.isfinite(v) for v in values))
        obs['robots'][0]['buffer_energy_j']=60
        obs['robots'][0]['energy']=0
        obs['robots'][0]['ammo']=600
        a=sentry_v3.encode(obs,state)
        obs['robots'][0]['ammo']=700
        b=sentry_v3.encode(obs,state)
        self.assertEqual(a[sentry_v3.FEATURE_INDEX['self.buffer_j']],1)
        self.assertNotEqual(a[sentry_v3.FEATURE_INDEX['self.ammo']],b[sentry_v3.FEATURE_INDEX['self.ammo']])
        for forbidden in ('respawn_progress','capture_progress','enemy.max_hp','course_kind'):
            self.assertFalse(any(forbidden in name for name in sentry_v3.FEATURE_INDEX))
        state['course_kind']='secret';self.assertEqual(b,sentry_v3.encode(obs,state))

    def test_goal_remains_rl_owned_during_low_health_and_decodes_six_actions(self):
        obs=copy.deepcopy(self.obs);obs['robots'][0].update(hp=1,weak=True)
        nav=SdkNavigator();state={}
        with patch.object(nav,'resolve_goal',side_effect=lambda r,p,a:p):
            select_goal(obs,state,[-.5,0,1,1,-1,-1],nav,self.config)
        desired=list(state['goal'])
        result=execute_goal(obs,state,nav,self.config,None)
        self.assertEqual(state['executed_goal'],desired)
        self.assertNotIn('leg_height',state['sentry_intent'])
        self.assertFalse(result['7']['fire'])
        self.assertIn('fold_unsupported',state['execution_trace']['rejections'])
        with self.assertRaises(ValueError):select_goal(obs,state,[0]*5,nav,self.config)

    def test_fire_decision_is_independent_of_autoaim(self):
        obs=copy.deepcopy(self.obs);obs['robots'][0].update(ammo=100,heat=0,max_heat=500,powered=True)
        obs['autoaim']=dict(valid=True,aligned=True,aim=0,pitch=0,distance=2)
        state=dict(goal=[3,7],sentry_intent=dict(fold=-1,jump=-1,spin=-1,fire=-1))
        self.assertFalse(sentry_v3.execute(obs,state,SdkNavigator(),self.config)['7']['fire'])
        state['sentry_intent']['fire']=1
        self.assertTrue(sentry_v3.execute(obs,state,SdkNavigator(),self.config)['7']['fire'])
        obs['autoaim']['aligned']=False
        self.assertFalse(sentry_v3.execute(obs,state,SdkNavigator(),self.config)['7']['fire'])

    def test_tunnel_spin_guard_requires_actual_stop_and_heading_then_safe_exit(self):
        obs=copy.deepcopy(self.obs);r=obs['robots'][0]
        roof=dict(id='test_tunnel_roof',x=5,y=5,w=2,h=1,bottom=.3,height=.2,rotation=0)
        obs['arena']=dict(obs['arena'],obstacles=[roof])
        r.update(x=4.5,y=5.5,vx=1,vy=0,heading=0,angular_velocity=[0,0,5])
        nav={};fields=dict(move=[1,0],heading=0)
        self.assertTrue(sentry_v3.spin_guard(r,obs,nav,fields,True))
        self.assertEqual(fields['move'],(0.,0.));self.assertNotIn('yaw_rate',fields)
        r.update(angular_velocity=[0,0,0],vx=0,heading=.2)
        fields=dict(move=[1,0],heading=0)
        sentry_v3.spin_guard(r,obs,nav,fields,True);self.assertEqual(fields['move'],(0.,0.))
        r['heading']=0;fields=dict(move=[1,0],heading=0)
        sentry_v3.spin_guard(r,obs,nav,fields,True);self.assertEqual(fields['move'],[1,0])
        r.update(x=9.,vx=0);fields=dict(move=[1,0],heading=0)
        self.assertFalse(sentry_v3.spin_guard(r,obs,nav,fields,True));self.assertEqual(fields['yaw_rate'],math.tau)
        nav['passage']={'id':'test_tunnel_roof'};fields=dict(move=[-1,0],heading=0)
        self.assertFalse(sentry_v3.spin_guard(r,obs,nav,fields,True));self.assertEqual(fields['yaw_rate'],math.tau)

    def test_distant_planned_passage_does_not_brake_but_near_roof_still_interlocks(self):
        obs=copy.deepcopy(self.obs);r=obs['robots'][0]
        obs['arena']=dict(obs['arena'],obstacles=[dict(id='test_tunnel_roof',x=5,y=5,w=2,h=1,bottom=.3,height=.2,rotation=0)])
        r.update(x=1.,y=5.5,heading=.4,vx=0.,vy=0.,angular_velocity=[0,0,.3])
        nav={'passage':{'id':'test_tunnel_roof'}};fields=dict(move=[1.,0.],heading=0.)
        self.assertFalse(sentry_v3.spin_guard(r,obs,nav,fields,False))
        self.assertEqual(fields['move'],[1.,0.]);self.assertNotIn('brake',fields)
        r['x']=4.9;fields=dict(move=[1.,0.],heading=0.,yaw_rate=math.tau)
        self.assertTrue(sentry_v3.spin_guard(r,obs,nav,fields,True))
        self.assertEqual(fields['move'],(0.,0.));self.assertTrue(fields['brake'])
        self.assertNotIn('yaw_rate',fields)

    def test_no_policy_leg_height_and_jump_requires_rising_edge(self):
        obs=copy.deepcopy(self.obs);r=obs['robots'][0]
        r.update(model_slot=2772,x=3,y=7,heading=0,grounded=True,upright=1,legs=[.12,.12])
        state={};obs.update(navigation_mode='adaptive',time=2.)
        intent=dict(transform=-1,jump=1)
        one=sdk.navigate_action(r,[4,7],obs,state,locomotion_intent=intent)
        two=sdk.navigate_action(r,[4,7],dict(obs,time=3.),state,locomotion_intent=intent)
        self.assertEqual(one['jump'],1);self.assertEqual(two['jump'],0)

    def test_tunnel_queue_yields_without_replacing_goal_and_then_reapproaches(self):
        obs=copy.deepcopy(self.obs);r=obs['robots'][0]
        r.update(model_slot=2773,x=9.02,y=11.70,heading=0,vx=0,vy=0,grounded=True,upright=1,transform_progress=1.,radius=.318)
        peer=dict(id='red-4',slot=4,team='red',role='infantry',x=9.65,y=11.73,radius=.318,vx=0,vy=0)
        obs.update(robots=[r,peer],navigation_mode='adaptive',time=1.)
        state={};goal=[12.,7.5];intent=dict(transform=1,jump=-1)
        fields=sdk.navigate_action(r,goal,obs,state,locomotion_intent=intent)
        self.assertEqual(state['wait_for_peer'],'red-4');self.assertEqual(state['goal'],goal)
        self.assertLess(fields['move'][1],0)
        r.update(x=9.32,y=10.6);peer.update(x=9.32,y=13.5);obs['time']=2.
        fields=sdk.navigate_action(r,goal,obs,state,locomotion_intent=intent)
        self.assertEqual(state['phase'],'approach');self.assertGreater(fields['move'][1],0)

    def test_runup_survives_target_jitter_and_landing_is_not_braked_as_unlaunched(self):
        obs=copy.deepcopy(self.obs);r=obs['robots'][0]
        r.update(model_slot=2772,x=10.53,y=7.5,heading=0,vx=0,vy=0,vz=0,grounded=True,upright=1,legs=[.12,.12])
        obs.update(navigation_mode='adaptive',time=13.5,policy_dt=.05)
        route=dict(edge=9.18,lane=7.5,direction=1.,end=10.81,second_edge=10.13,stage=0,since=10.6,staged=True)
        state=dict(goal=[12.,7.5],jump_route=route,last_jump=11.9,stable_since=13.)
        fields=sdk.navigate_action(r,[12.25,7.5],obs,state,locomotion_intent=dict(transform=-1,jump=-1))
        self.assertIs(state['jump_route'],route);self.assertEqual(state['goal'],[12.25,7.5])
        self.assertGreater(fields['move'][0],0)
        r.update(x=7.913,y=7.669)
        fields=sdk.navigate_action(r,[12.25,7.5],obs,state,locomotion_intent=dict(transform=-1,jump=-1))
        self.assertLess(fields['move'][1],0)
        sdk.navigate_action(r,[7.,7.5],obs,state,locomotion_intent=dict(transform=-1,jump=-1))
        self.assertNotIn('jump_route',state)

    def test_destination_jitter_does_not_reverse_completed_tunnel_crossing(self):
        obs=copy.deepcopy(self.obs);r=obs['robots'][0]
        r.update(model_slot=2773,x=9.32,y=13.60,heading=math.pi/2,vx=0,vy=0,
                 grounded=True,upright=1,transform_progress=0.,radius=.318)
        obs.update(navigation_mode='adaptive',time=20.)
        state=dict(goal=[12.,7.5],passed_passages=['red_road_tunnel_roof'])
        fields=sdk.navigate_action(r,[16.25,7.5],obs,state,locomotion_intent=dict(transform=-1,jump=-1))
        self.assertEqual(state['goal'],[16.25,7.5])
        self.assertIn('red_road_tunnel_roof',state['passed_passages'])
        self.assertNotIn('passage',state)
        self.assertEqual(state['phase'],'route')
        self.assertFalse(fields['transform'])
        sdk.navigate_action(r,[7.,7.5],obs,state,locomotion_intent=dict(transform=-1,jump=-1))
        self.assertNotIn('passed_passages',state)

    def test_actor_and_controller_cadence(self):
        s=ArenaSession(self.config)
        try:
            s.reset(1001)
            _,_,_,_,info=s.step([-.5,0,-1,-1,-1,-1]);self.assertEqual(s.controller.calls,2)
            self.assertEqual([x['time'] for x in info['action_trace']['control_steps']],[0.,.05])
            self.assertEqual([x['result_time'] for x in info['action_trace']['control_steps']],[.05,.1])
            s.step([-.4,0,-1,-1,-1,-1]);self.assertEqual(s.controller.calls,4)
            self.assertEqual(s.battle.match.tick,4)
        finally:s.close()

    def test_first_control_jump_pulse_is_counted_once(self):
        from rmuc2027_research.rl.sentry_training import configuration
        cfg=configuration(2772);cfg.update(sim_backend='rapier',navigation_precompute=False)
        s=ArenaSession(cfg)
        try:
            s.reset(1001)
            for _ in range(4):s.step([-.75,0,-1,-1,-1,-1])
            _,_,_,_,info=s.step([-.75,0,-1,1,-1,-1])
            self.assertEqual([x['accepted']['jump'] for x in info['action_trace']['control_steps']],[True,False])
            self.assertEqual(info['metrics']['jump_commands'],1)
        finally:s.close()

    def test_public_objective_progress_does_not_override_actor_goal(self):
        s=ArenaSession(self.config)
        try:
            s.reset(1001)
            _,reward,_,_,info=s.step([-.5,0,-1,-1,-1,-1])
            self.assertTrue(math.isfinite(reward))
            self.assertIn('objective_gap_m',info['metrics'])
            self.assertEqual(s.controller.state['goal'],s.controller.state['executed_goal'])
            self.assertFalse(any('objective_gap' in n for n in sentry_v3.FEATURE_INDEX))
        finally:s.close()

    def test_demonstration_retention_updates_actor_without_fake_samples(self):
        import numpy as np
        import torch
        from rmuc2027_research.rl.env import SentryEnv
        from rmuc2027_research.rl.learning import TimedPPO
        from rmuc2027_research.rl.sentry_training import retain_demonstrations
        torch.set_num_threads(1)
        env=SentryEnv(self.config)
        try:
            model=TimedPPO('MlpPolicy',env,n_steps=8,batch_size=8,learning_rate=5e-5,seed=42)
            x=np.asarray([sentry_v3.encode(self.obs,{})]*8,dtype=np.float32)
            y=np.tile(np.asarray([-.4,.3,-.9,-.9,.9,.9],dtype=np.float32),(8,1))
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'demo.npz';np.savez(path,observations=x,targets=y)
                model.sentry_guidance_path=str(path)
                before=retain_demonstrations(model)
                for _ in range(12):after=retain_demonstrations(model)
                self.assertLess(after,before);self.assertEqual(model.num_timesteps,0)
                self.assertIn('_sentry_guidance_cache',model._excluded_save_params())
        finally:env.close()

    def test_capture_demonstration_stays_on_real_polygon_after_projection(self):
        from rmuc2027_research.rl.sentry_training import teacher_action
        state={'teacher_staging_reached':True}
        a=teacher_action(self.obs,state)
        goal=SdkNavigator().resolve_goal(own_robot(self.obs,7),[(a[0]+1)*14,(a[1]+1)*7.5],self.obs['arena'])
        zone=next(z for z in self.obs['arena']['zones'] if z['id']=='blue_outpost')
        self.assertTrue(sdk.point_in_polygon(*goal,zone['polygon']))
        s=ArenaSession(self.config);s.objective_zones=[zone]
        self.assertGreater(s.objective_gap({'x':16.25,'y':3.45}),0.)
        self.assertEqual(s.objective_gap(dict(zip(('x','y'),goal))),0.)

    def test_teacher_does_not_send_crossed_staging_trajectory_backwards(self):
        from rmuc2027_research.rl.sentry_training import teacher_action
        for team in ('red','blue'):
            obs=copy.deepcopy(self.obs);obs['team']=team;r=obs['robots'][0]
            r.update(model_slot=2772,team=team,id=team+'-7',x=12.8 if team=='red' else 15.2,y=7.5)
            state={};a=teacher_action(obs,state)
            self.assertTrue(state['teacher_staging_reached'])
            self.assertGreater((a[0]+1)*14,12.8)

    def test_exported_actor_and_native_executor_match(self):
        from rmuc2027_research.rl.exporting import export_layers
        from rmuc2027_research.rl.learning import experiment
        from rmuc2027_core.submissions import load_submission
        from rmuc2027_core.policy import Policy
        n=sentry_v3.schema()['obs_size'];action=[-.5,0,-1,-1,-1,-1]
        layers=[dict(weight=[[0.]*n for _ in range(6)],bias=action)]
        obs=copy.deepcopy(self.obs);obs.update(navigation_mode='adaptive',policy_dt=.05)
        state={};nav=SdkNavigator()
        select_goal(obs,state,action,nav,self.config)
        expected=execute_goal(obs,state,nav,self.config,None)
        with tempfile.TemporaryDirectory() as temp:
            export_layers(layers,experiment(self.config),Path(temp)/'squad')
            _,entries=load_submission(Path(temp)/'squad')
            policy=Policy(entries['7'],1,'trusted-local',5.)
            try:
                actual,error=policy.decide(obs)
                self.assertIsNone(error);self.assertEqual(actual,expected)
            finally:policy.close()


if __name__=='__main__':unittest.main()
