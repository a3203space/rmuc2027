"""Pure geometry/control regression for legal missing coordinate observations."""
import copy
import math
import unittest
from unittest.mock import patch
from rmuc2027_core import sdk


class MissingPositionTests(unittest.TestCase):
    def scene(self):
        robot=dict(id='red-7',team='red',slot=7,role='sentry',alive=True,grounded=True,
            x=2.,y=2.,vx=0.,vy=0.,radius=.3,speed_reference_mps=3.)
        obs=dict(navigation_mode='predictive',time=1.,policy_dt=.1,
            arena=dict(width=28,height=15,layout_revision='chapter4-2027-v1'),robots=[robot],enemies=[])
        return robot,obs,([],[],lambda *args:True,lambda *args:{})

    def test_legal_unknown_enemy_coordinates_do_not_raise_or_invent_positions(self):
        robot,obs,graph=self.scene()
        fields=dict(move=(1.,0.),fire=True,jump=0.)
        missing=[dict(id='blue-'+str(slot),team='blue',slot=slot,position_valid=False,
            hp_valid=False,status_valid=False) for slot in (1,3,4,7)]
        with patch.object(sdk,'_prepare_chapter4',return_value=graph),patch.object(sdk,'_design',return_value={}):
            reference=sdk.predictive_navigation(robot,fields,obs,{})
            obs['enemies']=missing;before=copy.deepcopy(obs)
            self.assertEqual(sdk.predictive_navigation(robot,fields,obs,{}),reference)
            self.assertEqual(obs,before)

    def test_invalid_retained_coordinates_do_not_block_but_known_peer_still_does(self):
        robot,obs,graph=self.scene()
        peer=dict(id='blue-7',team='blue',slot=7,x=2.65,y=2.,radius=.3,vx=0.,vy=0.)
        fields={'move':(1.,0.)}
        with patch.object(sdk,'_prepare_chapter4',return_value=graph),patch.object(sdk,'_design',return_value={}):
            reference=sdk.predictive_navigation(robot,fields,obs,{})
            obs['enemies']=[dict(peer,position_valid=False)]
            self.assertEqual(sdk.predictive_navigation(robot,fields,obs,{}),reference)
            for valid in (True,None):
                obs['enemies']=[dict(peer,position_valid=valid)]
                self.assertTrue(sdk.predictive_navigation(robot,fields,obs,{}).get('brake'))

    def test_malformed_position_is_unknown_and_valid_zero_coordinate_is_known(self):
        for coordinate in (None,True,float('nan'),float('inf'),'2'):
            self.assertFalse(sdk._navigation_position_known(dict(x=coordinate,y=2.)))
        self.assertTrue(sdk._navigation_position_known(dict(x=0.,y=0.,position_valid=True)))

    def test_unknown_teammate_cannot_create_detour_or_raise(self):
        robot,obs,graph=self.scene();fields={'move':(1.,0.)}
        for peer in (dict(id='red-4',team='red',position_valid=False),
                dict(id='red-4',team='red',x=2.65,y=2.,position_valid=False)):
            obs['robots']=[robot,peer];state={'teammate_wait_since':0.}
            with patch.object(sdk,'_prepare_chapter4',return_value=graph):
                self.assertEqual(sdk._stationary_teammate_detour(robot,(6.,2.),fields,obs,state),fields)
            self.assertNotIn('teammate_detour',state)

    def test_lost_and_reacquired_peer_changes_guard_without_sticky_stop(self):
        robot,obs,graph=self.scene();state={};fields={'move':(1.,0.)}
        peer=dict(id='blue-7',team='blue',x=2.65,y=2.,radius=.3,position_valid=True)
        with patch.object(sdk,'_prepare_chapter4',return_value=graph),patch.object(sdk,'_design',return_value={}):
            for peers,blocked in (([peer],True),([dict(id='blue-7',position_valid=False)],False),([],False),([peer],True)):
                obs['enemies']=peers
                result=sdk.predictive_navigation(robot,fields,obs,state)
                self.assertEqual(bool(result.get('brake')),blocked)
                self.assertEqual(state['limited'],blocked)
                self.assertTrue(all(math.isfinite(v) for v in result['move']))

    def test_invalid_teammate_position_cannot_hold_tunnel_queue(self):
        robot,obs,_=self.scene();robot.update(model_slot=2773,x=9.32,y=10.7)
        obs.update(navigation_mode='adaptive',arena=sdk._REFERENCE_ARENA)
        passage=dict(center=[9.32,12.735],axis=[0.,1.],half=.175,width=1.,since=0.,id='red_road_tunnel_roof')
        peer=dict(id='red-4',team='red',slot=4,x=9.32,y=11.74,radius=.3)
        ctx=dict(stable=True,can_fold=True,can_unfold=True,can_jump=False,legged=False,cooldown=0.,roof=None)
        with patch.object(sdk,'locomotion_context',return_value=ctx),\
                patch.object(sdk,'_trajectory_move',return_value={'move':(0.,1.)}):
            for valid in (False,True):
                obs['robots']=[robot,dict(peer,position_valid=valid)]
                state=dict(time=0.,goal=[12.,7.5],passage=copy.deepcopy(passage))
                sdk.navigate_action(robot,(12.,7.5),obs,state)
                self.assertEqual(state['waiting_peer_active'],valid)


class NavigationLifecycleTests(unittest.TestCase):
    def test_clock_reset_across_sentry_chassis_and_sides_matches_fresh_state(self):
        shared=dict(time=30.,passage={'id':'old'},jump_route={'phase':'launch'},
            fold_intent=True,fold_since=30.,jump_held=True,last_jump=30.,
            passed_passages=['old'],trajectory=[[99.,99.]],avoid_offset=.5)
        for model in (2771,2773,2772,2773):
            for team in ('red','blue'):
                with self.subTest(model=model,team=team):
                    robot=dict(id=team+'-7',team=team,slot=7,model_slot=model,role='sentry',alive=True,powered=True,
                        grounded=True,upright=1.,x=2. if team=='red' else 26.,y=7. if team=='red' else 8.,
                        z=0.,vx=0.,vy=0.,vz=0.,radius=.28,transform_progress=0.,speed_reference_mps=3.)
                    obs=dict(time=0.,arena=sdk._REFERENCE_ARENA,navigation_mode='adaptive',policy_dt=.05,
                        robots=[robot],enemies=[])
                    goal=(4.,7.) if team=='red' else (24.,8.);fresh={}
                    intent=dict(transform=-1.,jump=-1.)
                    expected=sdk.navigate_action(robot,goal,obs,fresh,locomotion_intent=intent)
                    actual=sdk.navigate_action(robot,goal,obs,shared,locomotion_intent=intent)
                    self.assertEqual(actual,expected);self.assertEqual(shared,fresh)
                    self.assertTrue(all(math.isfinite(v) for v in actual['move']))
                    shared['time']=30.

    def test_death_then_respawn_cannot_reuse_old_passage_or_brake(self):
        state=dict(time=5.,passage={'id':'old'},limited=True,jump_held=True,fold_intent=True)
        result=sdk.navigate_action({'alive':False},(4.,7.),dict(time=6.,navigation_mode='adaptive'),state)
        self.assertEqual(state,{});self.assertEqual(result,{'move':(0.,0.)})


if __name__=='__main__':unittest.main()
