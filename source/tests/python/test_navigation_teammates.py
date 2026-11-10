import unittest
from unittest.mock import patch
from rmuc2027_core import sdk

class TeammateNavigationTests(unittest.TestCase):
    def setup_scene(self):
        r=dict(id='red-7',team='red',alive=True,grounded=True,x=2.,y=5.,radius=.3,role='sentry')
        peer=dict(id='red-4',team='red',x=2.8,y=5.,radius=.3,role='infantry',vx=0.,vy=0.)
        obs=dict(time=1.,robots=[r,peer],arena=dict(layout_revision='chapter4-2027-v1'))
        return r,obs

    def test_detour_changes_local_move_but_keeps_goal(self):
        r,obs=self.setup_scene();state={'goal':[12.,5.],'teammate_wait_since':0.};fields={'move':(1.,0.),'fire':True}
        graph=([],[],lambda *args:True,lambda *args:{})
        with patch.object(sdk,'_prepare_chapter4',return_value=graph):
            command=sdk._stationary_teammate_detour(r,(12.,5.),fields,obs,state)
        self.assertGreater(abs(command['move'][1]),.5)
        self.assertTrue(command['fire'])
        self.assertEqual(state['goal'],[12.,5.])
        self.assertEqual(fields['move'],(1.,0.))

    def test_no_clear_detour_preserves_command_and_manoeuvre(self):
        r,obs=self.setup_scene();fields={'move':(1.,0.)}
        graph=([],[],lambda *args:False,lambda *args:{})
        with patch.object(sdk,'_prepare_chapter4',return_value=graph):
            self.assertEqual(sdk._stationary_teammate_detour(r,(12.,5.),fields,obs,{}),fields)
        state={'entry':{'phase':'align'}}
        self.assertEqual(sdk._stationary_teammate_detour(r,(12.,5.),fields,obs,state),fields)

    def test_detour_requires_sustained_stop_and_resets_after_motion(self):
        r,obs=self.setup_scene();fields={'move':(1.,0.)};state={}
        graph=([],[],lambda *args:True,lambda *args:{})
        with patch.object(sdk,'_prepare_chapter4',return_value=graph):
            for now in (1.,1.25,1.5):
                obs['time']=now
                self.assertEqual(sdk._stationary_teammate_detour(r,(12.,5.),fields,obs,state),fields)
                self.assertNotIn('teammate_detour',state)
            r['vx']=.3;obs['time']=1.7
            self.assertEqual(sdk._stationary_teammate_detour(r,(12.,5.),fields,obs,state),fields)
            self.assertNotIn('teammate_wait_since',state)
            r['vx']=0.;obs['time']=2.
            self.assertEqual(sdk._stationary_teammate_detour(r,(12.,5.),fields,obs,state),fields)
            obs['time']=2.7
            command=sdk._stationary_teammate_detour(r,(12.,5.),fields,obs,state)
            self.assertGreater(abs(command['move'][1]),.5)

    def test_moving_runup_does_not_start_detour(self):
        r,obs=self.setup_scene();r['vx']=2.
        state={'entry':{'phase':'approach'},'teammate_wait_since':0.}
        fields={'move':(1.,0.)}
        graph=([],[],lambda *args:True,lambda *args:{})
        with patch.object(sdk,'_prepare_chapter4',return_value=graph):
            self.assertEqual(sdk._stationary_teammate_detour(r,(12.,5.),fields,obs,state),fields)
        self.assertNotIn('teammate_detour',state)

    def test_fast_tunnel_approach_brakes_before_folding(self):
        r=dict(model_slot=2773,id='red-7',team='red',alive=True,grounded=True,
               x=8.9,y=11.7,vx=2.,vy=0.,mass_kg=27.15)
        with patch.object(sdk,'navigate',return_value=(1.,0.)):
            command=sdk._navigate_action_legacy(r,(12.,7.5),{'time':1.,'arena':{}},{})
        self.assertEqual(command['move'],(0.,0.))
        self.assertTrue(command['brake'])
        self.assertFalse(command.get('transform',False))
