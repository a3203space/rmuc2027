import unittest
from unittest.mock import patch
from rmuc2027_core import sdk

class NavigationTests(unittest.TestCase):
    def test_goal_near_outpost_does_not_allow_driving_through_pedestal(self):
        robot = dict(team='blue', x=16., y=1.8, model_slot=2732, transform_progress=0.)
        goal = (16.881168202514765, 4.27379453665575)
        with patch.object(sdk, '_square_to_bumps', side_effect=lambda r,p,a:p):
            waypoint = sdk._chapter4_navigate(robot, goal, sdk._REFERENCE_ARENA)
        self.assertNotEqual(waypoint, goal)
        # A waypoint to the left of the pedestal makes the final approach from
        # its open side, instead of relaxing the entire obstacle at the goal.
        self.assertLess(waypoint[0], 16.694)
        graph = sdk._prepare_chapter4(sdk._REFERENCE_ARENA, 'blue', False)
        self.assertTrue(graph[2]((robot['x'],robot['y']), waypoint))

    def test_route_retains_obstructed_corner(self):
        nodes = [(1.,2.), (3.,2.)]
        def clear(a,b,*args):
            return not (a[1]<1.5 and b[0]>2. or b[1]<1.5 and a[0]<2.)
        graph = (nodes,[[(1,2.)],[(0,2.)]],clear,lambda *a: {})
        path = sdk._complete_route(graph,(1.,1.),(3.,1.))
        self.assertEqual(path,[(1.,2.),(3.,2.),(3.,1.)])

    def test_heavy_workbench_route_reaches_goal_inside_only_safety_margin(self):
        graph=sdk._prepare_chapter4(sdk._REFERENCE_ARENA,'red',False,.429)
        goal=(1.22,12.825)
        path=sdk._complete_route(graph,(13.06924,6.70042),goal)
        self.assertTrue(path)
        self.assertEqual(path[-1],goal)
        robot=dict(team='red',x=3.2814,y=10.65626,model_slot=2711,radius=.389,speed_reference_mps=3.)
        state=dict(trajectory_goal=list(goal),trajectory=[(3.281,10.655),goal])
        fields=sdk._trajectory_move(robot,goal,dict(arena=sdk._REFERENCE_ARENA,policy_dt=.1),state)
        self.assertEqual(state['trajectory'],[goal])
        self.assertGreater(sum(v*v for v in fields['move']),.001)

    def test_complete_route_near_goal_keeps_pedestal_as_obstacle(self):
        graph=sdk._prepare_chapter4(sdk._REFERENCE_ARENA,'blue',False)
        start=(16.,1.8);goal=(16.881168202514765,4.27379453665575)
        path=sdk._complete_route(graph,start,goal)
        self.assertTrue(path)
        self.assertNotEqual(path[0],goal)
        self.assertTrue(graph[2](start,path[0]))

    def test_predictive_stop_and_legacy_unchanged(self):
        graph = ([],[],lambda *a: True,lambda *a: {})
        robot = dict(id='red:7',team='red',x=2.,y=2.,vx=0.,vy=0.,radius=.3,speed_reference_mps=3.)
        obs = dict(navigation_mode='predictive',policy_dt=.5,
                   arena=dict(width=28,height=15),robots=[robot],enemies=[dict(id='blue:7',x=2.7,y=2.,radius=.3)])
        fields = {'move':(1.,0.),'power':200}
        with patch.object(sdk,'_prepare_chapter4',return_value=graph), patch.object(sdk,'_design',return_value={}):
            result = sdk.predictive_navigation(robot,fields,obs,{})
            self.assertTrue(result.get('brake'))
            obs['navigation_mode']='legacy'
            self.assertEqual(sdk.predictive_navigation(robot,fields,obs,{}),fields)

    def test_half_second_corner_limit(self):
        graph = ([],[],lambda *a: True,lambda *a: {})
        robot = dict(team='red',x=2.,y=2.,speed_reference_mps=3.)
        obs = dict(arena={},policy_dt=.5)
        with patch.object(sdk,'_prepare_chapter4',return_value=graph), patch.object(sdk,'_design',return_value={}), patch.object(sdk,'_square_to_bumps',return_value=(1.,0.)):
            state = {'trajectory_goal':[5.,5.], 'trajectory':[(3.,2.),(3.,5.),(5.,5.)]}
            move = sdk._trajectory_move(robot,(5.,5.),obs,state)['move']
            self.assertGreater(move[0],0)
            self.assertLess(move[0],1)

    def test_existing_peer_margin_allows_escape_but_not_approach(self):
        graph = ([],[],lambda *a: True,lambda *a: {})
        robot = dict(id='red:7',team='red',x=2.,y=2.,vx=0.,vy=0.,radius=.3,speed_reference_mps=3.)
        peer = dict(id='red:4',x=2.65,y=2.,radius=.3,vx=0.,vy=0.)
        obs = dict(navigation_mode='predictive',policy_dt=.1,
                   arena=dict(width=28,height=15),robots=[robot,peer])
        with patch.object(sdk,'_prepare_chapter4',return_value=graph), patch.object(sdk,'_design',return_value={}):
            escaping = sdk.predictive_navigation(robot,{'move':(-1.,0.)},obs,{})
            approaching = sdk.predictive_navigation(robot,{'move':(1.,0.)},obs,{})
            self.assertLess(escaping['move'][0], 0.)
            self.assertTrue(approaching['brake'])
            # Moving traffic still receives the complete predictive margin.
            peer['vx']=-.2
            self.assertTrue(sdk.predictive_navigation(robot,{'move':(-1.,0.)},obs,{}).get('brake'))


    def test_training_presets_keep_working_navigation(self):
        from rmuc2027_research.rl.session import load_config
        from rmuc2027_core.paths import ROOT
        for path in (ROOT/'configs/rl').glob('sentry-mjwarp*.json'):
            config = load_config(path)
            self.assertEqual(config['navigation_mode'], 'legacy')
            self.assertEqual(config['learning_control_hz'], 2)
            self.assertEqual(config['rl_seconds'], .5)
            self.assertFalse(config['drones_enabled'])

    def test_legacy_does_not_inject_braking(self):
        robot = {'id':'red-7'}
        fields = {'move':(1.,0.), 'heading':.2, 'power':150}
        with patch.object(sdk, '_navigate_action_legacy', return_value=fields):
            result = sdk.navigate_action(robot,(10.,5.),{'navigation_mode':'legacy','time':0.}, {})
            self.assertEqual(result, fields)
            self.assertNotIn('brake',result)
