"""SDK ballistic limit regressions; pure geometry/integration, no engine."""
import copy
import math
import unittest
from rmuc2027_core import sdk

FIXTURES = [{'robot': {'id': 'red-3', 'team': 'red', 'slot': 3, 'model_slot': 2731, 'x': 15.694130563805714, 'y': 6.236831187056359, 'z': 0.43834639111187623, 'vx': -7.05204065897363e-05, 'vy': -0.0003798674721447361, 'hp': 200.0, 'alive': True, 'tipped': False, 'grounded': True, 'transform_progress': 0.0, 'ammo': 58, 'heat': 35.99999999999946, 'max_heat': 40.0, 'weak': False, 'overheated': False, 'heat_permanent_lock': False, 'overspeed_locked': False, 'overspeed_permanent_lock': False, 'launcher_fault': False, 'red_card': False, 'pending_ammo': [], 'last_shot_time': 59.658333333, 'last_damage_time': -1000.0, 'heading': 1.4620768427542257, 'aim': 6.983410504992018, 'pitch': -0.2548325959094274, 'quaternion': [0.0010891850118731983, 0.0007854184546365164, 0.6676426200086139, 0.744480710792358], 'muzzle_height_m': 0.435, 'muzzle_reach_m': 0.22, 'muzzle_speed_mps': 24.6, 'armor_center_z_m': 0.28, 'chassis_size_m': [0.42, 0.36, 0.12], 'role': 'infantry'}, 'target': {'id': 'blue-7', 'team': 'blue', 'slot': 7, 'model_slot': 2773, 'x': 16.275686533328603, 'y': 6.529164258126851, 'z': 0.45297808842159454, 'vx': 2.3974540783078617e-121, 'vy': -2.188469662024319e-18, 'hp': 325.0, 'alive': True, 'tipped': False, 'grounded': True, 'transform_progress': 0.0, 'ammo': 300, 'heat': 0.0, 'max_heat': 260.0, 'weak': False, 'overheated': False, 'heat_permanent_lock': False, 'overspeed_locked': False, 'overspeed_permanent_lock': False, 'launcher_fault': False, 'red_card': False, 'pending_ammo': [], 'last_shot_time': -1000.0, 'last_damage_time': 28.908333333, 'heading': 1.3048490276674203, 'aim': 3.140314726785216, 'pitch': 0.03998290561571092, 'quaternion': [0.01161192694271849, -0.008872939456265959, 0.6070929378840626, 0.7944964435851051], 'muzzle_height_m': 0.4, 'muzzle_reach_m': 0.23, 'muzzle_speed_mps': 24.6, 'armor_center_z_m': 0.15, 'chassis_size_m': [0.46, 0.44, 0.025], 'role': 'sentry'}}, {'robot': {'id': 'red-4', 'team': 'red', 'slot': 4, 'model_slot': 2732, 'x': 16.59178656204376, 'y': 7.211815393933899, 'z': 0.4708556690680146, 'vx': 7.561757635347305e-05, 'vy': 0.0001423657041947842, 'hp': 225.0, 'alive': True, 'tipped': False, 'grounded': True, 'transform_progress': 0.0, 'ammo': 27, 'heat': 36.699999999999896, 'max_heat': 48.0, 'weak': False, 'overheated': False, 'heat_permanent_lock': False, 'overspeed_locked': False, 'overspeed_permanent_lock': False, 'launcher_fault': False, 'red_card': False, 'pending_ammo': [], 'last_shot_time': 59.458333333, 'last_damage_time': -1000.0, 'heading': -1.6164826081292119, 'aim': 4.224901168568919, 'pitch': -0.23874230022022608, 'quaternion': [0.010088782395206268, 0.010564905510095436, -0.7229895242489544, 0.6907045294969589], 'muzzle_height_m': 0.4, 'muzzle_reach_m': 0.23, 'muzzle_speed_mps': 24.6, 'armor_center_z_m': 0.15, 'chassis_size_m': [0.46, 0.44, 0.025], 'role': 'infantry'}, 'target': {'id': 'blue-7', 'team': 'blue', 'slot': 7, 'model_slot': 2773, 'x': 16.275686533328603, 'y': 6.529164258126851, 'z': 0.45297808842159454, 'vx': 2.3974540783078617e-121, 'vy': -2.188469662024319e-18, 'hp': 325.0, 'alive': True, 'tipped': False, 'grounded': True, 'transform_progress': 0.0, 'ammo': 300, 'heat': 0.0, 'max_heat': 260.0, 'weak': False, 'overheated': False, 'heat_permanent_lock': False, 'overspeed_locked': False, 'overspeed_permanent_lock': False, 'launcher_fault': False, 'red_card': False, 'pending_ammo': [], 'last_shot_time': -1000.0, 'last_damage_time': 28.908333333, 'heading': 1.3048490276674203, 'aim': 3.140314726785216, 'pitch': 0.03998290561571092, 'quaternion': [0.01161192694271849, -0.008872939456265959, 0.6070929378840626, 0.7944964435851051], 'muzzle_height_m': 0.4, 'muzzle_reach_m': 0.23, 'muzzle_speed_mps': 24.6, 'armor_center_z_m': 0.15, 'chassis_size_m': [0.46, 0.44, 0.025], 'role': 'sentry'}}]

class SDKMinimumPitchTests(unittest.TestCase):
    def setUp(self): sdk._AIM_CACHE.clear()

    def test_observed_close_targets_below_minimum_pitch_are_unreachable(self):
        for pair in FIXTURES:
            robot,target=copy.deepcopy(pair['robot']),copy.deepcopy(pair['target'])
            before=copy.deepcopy((robot,target))
            self.assertFalse(sdk.ballistic_reachable(robot,target))
            self.assertEqual(sdk.aim_pitch(robot,target),-math.pi/12)
            self.assertEqual((robot,target),before)
            self.assertFalse(sdk.ballistic_reachable(robot,target))  # cached result agrees

    def test_same_real_muzzle_and_armor_geometry_reachable_at_standoff(self):
        for pair in FIXTURES:
            robot,target=copy.deepcopy(pair['robot']),copy.deepcopy(pair['target'])
            dx,dy=robot['x']-target['x'],robot['y']-target['y'];distance=math.hypot(dx,dy)
            robot.update(x=target['x']+1.7*dx/distance,y=target['y']+1.7*dy/distance,vx=0.,vy=0.)
            self.assertTrue(sdk.ballistic_reachable(robot,target))
            angle=sdk.aim_pitch(robot,target)
            self.assertGreater(angle,-math.pi/12);self.assertLess(angle,math.pi/4)
            point=sdk._aim_target(robot,target)
            # Independent evaluation of the returned aim using the same public
            # muzzle dimensions and drag constants checks aim/reach consistency.
            distance=math.hypot(point[0]-robot['x'],point[1]-robot['y'])
            base=robot['z']+sdk._rotate(robot['quaternion'],(0,0,robot['muzzle_height_m']))[2]
            x=robot['muzzle_reach_m']*math.cos(angle);z=base+robot['muzzle_reach_m']*math.sin(angle)
            vx=robot['muzzle_speed_mps']*math.cos(angle);vz=robot['muzzle_speed_mps']*math.sin(angle)
            drag=.5*1.225*.47*math.pi*.0084**2/.0032;dt=1/120
            for _ in range(540):
                speed=math.hypot(vx,vz);ax=-drag*speed*vx;az=-9.81-drag*speed*vz
                xn=x+vx*dt+.5*ax*dt*dt;zn=z+vz*dt+.5*az*dt*dt
                if xn>=distance:
                    height=z+(zn-z)*(distance-x)/(xn-x);break
                x,z=xn,zn;vx+=ax*dt;vz+=az*dt
            else: self.fail('returned reachable trajectory never reaches target plane')
            self.assertAlmostEqual(height,point[2],delta=.002)

    def test_above_maximum_pitch_remains_unreachable(self):
        pair=copy.deepcopy(FIXTURES[0]);pair['target']['z']+=20
        self.assertFalse(sdk.ballistic_reachable(pair['robot'],pair['target']))
        self.assertEqual(sdk.aim_pitch(pair['robot'],pair['target']),math.pi/4)

if __name__=='__main__':unittest.main()
