"""Input edges, optical-only hold-to-track and real-engine weapon feedback."""
import copy
import math
import tempfile
import time
import unittest
from rmuc2027_core.engine import Match
from rmuc2027_server.aim import AimAssist, wrap
from rmuc2027_server.control import HumanController, baseline_helpers
from rmuc2027_server.hud import weapon_status
from rmuc2027_server.projection import project
from rmuc2027_server.protocol import Input, Join, Seat
from rmuc2027_server.room import Room


def intent(seq=1, **kwargs):
    return Input(match_id='combat-test', seq=seq, yaw=kwargs.pop('yaw', 0), pitch=kwargs.pop('pitch', 0), **kwargs)


class AimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        match = Match()
        try:
            cls.source = copy.deepcopy(match._frame)
        finally:
            match.close()

    def setUp(self):
        self.robot = copy.deepcopy(next(r for r in self.source['robots'] if r['id']=='red-7'))
        self.robot.update(x=0, y=0, z=0, aim=0, pitch=0, vx=0, vy=0, vz=0, quaternion=[0,0,0,1])
        self.enemy = copy.deepcopy(next(r for r in self.source['robots'] if r['id']=='blue-3'))
        self.enemy.update(x=4, y=.4, z=0, vx=0, vy=0, alive=True, targetable=True, invulnerable=False)
        self.observation = {'vision': {'7': {'enemies': ['blue-3'], 'camera': {
            'origin': [0,0,self.robot['muzzle_height_m']+.08], 'direction': [1,0,0]}}},
            'enemies': [self.enemy]}
        self.aim = AimAssist()

    def test_hold_acquires_and_release_returns_actual_gimbal_without_jump(self):
        self.aim.angles(self.observation, self.robot, intent(aim_assist=True))
        self.assertEqual(self.aim.snapshot(), {'state':'tracking','target_id':'blue-3'})
        self.robot.update(aim=.12, pitch=.03)
        yaw, pitch = self.aim.angles(self.observation, self.robot, intent(2, aim_assist=False))
        self.assertAlmostEqual(yaw,.12); self.assertAlmostEqual(pitch,.03)
        yaw, pitch = self.aim.angles(self.observation, self.robot, intent(3,yaw=.04,pitch=.02))
        self.assertAlmostEqual(yaw,.16); self.assertAlmostEqual(pitch,.05)
        self.assertEqual(self.aim.state,'idle'); self.assertIsNone(self.aim.target_id)

    def test_radar_or_another_camera_cannot_supply_a_hidden_target(self):
        self.observation['vision']['7']['enemies'] = []
        self.observation['vision']['3'] = {'enemies':['blue-3']}
        self.aim.angles(self.observation,self.robot,intent(aim_assist=True))
        self.assertEqual(self.aim.state,'searching'); self.assertIsNone(self.aim.target_id)

    def test_closest_to_crosshair_then_stable_identity(self):
        other = dict(self.enemy,id='blue-4',y=.8)
        self.observation['enemies'].append(other)
        self.observation['vision']['7']['enemies'].append('blue-4')
        self.aim.angles(self.observation,self.robot,intent(aim_assist=True))
        other['y'] = 0
        self.aim.angles(self.observation,self.robot,intent(2,aim_assist=True))
        self.assertEqual(self.aim.target_id,'blue-3')

    def test_obstruction_loss_and_reacquisition(self):
        self.aim.angles(self.observation,self.robot,intent(aim_assist=True))
        self.observation['vision']['7']['enemies'] = []
        self.aim.angles(self.observation,self.robot,intent(2,aim_assist=True))
        self.assertEqual(self.aim.state,'lost'); self.assertIsNone(self.aim.target_id)
        self.observation['vision']['7']['enemies'] = ['blue-3']
        self.aim.angles(self.observation,self.robot,intent(3,aim_assist=True))
        self.assertEqual(self.aim.state,'tracking')

    def test_angular_cone_dead_invulnerable_aerial_and_friendly_are_rejected(self):
        for changed in ({'y':3}, {'alive':False}, {'invulnerable':True}, {'role':'aerial'}, {'team':'red'}):
            with self.subTest(changed=changed):
                original=dict(self.enemy); self.enemy.update(changed)
                self.aim.reset(); self.aim.angles(self.observation,self.robot,intent(aim_assist=True))
                self.assertIsNone(self.aim.target_id)
                self.enemy.clear(); self.enemy.update(original)

    def test_acquisition_and_retention_use_different_cones(self):
        self.aim.angles(self.observation,self.robot,intent(aim_assist=True))
        self.enemy['y']=4*math.tan(math.radians(18))
        self.aim.angles(self.observation,self.robot,intent(2,aim_assist=True))
        self.assertEqual(self.aim.target_id,'blue-3')
        self.enemy['y']=4*math.tan(math.radians(22))
        self.aim.angles(self.observation,self.robot,intent(3,aim_assist=True))
        self.assertIsNone(self.aim.target_id)
        self.assertEqual(self.aim.state,'lost')

    def test_moving_target_has_lead_and_yaw_wrap_is_continuous(self):
        stationary = self.aim.angles(self.observation,self.robot,intent(aim_assist=True))[0]
        self.enemy['vy']=1
        moving = self.aim.angles(self.observation,self.robot,intent(2,aim_assist=True))[0]
        self.assertGreater(moving,stationary)
        self.aim.reset()
        self.robot['aim']=math.pi-.02
        self.aim.angles(self.observation,self.robot,intent(yaw=math.pi-.02))
        yaw,_=self.aim.angles(self.observation,self.robot,intent(2,yaw=-math.pi+.02))
        self.assertAlmostEqual(wrap(yaw-(math.pi-.02)),.04)

    def test_reset_discards_target_and_reanchors(self):
        self.aim.angles(self.observation,self.robot,intent(aim_assist=True))
        self.aim.reset(); self.robot['aim']=.7
        yaw,_=self.aim.angles(self.observation,self.robot,intent())
        self.assertAlmostEqual(yaw,.7); self.assertEqual(self.aim.state,'idle')

    def test_assistance_never_fires_and_manual_fire_cancels_heavy_assistance(self):
        match=Match()
        try:
            robot=next(r for r in match._frame['robots'] if r['id']=='red-1')
            controller=HumanController(baseline_helpers())
            obs=match.compact_observation('red',[1])
            action=controller.action(obs,robot,intent(aim_assist=True))
            self.assertFalse(action['fire'])
            controller.mode='pick'
            action=controller.action(obs,robot,intent(fire=True))
            self.assertTrue(action['fire']); self.assertEqual(controller.mode,'')
            action=controller.action(obs,robot,intent(active=False,fire=True,aim_assist=True))
            self.assertFalse(action['fire']); self.assertEqual(controller.aim.state,'idle')
        finally:match.close()

    def test_inactive_rpc_angles_do_not_corrupt_resume_mouse_baseline(self):
        self.aim.angles(self.observation,self.robot,intent(aim_assist=True))
        self.robot.update(aim=.6,pitch=.1)
        self.aim.angles(self.observation,self.robot,intent(2,active=False,yaw=.6,pitch=.1))
        yaw,pitch=self.aim.angles(self.observation,self.robot,intent(3,yaw=0,pitch=0))
        self.assertAlmostEqual(yaw,.6);self.assertAlmostEqual(pitch,.1)
        yaw,_=self.aim.angles(self.observation,self.robot,intent(4,yaw=.02))
        self.assertAlmostEqual(yaw,.62)

    def test_camera_loss_still_allows_queued_revival_without_combat(self):
        match=Match()
        try:
            robot=copy.deepcopy(next(r for r in match._frame['robots'] if r['id']=='red-3'))
            robot['alive']=False
            controller=HumanController(baseline_helpers())
            controller.operation('revive','',robot)
            controller.stop_combat()
            action=controller.action(match.compact_observation('red',[3]),robot,intent(active=False))
            self.assertTrue(action['revive']); self.assertFalse(action['fire'])
            self.assertEqual(controller.aim.state,'idle')
        finally:match.close()


class FireInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.room=Room(owner_key='combat',directory=self.temp.name,duration=3,ai=False,ceremony=False)
        self.addCleanup(self.room.close)
        self.token=self.room.join(Join(nickname='操作手',host_key='combat'))['token']
        self.room.attach(self.token); self.room.select(self.token,Seat(team='red',slot=7))
        self.room.status='running'; self.room.match_id='combat-test'
        self.player=self.room.player(self.token)

    def test_short_press_survives_release_and_duplicate_packets(self):
        self.room.submit(self.token,intent(fire=True,fire_press_seq=1))
        self.room.submit(self.token,intent(2,fire=False,fire_press_seq=1))
        self.room.submit(self.token,intent(2,fire=False,fire_press_seq=1))
        self.assertFalse(self.player.intent.fire)
        self.assertEqual(self.player.fire_presses,[1])

    def test_queue_bound_inactive_disconnect_and_resume_clear_input(self):
        self.room.submit(self.token,intent(fire_press_seq=100,fire=True,aim_assist=True))
        self.assertEqual(len(self.player.fire_presses),8)
        self.room.submit(self.token,intent(2,fire_press_seq=100,active=False))
        self.assertEqual(self.player.fire_presses,[])
        self.room.submit(self.token,intent(3,fire_press_seq=101,fire=True))
        self.room.detach(self.token,self.player.generation)
        self.assertEqual(self.player.fire_presses,[])
        self.room.attach(self.token)
        self.room.submit(self.token,intent(4,fire_press_seq=102,fire=True))
        self.assertTrue(self.player.needs_resume); self.assertEqual(self.player.fire_presses,[])

    def test_real_physics_short_press_fires_exactly_once_and_hold_repeats(self):
        self.room.status='lobby'
        self.room.ready(self.token,True); self.room.start(self.token)
        deadline=time.monotonic()+3
        while not self.room.snapshot(self.token)['player'] and time.monotonic()<deadline:time.sleep(.01)
        self.assertIsNotNone(self.room.snapshot(self.token)['player'])
        seq=0
        with self.room.lock:
            self.room.submit(self.token,Input(match_id=self.room.match_id,seq=seq,yaw=0,pitch=0,fire=True,fire_press_seq=1));seq+=1
            self.room.submit(self.token,Input(match_id=self.room.match_id,seq=seq,yaw=0,pitch=0,fire=False,fire_press_seq=1));seq+=1
        deadline=time.monotonic()+1
        while time.monotonic()<deadline:
            snapshot=self.room.snapshot(self.token)['player']
            if snapshot['control'].get('last_press_seq')==1:break
            time.sleep(.01)
        self.assertEqual(snapshot['self']['ammo'],299)
        self.assertEqual(snapshot['control']['last_press_result'],'已发射')
        for _ in range(5):
            self.room.submit(self.token,Input(match_id=self.room.match_id,seq=seq,yaw=0,pitch=0,fire=False,fire_press_seq=1));seq+=1
            time.sleep(.04)
        self.assertEqual(self.room.snapshot(self.token)['player']['self']['ammo'],299)
        for _ in range(12):
            self.room.submit(self.token,Input(match_id=self.room.match_id,seq=seq,yaw=0,pitch=0,fire=True,fire_press_seq=2));seq+=1
            time.sleep(.025)
        self.assertLess(self.room.snapshot(self.token)['player']['self']['ammo'],298)
        self.room.submit(self.token,Input(match_id=self.room.match_id,seq=seq,yaw=0,pitch=0,active=False,fire_press_seq=2))
        time.sleep(.1)
        stopped=self.room.snapshot(self.token)['player']['self']['ammo']
        time.sleep(.15)
        self.assertEqual(self.room.snapshot(self.token)['player']['self']['ammo'],stopped)

    def test_two_players_have_independent_fire_and_aim_state(self):
        other=self.room.join(Join(nickname='蓝方'))['token']
        self.room.attach(other); self.room.select(other,Seat(team='blue',slot=7))
        self.room.submit(self.token,intent(fire=True,fire_press_seq=1,aim_assist=True))
        self.room.submit(other,intent(fire=False,fire_press_seq=0,aim_assist=False))
        self.assertEqual(self.player.fire_presses,[1])
        self.assertEqual(self.room.player(other).fire_presses,[])
        self.assertTrue(self.player.intent.aim_assist)
        self.assertFalse(self.room.player(other).intent.aim_assist)

    def test_protocol_rejects_invalid_counters(self):
        for value in (-1,True,1.5,9007199254740992):
            with self.subTest(value=value),self.assertRaises(ValueError):intent(fire_press_seq=value)


class WeaponTests(unittest.TestCase):
    def test_initial_ammo_and_authoritative_reasons(self):
        match=Match()
        try:
            f=match._frame
            for slot,ammo,reason in ((1,0,'launcher'),(3,0,'no_ammo'),(4,0,'no_ammo'),(6,0,'hangar'),(7,300,'')):
                r=next(r for r in f['robots'] if r['id']==f'red-{slot}')
                self.assertEqual(r['ammo'],ammo);self.assertEqual(r['fire_block_reason'],reason)
                self.assertEqual(weapon_status(f,r)['reason'],reason)
            frame=match.step({'red':{'7':{'fire':True}}})
            r=next(r for r in frame['robots'] if r['id']=='red-7')
            self.assertEqual(r['ammo'],299);self.assertEqual(r['fire_block_reason'],'cooldown')
        finally:match.close()

    def test_blocked_press_is_consumed_without_delayed_retry(self):
        match=Match()
        try:
            robot=next(r for r in match._frame['robots'] if r['id']=='red-3')
            controller=HumanController(baseline_helpers())
            action=controller.action(match.compact_observation('red',[3]),robot,intent(fire=False),fire_press_seq=2)
            frame=match.step({'red':{'3':action}})
            robot=next(r for r in frame['robots'] if r['id']=='red-3')
            controller.observe(frame,robot)
            self.assertEqual(robot['ammo'],0)
            self.assertEqual(controller.last_press_seq,2)
            self.assertEqual(controller.last_press_result,'无弹丸，请按 B 购弹')
            for _ in range(5):
                action=controller.action(match.compact_observation('red',[3]),robot,intent(fire=False))
                self.assertFalse(action['fire'])
                frame=match.step({'red':{'3':action}})
                robot=next(r for r in frame['robots'] if r['id']=='red-3')
            self.assertFalse(controller._attempted_press)
        finally:match.close()

    def test_delivery_events_project_to_owner_only(self):
        match=Match()
        try:
            frame=copy.deepcopy(match._frame)
            frame['events']=[{'type':'ammo_delivered','robot':'red-3','amount':100},
                             {'type':'ammo_delivered','robot':'blue-3','amount':100}]
            view=project(frame,match.compact_observation('red',[3]),'red',3,'test')
            self.assertEqual(view['events'],[frame['events'][0]])
        finally:match.close()

if __name__=='__main__':unittest.main()
