"""Actual physics bridges: per-seat capabilities and reused Warp worlds, no RL."""
import copy
import importlib.util
import math
import unittest

from rmuc2027_core.engine import Match, binary_path
from rmuc2027_core.models import robot_models
from rmuc2027_research.rl.capabilities import AIM_PROFILES, MOBILITY_PROFILES, CapabilitySampler, SlowCapabilityDrift
from rmuc2027_research.rl.scenarios import ScenarioSpec

WARP_AVAILABLE = all(importlib.util.find_spec(name) for name in ('warp', 'numpy'))
MJ_AVAILABLE = WARP_AVAILABLE and all(importlib.util.find_spec(name) for name in ('mujoco', 'mujoco_warp'))


def profile(name):
    return dict(mobility=copy.deepcopy(MOBILITY_PROFILES[name]), aim=copy.deepcopy(AIM_PROFILES['learner']))


def lineups():
    return ScenarioSpec((303, 303), (303, 303)).lineups()


def wire_lineups(roster=None):
    return {team:[dict(slot=int(slot),model=model) for slot,model in values.items()]
            for team,values in (roster or lineups()).items()}


def motion_request(name='nominal', sequence=0, include_design=True):
    from rmuc2027_research.rl.warp_kinematic.protocol import DT
    nominal = robot_models()['303']
    drive = {key: nominal[key]*value for key, value in MOBILITY_PROFILES[name].items()}
    robot = dict(id='red-3', model=303, team='red', role='infantry', pos=[4., 4., 0.],
                 velocity=[0., 0., 0.], heading=0., folded=0., grounded=True, enabled=True,
                 alive=True, radius=.25, power_limit=100., motor_budget=100., move=[.8, 0.],
                 heading_goal=0., yaw_rate=None, transform=False, jump=0., coast=False,
                 quaternion=[0., 0., 0., 1.], angular_velocity=[0., 0., 0.],
                 design={**nominal, **drive} if include_design else None, drive_parameters=drive)
    return dict(op='kinematic_frame', version=2, sequence=sequence, dt=DT, substeps=6,
                time=sequence*.05, robots=[robot],
                arena=dict(width=10., height=10., terrain=[], obstacles=[], forbidden=[]))


def warp_devices():
    import warp as wp
    wp.init()
    if not wp.is_cuda_available():
        raise unittest.SkipTest('Warp locomotion requires CUDA; no CPU fallback')
    return ['cuda:0']


class RecordingMotion:
    protocol = 'kinematic_frame'

    def __init__(self, device):
        from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
        self.engine = ArenaMotion(device); self.requests = []

    def step(self, request):
        self.requests.append(copy.deepcopy(request))
        return self.engine.step(request)

    def close(self):
        self.engine.close()


class RapierCapabilityTests(unittest.TestCase):
    def test_actual_match_drifts_enemies_after_fifteen_seconds_only(self):
        binary_path()
        roster = lineups(); profiles = CapabilitySampler(53).sample(roster, 'red')
        expected = SlowCapabilityDrift(profiles, 67+104729, learner_side='red')
        with Match(seed=67, duration=20, lineups=wire_lineups(roster), training_capabilities=profiles,
                   battle_settings={'drones_enabled':False, 'timed_assembly':True}) as match:
            before = match.snapshot()
            for tick in range(321):
                time_before = match.time
                result = match.step({})
                self.assertEqual(match.training_capabilities, expected.update(time_before))
            self.assertNotEqual(match.training_capabilities['blue:3']['mobility'], profiles['blue:3']['mobility'])
            for seat in ('red:1', 'red:3', 'red:4', 'red:7'):
                self.assertEqual(match.training_capabilities[seat], profiles[seat])
            self.assertEqual(match.training_capabilities_initial, profiles)
            # Ask Rust for a fresh snapshot rather than checking Python's static cache.
            actual = match._request({'op':'snapshot'})
            self.assertEqual(actual['robot_models'], before['robot_models'])
            self.assertEqual({r['id']:r['power_limit'] for r in actual['robots']},
                             {r['id']:r['power_limit'] for r in before['robots']})
            self.assertTrue(all(math.isfinite(r[k]) for r in result['robots'] for k in ('x','y','z','vx','vy','vz')))


@unittest.skipUnless(WARP_AVAILABLE, 'Optional Warp dependencies unavailable')
class WarpCapabilityTests(unittest.TestCase):
    def test_cpu_is_explicitly_rejected_without_fallback(self):
        from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
        engine = ArenaMotion('cpu')
        try:
            with self.assertRaisesRegex(ValueError, 'requires CUDA'):
                engine.step(motion_request())
        finally:
            engine.close()

    def test_rust_nominal_multipliers_seat_isolation_and_public_limits(self):
        nominal_catalog = robot_models()
        for device in warp_devices():
            with self.subTest(device=device):
                adapter = RecordingMotion(device)
                profiles = {'red:3':profile('low'), 'red:4':profile('high')}
                with Match(lineups=wire_lineups(), training_capabilities=profiles, physics_adapter=adapter,
                           battle_settings={'drones_enabled':False, 'timed_assembly':True}) as match:
                    initial = match.snapshot()
                    for names in (('low','high'), ('low','high'), ('high','low'), ('nominal','nominal')):
                        match.training_capabilities = {'red:3':profile(names[0]), 'red:4':profile(names[1])}
                        match.step({'red':{'3':{'move':[.8,0.]}, '4':{'move':[.8,0.]}}})
                        request = adapter.requests[-1]
                        robots = {r['id']:r for r in request['robots']}
                        for seat, name in (('red-3',names[0]), ('red-4',names[1]), ('blue-3','nominal')):
                            for key, multiplier in MOBILITY_PROFILES[name].items():
                                self.assertAlmostEqual(robots[seat]['drive_parameters'][key], nominal_catalog['303'][key]*multiplier)
                        for r in request['robots']:
                            original = next(v for v in initial['robots'] if v['id']==r['id'])
                            self.assertEqual(r['power_limit'], original['power_limit'])
                    actual = match._request({'op':'snapshot'})
                    self.assertEqual(actual['robot_models'], initial['robot_models'])
                    self.assertEqual(robot_models(), nominal_catalog)
                    self.assertIsNone(adapter.requests[-1]['robots'][0]['design'])

    def test_topology_ignores_only_the_uploaded_drive_scalars(self):
        from rmuc2027_research.rl.warp_kinematic.batching import topology_signature
        low, high = motion_request('low'), motion_request('high')
        self.assertEqual(topology_signature(low['robots']), topology_signature(high['robots']))
        high['robots'][0]['design']['mass_kg'] += 1
        self.assertNotEqual(topology_signature(low['robots']), topology_signature(high['robots']))

    def test_reused_world_refreshes_all_drive_uploads_every_frame(self):
        import numpy as np
        from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
        nominal = robot_models()['303']
        for device in warp_devices():
            with self.subTest(device=device):
                engine = ArenaMotion(device)
                speeds = []
                try:
                    identity = None
                    for sequence, name in enumerate(('low','high','low')):
                        request = motion_request(name, sequence, include_design=sequence==0)
                        frame = engine.step(request)
                        if identity is None: identity = engine.motion
                        self.assertIs(engine.motion, identity)
                        expected = request['robots'][0]['drive_parameters']
                        np.testing.assert_allclose(engine.motion.limits.numpy()[0,0], expected['speed_reference_mps'], rtol=1e-6)
                        np.testing.assert_allclose(engine.motion.motors.numpy()[0,1], expected['motor_force_max_n'], rtol=1e-6)
                        np.testing.assert_allclose(engine.motion.drive.numpy()[0,1], expected['drive_efficiency'], rtol=1e-6)
                        self.assertAlmostEqual(float(engine.motion.motors.numpy()[0,0]), nominal['velocity_gain_per_s'])
                        speeds.append(frame['velocities'][-1][0][0])
                    self.assertGreater(speeds[1], speeds[0])
                    self.assertAlmostEqual(speeds[0], speeds[2], places=6)
                finally:
                    engine.close()

    def test_two_world_reordering_and_profile_changes_match_isolated_worlds(self):
        import numpy as np
        from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
        from rmuc2027_research.rl.warp_kinematic.batching import BatchClient, WorldBatch
        for device in warp_devices():
            with self.subTest(device=device):
                batch = WorldBatch(device)
                clients = [BatchClient(None,device), BatchClient(None,device)]
                solo = [ArenaMotion(device), ArenaMotion(device)]
                for client in clients: client.ids = ('red-3',)
                try:
                    identity = None
                    for sequence, names in enumerate((('low','high'), ('high','low'), ('nominal','high'), ('low','nominal'))):
                        requests = [motion_request(name, sequence, include_design=sequence==0) for name in names]
                        if sequence == 0:
                            for client, request in zip(clients, requests):
                                client.designs = [copy.deepcopy(r['design']) for r in request['robots']]
                                client.arena = request['arena']
                        order = (0,1) if sequence%2==0 else (1,0)
                        frames = batch.step_many([(clients[i], requests[i]) for i in order])
                        if identity is None: identity = batch.engine.motion
                        self.assertIs(batch.engine.motion, identity)
                        for i, actual in zip(order, frames):
                            expected = solo[i].step(copy.deepcopy(requests[i]))
                            for key in ('positions','velocities','rotations','angular_velocities','grounded'):
                                np.testing.assert_allclose(actual[key], expected[key], atol=1e-6, err_msg=str((device,sequence,i,key)))
                finally:
                    batch.close()
                    for item in clients+solo: item.close()


@unittest.skipUnless(MJ_AVAILABLE, 'Optional MuJoCo Warp dependencies unavailable')
class MJCapabilityTests(unittest.TestCase):
    def test_real_mj_bridge_accepts_dynamic_capabilities(self):
        from rmuc2027_research.rl.mjwarp.arena import ArenaPhysics
        adapter = ArenaPhysics('cpu')
        with Match(lineups=wire_lineups(), training_capabilities={'red:3':profile('low')}, physics_adapter=adapter,
                   battle_settings={'drones_enabled':False, 'timed_assembly':True}) as match:
            initial = match.snapshot()
            match.step({'red':{'3':{'move':[.5,0.]}}})
            match.training_capabilities = {'red:3':profile('high')}
            result = match.step({'red':{'3':{'move':[.5,0.]}}})
            self.assertTrue(all(math.isfinite(r[k]) for r in result['robots'] for k in ('x','y','z','vx','vy','vz')))
            actual = match._request({'op':'snapshot'})
            self.assertEqual(actual['robot_models'], initial['robot_models'])
            self.assertEqual({r['id']:r['power_limit'] for r in actual['robots']},
                             {r['id']:r['power_limit'] for r in initial['robots']})


if __name__ == '__main__':
    unittest.main()
