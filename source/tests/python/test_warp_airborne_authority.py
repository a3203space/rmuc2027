"""Authority restoration only: fake integration and explicit CPU kernels."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from rmuc2027_core.sdk import DESIGNS
from rmuc2027_research.rl.warp_kinematic import arena, batching
from rmuc2027_research.rl.warp_kinematic.pose import parameters, resting_origin
from rmuc2027_research.rl.warp_kinematic.protocol import DT
from rmuc2027_research.rl.warp_kinematic.terrain import TerrainData
from rmuc2027_research.rl.warp_kinematic.uploads import HostUploads
from test_warp_batch_state_faults import FakeArena, RecordingService, request
from test_warp_jump_strength import JumpState


class BoundaryReached(RuntimeError):
    pass


class BoundaryUploads(HostUploads):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner

    def assign(self, target, values):
        if target is self.owner.support:
            raise BoundaryReached()
        super().assign(target, values)


class AuthorityArena(FakeArena):
    """Run actual input preparation, stopping before integration parameters."""
    def __init__(self, device='cpu', carry_state=True):
        super().__init__(device, carry_state)
        self.support = object()
        self.host_uploads = BoundaryUploads(self)

    def initialize(self, value):
        super().initialize(value)
        robots = value['robots']
        self.designs = [copy.deepcopy(r['design']) for r in robots]
        n = len(robots)

        def zero(dtype=float):
            return wp.zeros(n, dtype=dtype, device='cpu')

        m = self.motion
        m.device, m.count = 'cpu', n
        m.pos, m.vel = zero(wp.vec2), zero(wp.vec2)
        m.limits, m.motors = zero(wp.vec3), zero(wp.vec4)
        m.enabled, m.coast = zero(wp.int32), zero(wp.int32)
        self.previous_velocity, self.requested, self.targets = (zero(wp.vec2) for _ in range(3))
        self.tilt, self.flags = zero(wp.vec4), zero(wp.vec3i)
        self.heading, self.fold, self.yaw_velocity = zero(), zero(), zero()
        self.jump_command, self.last_z, self.leg_goal, self.passive_vz = (zero() for _ in range(4))
        self.stance = wp.array([parameters(d)[0] for d in self.designs], dtype=wp.vec4, device='cpu')
        self.ground_agents = wp.array([int(r['role'] != 'aerial') for r in robots], dtype=wp.int32, device='cpu')
        data = TerrainData()
        data.width = data.height = 20.
        data.floor_slope = 0.
        for name, dtype in (('bounds', wp.vec4), ('ranges', wp.vec2i), ('vertices', wp.vec2),
                            ('profiles', wp.vec4), ('ramps', wp.vec4), ('kinds', wp.int32), ('edge_ramps', float)):
            setattr(data, name, wp.empty(0, dtype=dtype, device='cpu'))
        self.terrain.data = data

    def restore(self, state):
        wp.launch(batching.restore_state, len(state), inputs=[
            wp.array(state, dtype=float, device='cpu'), self.jump_start, self.jump_end,
            self.jump_elapsed, self.jump_duration, self.jump_cooldown, self.jump_status,
            self.fall_speed, self.motion.flight_z, self.motion.height_offsets, self.base,
            self.motion.flight, self.leg_command, self.jump_latched, self.jump_strength], device='cpu')

    def step_frame(self, value):
        try:
            arena.ArenaMotion.step_frame(self, value)
        except BoundaryReached:
            pass
        else:
            raise AssertionError('input preparation did not stop before simulation')
        return super().step_frame(value)

    def close(self):
        self.host_uploads.close()
        super().close()


def command(name='a', version=2, enabled=True, grounded=False, vz=-1.25, role='infantry'):
    value = request(ids=(name,), version=version)
    value['robots'][0].update(design=copy.deepcopy(DESIGNS['2772']),
                              pos=[4., 4., 2.], velocity=[.4, 0., vz], enabled=enabled,
                              grounded=grounded, role=role, jump=0., heading=0., folded=0.,
                              move=[0., 0.], transform=False, power_limit=100., team='red')
    return value


def client(value):
    item = batching.BatchClient(RecordingService(), 'cpu')
    item.step_frame(value)
    return item


class AirborneAuthorityTests(unittest.TestCase):
    def setUp(self):
        fake = patch.object(batching, 'ArenaMotion', AuthorityArena)
        fake.start()
        self.addCleanup(fake.stop)
        # Even an accidental fall-through must never run a full adapter step.
        stop = patch.object(arena.ArenaMotion, '_simulate', side_effect=AssertionError('full simulation forbidden'))
        stop.start()
        self.addCleanup(stop.stop)

    def batch(self):
        item = batching.WorldBatch('cpu')
        self.addCleanup(item.close)
        return item

    def cold(self, value):
        item = client(value)
        self.batch().step_many([(item, value)])
        return item.state.copy()

    @staticmethod
    def old_state(cold):
        old = cold.copy()
        old[:, 6] = .07
        old[:, 8] = .6
        old[:, 9] = 1.
        old[:, 10] = 3.
        old[:, 11] = .75
        old[:, 15] = 1.
        if old.shape[1] == 19:
            old[:, 17:19] = [.8, .4]
        return old

    def warm_batch(self, value, old, changed=True):
        item = client(value)
        item.state = old.copy()
        item.expected_positions = np.asarray([r['pos'] for r in value['robots']], dtype=float)
        if changed:
            item.expected_positions[:, 0] -= .2
        self.batch().step_many([(item, value)])
        return item.state.copy()

    def direct(self, value, old, changed=True):
        engine = AuthorityArena()
        self.addCleanup(engine.close)
        engine.initialize(value)
        engine.restore(old)
        engine.expected_positions = np.asarray([r['pos'] for r in value['robots']], dtype=float)
        if changed:
            engine.expected_positions[0, 0] -= .2
        engine.step_frame(value)
        return engine.last_state.copy()

    def test_powered_airborne_correction_matches_fresh_ballistic_state(self):
        for vz in (-1.25, 1.5):
            with self.subTest(vz=vz):
                value = command(vz=vz)
                cold = self.cold(value)
                old = self.old_state(cold)
                warm = self.warm_batch(value, old)
                direct = self.direct(value, old)
                active = [6, 8, 9, 10, 11, 15, 17, 18]
                np.testing.assert_allclose(warm[:, active], cold[:, active])
                np.testing.assert_array_equal(direct, warm)
                self.assertEqual(warm[0, 9], 4.)
                self.assertEqual(warm[0, 15], 1.)
                self.assertAlmostEqual(warm[0, 14], resting_origin(value['robots'][0]['design']))
                expected_vz = vz - 9.81*DT
                expected_z = 2. + expected_vz*DT
                for restored in (cold, warm, direct):
                    state = JumpState([0.])
                    state.restore(restored)
                    state.step()
                    self.assertAlmostEqual(float(state.vz.numpy()[0]), expected_vz, places=5)
                    self.assertAlmostEqual(float(state.z.numpy()[0]), expected_z, places=5)
                    self.assertEqual(int(state.flight.numpy()[0]), 1)

    def test_correction_does_not_change_second_world(self):
        values = [command('a'), command('b', vz=.2)]
        cold = [self.cold(value) for value in values]
        old = [self.old_state(item) for item in cold]
        items = [client(value) for value in values]
        for i, (item, value) in enumerate(zip(items, values)):
            item.state = old[i].copy()
            item.expected_positions = np.asarray([value['robots'][0]['pos']], dtype=float)
        items[0].expected_positions[:, 0] -= .2
        self.batch().step_many(list(zip(items, values)))
        np.testing.assert_array_equal(items[1].state, old[1])
        combined = copy.deepcopy(values[0])
        combined['robots'] += copy.deepcopy(values[1]['robots'])
        direct = self.direct(combined, np.concatenate(old))
        np.testing.assert_array_equal(direct, np.concatenate([item.state for item in items]))
        self.assertEqual(items[0].state[0, 15], 1.)
        self.assertEqual(items[0].state[0, 11], 2.)

    def test_grounded_correction_keeps_existing_reset_semantics(self):
        value = command(grounded=True)
        old = self.old_state(self.cold(value))
        warm = self.warm_batch(value, old)
        direct = self.direct(value, old)
        np.testing.assert_array_equal(direct, warm)
        np.testing.assert_array_equal(warm[0, [6, 8, 9, 10, 15, 17, 18]], 0.)
        self.assertEqual(warm[0, 14], 2.)

    def test_disabled_airborne_correction_preserves_passive_velocity_rule(self):
        for vz in (-1.25, 1.5):
            with self.subTest(vz=vz):
                value = command(enabled=False, vz=vz)
                old = self.old_state(self.cold(value))
                warm = self.warm_batch(value, old)
                direct = self.direct(value, old)
                np.testing.assert_array_equal(direct, warm)
                self.assertEqual(warm[0, 9], 4.)
                self.assertEqual(warm[0, 15], 1.)
                self.assertEqual(warm[0, 11], 2.)
                self.assertEqual(warm[0, 10], min(0., vz))

    def test_unchanged_powered_flight_keeps_carried_state(self):
        value = command()
        old = self.old_state(self.cold(value))
        np.testing.assert_array_equal(self.warm_batch(value, old, changed=False), old)
        np.testing.assert_array_equal(self.direct(value, old, changed=False), old)

    def test_v1_and_aerial_keep_existing_reset_semantics(self):
        for value in (command(version=1), command(role='aerial')):
            with self.subTest(version=value['version'], role=value['robots'][0]['role']):
                old = self.old_state(self.cold(value))
                warm = self.warm_batch(value, old)
                direct = self.direct(value, old)
                np.testing.assert_array_equal(direct, warm)
                np.testing.assert_array_equal(warm[0, [6, 8, 9, 10, 15]], 0.)


if __name__ == '__main__':
    unittest.main()
