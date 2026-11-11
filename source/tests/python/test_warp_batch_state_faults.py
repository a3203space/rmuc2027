"""Real CPU state-restore kernels and fake integration; no GPU or native Match."""
import copy
from concurrent.futures import Future
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from rmuc2027_research.rl.warp_kinematic import batching
from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
from rmuc2027_research.rl.warp_kinematic.protocol import DT, MotionFrame


def request(ids=('a',), version=2, scene=20.):
    return dict(op='kinematic_frame', version=version, sequence=0, dt=DT, substeps=6,
                arena=dict(width=scene, height=20., terrain=[], obstacles=[]),
                robots=[dict(id=name, role='infantry', radius=.25, pos=[float(i+2), 3., 0.],
                             velocity=[0., 0., 0.], grounded=True,
                             design=dict(mass_kg=25., body_height_m=.4, leg_nominal_m=.1,
                                         speed_reference_mps=2., motor_force_max_n=80., drive_efficiency=.8))
                        for i, name in enumerate(ids)])


def frame_for(value):
    n = len(value['robots'])
    positions = np.tile(np.asarray([r['pos'] for r in value['robots']]), (6, 1, 1))
    args = [tuple(r['id'] for r in value['robots']), positions, np.zeros((6, n, 3)),
            np.zeros((6, n)), np.zeros((6, n)), np.ones((6, n), dtype=bool)]
    if value['version'] == 2:
        args.extend([np.tile([0., 0., 0., 1.], (6, n, 1)), np.zeros((6, n, 3)), np.zeros((6, n, 2))])
    return MotionFrame(*args)


class RecordingService:
    def __init__(self):
        self.calls = []

    def submit(self, client, value):
        self.calls.append(dict(version=value['version'], scene_signature=client.scene_signature,
                               design_signature=client.design_signature,
                               arena=copy.deepcopy(client.arena), designs=copy.deepcopy(client.designs)))
        return frame_for(value)


class FakeArena:
    """Keep validation/state transport real, while replacing all integration."""
    instances = []

    def __init__(self, device, carry_state=False):
        assert device == 'cpu'
        assert carry_state
        self.device = device
        self.motion = None
        self.closed = False
        self.next_sequence = 0
        self.expected_positions = None
        self.inputs = []
        self.instances.append(self)

    def initialize(self, value):
        self.ids = tuple(r['id'] for r in value['robots'])
        self.extended = value['version'] == 2
        self.arena = copy.deepcopy(value['arena'])
        n = len(self.ids)

        def zero(dtype=float):
            return wp.zeros(n, dtype=dtype, device='cpu')

        self.jump_start, self.jump_end = zero(wp.vec3), zero(wp.vec3)
        self.jump_elapsed, self.jump_duration, self.jump_cooldown = zero(), zero(), zero()
        self.jump_status, self.fall_speed = zero(wp.int32), zero()
        self.base, self.leg_command = zero(), zero()
        self.jump_latched, self.jump_strength = zero(), zero()
        self.motion = SimpleNamespace(flight_z=zero(), height_offsets=zero(wp.vec2), flight=zero(wp.int32))
        self.terrain = SimpleNamespace(heights=lambda positions: np.zeros(len(positions)))

    def step_frame(self, value):
        # On an identity mismatch, exercise the actual production validator.
        if (value['version'] == 2) != self.extended or tuple(r['id'] for r in value['robots']) != self.ids:
            return ArenaMotion.step_frame(self, value)
        self.inputs.append(copy.deepcopy(value))
        n = len(self.ids)
        state = np.zeros((n, 19 if self.extended else 17), dtype=np.float32)
        state[:, :3], state[:, 3:6] = self.jump_start.numpy(), self.jump_end.numpy()
        for column, array in ((6, self.jump_elapsed), (7, self.jump_duration), (8, self.jump_cooldown),
                              (9, self.jump_status), (10, self.fall_speed), (11, self.motion.flight_z),
                              (14, self.base), (15, self.motion.flight), (16, self.leg_command)):
            state[:, column] = array.numpy()
        state[:, 12:14] = self.motion.height_offsets.numpy()
        if self.extended:
            state[:, 17], state[:, 18] = self.jump_latched.numpy(), self.jump_strength.numpy()
        self.last_state = state
        self.next_sequence += 1
        return frame_for(value)

    def close(self):
        self.closed = True


class BatchIdentityTests(unittest.TestCase):
    def client(self, value):
        service = RecordingService()
        client = batching.BatchClient(service, 'cpu')
        client.step_frame(value)
        return client, service

    def test_protocol_change_rejected_before_service_and_client_remains_valid(self):
        for version in (1, 2):
            with self.subTest(version=version):
                first = request(version=version)
                client, service = self.client(first)
                bad = copy.deepcopy(first)
                bad.update(sequence=1, version=3-version)
                with self.assertRaisesRegex(ValueError, 'protocol'):
                    client.step_frame(bad)
                self.assertEqual(len(service.calls), 1)
                self.assertEqual(client.next_sequence, 1)
                first['sequence'] = 1
                client.step_frame(first)

    def test_scene_change_rejected_before_service(self):
        first = request()
        client, service = self.client(first)
        later = copy.deepcopy(first)
        later['sequence'] = 1
        later['arena']['width'] = 40.
        with self.assertRaisesRegex(ValueError, 'scene|Scene'):
            client.step_frame(later)
        self.assertEqual(len(service.calls), 1)

    def test_static_topology_change_rejected_in_full_and_compact_requests(self):
        for field, value in (('radius', .5), ('role', 'aerial'), ('mass_kg', 90.)):
            for compact in (False, True):
                if field == 'mass_kg' and compact:
                    continue
                with self.subTest(field=field, compact=compact):
                    first = request()
                    client, service = self.client(first)
                    later = copy.deepcopy(first)
                    later['sequence'] = 1
                    if compact:
                        later['arena'] = later['robots'][0]['design'] = None
                    if field == 'mass_kg':
                        later['robots'][0]['design'][field] = value
                    else:
                        later['robots'][0][field] = value
                    with self.assertRaisesRegex(ValueError, 'topology|Topology'):
                        client.step_frame(later)
                    self.assertEqual(len(service.calls), 1)

    def test_robot_reordering_rejected_before_service(self):
        first = request(ids=('a', 'b'))
        client, service = self.client(first)
        first['sequence'] = 1
        first['robots'].reverse()
        with self.assertRaisesRegex(ValueError, 'Robot order'):
            client.step_frame(first)
        self.assertEqual(len(service.calls), 1)

    def test_full_compact_full_drive_uploads_and_override_precedence(self):
        first = request()
        client, service = self.client(first)
        full = copy.deepcopy(first)
        full['sequence'] = 1
        full['robots'][0]['design']['motor_force_max_n'] = 120.
        client.step_frame(full)
        self.assertEqual(service.calls[-1]['designs'][0]['motor_force_max_n'], 120.)
        compact = copy.deepcopy(full)
        compact.update(sequence=2, arena=None)
        compact['robots'][0].update(design=None, drive_parameters={'motor_force_max_n': 60.})
        client.step_frame(compact)
        self.assertEqual(service.calls[-1]['designs'][0]['motor_force_max_n'], 60.)
        full['sequence'] = 3
        full['robots'][0]['drive_parameters'] = {'motor_force_max_n': 90.}
        client.step_frame(full)
        self.assertEqual(service.calls[-1]['designs'][0]['motor_force_max_n'], 90.)
        self.assertEqual(len({entry['design_signature'] for entry in service.calls}), 1)

    def test_initial_scene_and_design_are_owned_snapshots(self):
        first = request()
        client, _ = self.client(first)
        first['arena']['width'] = 99.
        first['robots'][0]['design']['mass_kg'] = 99.
        self.assertEqual(client.arena['width'], 20.)
        self.assertEqual(client.designs[0]['mass_kg'], 25.)

    def test_invalid_initial_compact_request_does_not_poison_client(self):
        service = RecordingService()
        client = batching.BatchClient(service, 'cpu')
        bad = request()
        bad['arena'] = bad['robots'][0]['design'] = None
        with self.assertRaises(ValueError):
            client.step_frame(bad)
        self.assertEqual(len(service.calls), 0)
        self.assertIsNone(client.ids)
        self.assertEqual(client.next_sequence, 0)
        client.step_frame(request())


class BatchStateTests(unittest.TestCase):
    def setUp(self):
        FakeArena.instances = []
        fake = patch.object(batching, 'ArenaMotion', FakeArena)
        fake.start()
        self.addCleanup(fake.stop)

    def batch(self):
        batch = batching.WorldBatch('cpu')
        self.addCleanup(batch.close)
        return batch

    def client(self, value):
        client = batching.BatchClient(RecordingService(), 'cpu')
        client.step_frame(value)
        return client

    def test_cache_reuse_with_different_client_ids_keeps_external_ids(self):
        batch = self.batch()
        for names in (('a', 'b'), ('x', 'y'), ('b', 'a')):
            value = request(ids=names)
            client = self.client(value)
            result = batch.step_many([(client, value)], frames=True)[0]
            self.assertEqual(result.ids, names)
            np.testing.assert_array_equal(result.positions[-1], [r['pos'] for r in value['robots']])

    def test_client_state_follows_world_order_and_teleport_resets_only_target(self):
        for version in (1, 2):
            with self.subTest(version=version):
                batch = self.batch()
                values = [request(version=version), request(version=version)]
                clients = [self.client(value) for value in values]
                width = 19 if version == 2 else 17
                for i, (client, value) in enumerate(zip(clients, values)):
                    client.state = np.arange(width, dtype=np.float32).reshape(1, width) + 20*i
                    client.expected_positions = np.asarray([r['pos'] for r in value['robots']])
                before = [client.state.copy() for client in clients]
                batch.step_many([(clients[1], values[1]), (clients[0], values[0])])
                for expected, client in zip(before, clients):
                    np.testing.assert_array_equal(client.state, expected)
                values[0]['robots'][0]['pos'] = [7., 8., 2.]
                batch.step_many(list(zip(clients, values)))
                cleared = [6, 8, 9, 10, 15] + ([17, 18] if version == 2 else [])
                np.testing.assert_array_equal(clients[0].state[0, cleared], 0.)
                self.assertEqual(clients[0].state[0, 12], 2.)
                self.assertEqual(clients[0].state[0, 14], 2.)
                self.assertAlmostEqual(clients[0].state[0, 16], .1)
                np.testing.assert_array_equal(clients[1].state, before[1])
                self.assertFalse(np.shares_memory(clients[0].state, clients[1].state))

    def test_fresh_client_in_reused_slot_has_no_prior_motion_state(self):
        batch = self.batch()
        value = request()
        old = self.client(value)
        old.state = np.full((1, 19), 7., dtype=np.float32)
        old.expected_positions = np.asarray([r['pos'] for r in value['robots']])
        batch.step_many([(old, value)])
        old.close()
        self.assertIsNone(old.state)
        self.assertIsNone(old.expected_positions)
        fresh = self.client(value)
        batch.step_many([(fresh, value)])
        np.testing.assert_array_equal(fresh.state[0, [6, 7, 8, 9, 10, 15, 17, 18]], 0.)

    def test_service_cache_separates_scene_protocol_and_static_topology(self):
        service = batching.PhysicsService(1)
        self.addCleanup(service.close)
        values = [request(version=1), request(version=2), request(version=2, scene=30.), request(version=2)]
        values[-1]['robots'][0]['radius'] = .4
        for value in values:
            client = batching.BatchClient(service, 'cpu')
            result = client.step_frame(value)
            self.assertEqual(result.response(0)['version'], value['version'])
        self.assertEqual(len(service.engines), 4)
        self.assertEqual(len(FakeArena.instances), 4)
        self.assertEqual([engine.arena['width'] for engine in FakeArena.instances], [20., 20., 30., 20.])

    def test_service_reuses_topology_for_different_names_and_drive_scalars(self):
        service = batching.PhysicsService(1)
        self.addCleanup(service.close)
        for i in range(3):
            value = request(ids=(f'robot-{i}',))
            value['robots'][0]['design']['motor_force_max_n'] = 80. + i
            client = batching.BatchClient(service, 'cpu')
            result = client.step_frame(value)
            self.assertEqual(result.ids, (f'robot-{i}',))
        self.assertEqual(len(service.engines), 1)
        self.assertEqual(len(FakeArena.instances), 1)

    def test_service_multiworld_order_reuses_cache_without_cross_client_state(self):
        service = batching.PhysicsService(2)
        self.addCleanup(service.close)
        values = [request(ids=('a',)), request(ids=('b',))]
        clients = [self.client(value) for value in values]
        saved = []
        for i, (client, value) in enumerate(zip(clients, values)):
            client.state = np.full((1, 19), 3. + i, dtype=np.float32)
            client.expected_positions = np.asarray([r['pos'] for r in value['robots']])
            saved.append(client.state.copy())
        for order in ((0, 1), (1, 0)):
            service.begin_round(2)
            futures = []
            for i in order:
                future = Future()
                service.queue.put((clients[i], values[i], future))
                futures.append(future)
            service.worker_done()
            service.worker_done()
            for i, future in zip(order, futures):
                result = future.result(timeout=2.)
                self.assertEqual(result.ids, clients[i].ids)
                np.testing.assert_array_equal(clients[i].state, saved[i])
        self.assertEqual(len(service.engines), 1)
        self.assertEqual(len(FakeArena.instances), 1)


if __name__ == '__main__':
    unittest.main()
