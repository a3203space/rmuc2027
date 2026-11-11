"""Request rejection must precede initialization, submission, and drive updates.

All services are fake; ArenaMotion stops at initialization/upload sentinels.
No Warp initialization, kernels, native backend, or GPU are needed here.
"""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
from rmuc2027_research.rl.warp_kinematic.batching import BatchClient
from rmuc2027_research.rl.warp_kinematic.protocol import DT
from test_warp_batch_state_faults import RecordingService, request


MISSING = object()


def invalid_values(field, sequence):
    return {
        'dt': [float('nan'), float('inf'), -float('inf'), False, True, None,
               str(DT), [], {}, 0, -1., DT*2, 10**400, MISSING],
        'sequence': [False, True, float(sequence), -.5, -1, str(sequence), None,
                     [], {}, sequence+1, MISSING],
        'version': [False, True, 0, 3, 1., 2., '1', None, [], {}, MISSING],
        'substeps': [False, True, 0, -1, 5, 7, 6., '6', None, [], {}, MISSING],
        'op': [False, True, 1, '', 'kinematic_frame_result', None, [], {},
               b'kinematic_frame', MISSING],
    }[field]


def normal_request(version, sequence, compact=False):
    value = request(version=version)
    value['sequence'] = sequence
    if compact:
        value['arena'] = None
        value['robots'][0]['design'] = None
        value['robots'][0]['drive_parameters'] = {'motor_force_max_n': 81.}
    return value


def invalid_request(valid, field, value):
    bad = copy.deepcopy(valid)
    if value is MISSING:
        del bad[field]
    else:
        bad[field] = value
    bad['robots'][0]['drive_parameters'] = {'motor_force_max_n': 999.}
    return bad


class ReachedExecution(RuntimeError):
    pass


class BatchRequestIdentityTests(unittest.TestCase):
    def check_field(self, field):
        for version in (1, 2):
            for sequence, compact in ((0, False), (1, False), (1, True)):
                for value in invalid_values(field, sequence):
                    with self.subTest(field=field, value=value, version=version,
                                      sequence=sequence, compact=compact):
                        service = RecordingService()
                        client = BatchClient(service, 'cpu')
                        if sequence:
                            client.step(normal_request(version, 0))
                        valid = normal_request(version, sequence, compact)
                        fields = ('next_sequence', 'ids', 'version', 'body_signature',
                                  'scene_signature', 'design_signature', 'arena', 'designs')
                        before = {name: copy.deepcopy(getattr(client, name)) for name in fields}
                        designs = client.designs
                        arena = client.arena
                        with self.assertRaisesRegex(ValueError, 'Invalid motion request identity'):
                            client.step(invalid_request(valid, field, value))
                        self.assertEqual(len(service.calls), sequence)
                        self.assertEqual({name: getattr(client, name) for name in fields}, before)
                        self.assertIs(client.designs, designs)
                        self.assertIs(client.arena, arena)
                        result = client.step(valid)
                        self.assertEqual(result['sequence'], sequence)
                        self.assertEqual(result['version'], version)
                        self.assertEqual(client.next_sequence, sequence+1)
                        self.assertEqual(len(service.calls), sequence+1)

    def test_invalid_dt(self):
        self.check_field('dt')

    def test_invalid_sequence(self):
        self.check_field('sequence')

    def test_invalid_version(self):
        self.check_field('version')

    def test_invalid_substeps(self):
        self.check_field('substeps')

    def test_invalid_op(self):
        self.check_field('op')

    def test_boolean_sequence_rejected_by_all_step_variants_before_submission(self):
        for method in ('step', 'step_frame', 'step_encoded'):
            with self.subTest(method=method):
                service = RecordingService()
                client = BatchClient(service, 'cpu')
                bad = request()
                bad['sequence'] = False
                with self.assertRaisesRegex(ValueError, 'Invalid motion request identity'):
                    getattr(client, method)(bad)
                self.assertEqual(service.calls, [])
                self.assertEqual(client.next_sequence, 0)
                self.assertIsNone(client.ids)

    def test_valid_versions_full_compact_and_dt_tolerance(self):
        for version in (1, 2):
            service = RecordingService()
            client = BatchClient(service, 'cpu')
            for sequence, delta in enumerate((0., .5e-12, -.5e-12)):
                valid = normal_request(version, sequence, compact=sequence == 1)
                valid['dt'] += delta
                result = client.step(valid)
                self.assertEqual(result['sequence'], sequence)
                self.assertEqual(result['version'], version)
            self.assertEqual(client.next_sequence, 3)


class ArenaRequestIdentityTests(unittest.TestCase):
    def check_field(self, field):
        for version in (1, 2):
            for sequence, compact in ((0, False), (1, False), (1, True)):
                for value in invalid_values(field, sequence):
                    with self.subTest(field=field, value=value, version=version,
                                      sequence=sequence, compact=compact):
                        adapter = ArenaMotion('cpu')
                        try:
                            if sequence:
                                adapter.next_sequence = sequence
                                adapter.ids = ('a',)
                                adapter.extended = version == 2
                                adapter.designs = [copy.deepcopy(request()['robots'][0]['design'])]
                                adapter.motion = SimpleNamespace(limits=object(), motors=object())
                            designs = getattr(adapter, 'designs', None)
                            before = copy.deepcopy(designs)
                            motion = adapter.motion
                            ids = getattr(adapter, 'ids', None)
                            valid = normal_request(version, sequence, compact)
                            with patch.object(adapter, 'initialize', side_effect=ReachedExecution('initialize')) as initialize, \
                                 patch.object(adapter.host_uploads, 'assign', side_effect=ReachedExecution('upload')) as upload:
                                with self.assertRaisesRegex(ValueError, 'Invalid motion request identity'):
                                    adapter.step_frame(invalid_request(valid, field, value))
                                initialize.assert_not_called()
                                upload.assert_not_called()
                                self.assertEqual(adapter.next_sequence, sequence)
                                self.assertEqual(getattr(adapter, 'ids', None), ids)
                                self.assertIs(adapter.motion, motion)
                                self.assertIs(getattr(adapter, 'designs', None), designs)
                                self.assertEqual(getattr(adapter, 'designs', None), before)
                                with self.assertRaises(ReachedExecution):
                                    adapter.step_frame(valid)
                                if sequence:
                                    upload.assert_called_once()
                                else:
                                    initialize.assert_called_once_with(valid)
                        finally:
                            adapter.close()

    def test_invalid_dt(self):
        self.check_field('dt')

    def test_invalid_sequence(self):
        self.check_field('sequence')

    def test_invalid_version(self):
        self.check_field('version')

    def test_invalid_substeps(self):
        self.check_field('substeps')

    def test_invalid_op(self):
        self.check_field('op')

    def test_valid_versions_and_dt_tolerance_reach_initialization(self):
        for version in (1, 2):
            for delta in (0., .5e-12, -.5e-12):
                adapter = ArenaMotion('cpu')
                try:
                    valid = normal_request(version, 0)
                    valid['dt'] += delta
                    with patch.object(adapter, 'initialize', side_effect=ReachedExecution) as initialize:
                        with self.assertRaises(ReachedExecution):
                            adapter.step_frame(valid)
                        initialize.assert_called_once_with(valid)
                finally:
                    adapter.close()


if __name__ == '__main__':
    unittest.main()
