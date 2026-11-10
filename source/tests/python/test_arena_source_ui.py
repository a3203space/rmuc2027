"""Public display data used by the ported Arena-1 components."""
import copy
import tempfile
import time
import unittest

from rmuc2027_core.engine import Match
from rmuc2027_server.projection import project
from rmuc2027_server.protocol import Join, Seat
from rmuc2027_server.room import Room


class SourceUiProjectionTests(unittest.TestCase):
    def test_damage_is_authoritative_and_enemy_weapon_state_stays_private(self):
        match = Match()
        try:
            frame = copy.deepcopy(match._frame)
            frame['teams']['red'].update(attack_damage=123.5, score=7)
            player = project(frame, match.compact_observation('red', [3]), 'red', 3, 'source-ui')
            self.assertEqual(player['tactical']['teams']['red']['attack_damage'], 123.5)
            self.assertEqual(player['tactical']['teams']['red']['score'], 7)
            for robot in player['tactical']['robots']:
                for field in ('heat', 'max_heat', 'level', 'drones', 'muzzle_speed_mps'):
                    self.assertNotIn(field, robot)
        finally:
            match.close()

    def test_final_summary_supplies_original_results_with_structures_and_models(self):
        with tempfile.TemporaryDirectory() as directory:
            room = Room(owner_key='source-ui', directory=directory, duration=1, ai=False, ceremony=False)
            try:
                token = room.join(Join(nickname='操作手', host_key='source-ui'))['token']
                room.attach(token)
                room.select(token, Seat(team='red', slot=3))
                room.ready(token, True)
                room.start(token)
                deadline = time.monotonic() + 8
                while room.status not in ('finished', 'closed') and time.monotonic() < deadline:
                    time.sleep(.02)
                self.assertIsNone(room.error)
                self.assertEqual(room.status, 'finished')
                result = room.state()['result']
                self.assertEqual(set(result['bases']), {'red', 'blue'})
                self.assertEqual(set(result['outposts']), {'red', 'blue'})
                self.assertEqual(len(result['positions']), 4)
                self.assertEqual(len(result['robots']), 10)
                self.assertNotIn('arena', result)  # Large static map stays in the static packet.
                for robot in result['robots']:
                    self.assertGreater(robot['level'], 0)
                    self.assertTrue(robot['model_asset'].endswith('.glb'))
                    self.assertIsInstance(robot['ammo'], int)
                for structure in result['bases'].values():
                    self.assertGreater(structure['max_hp'], 0)
            finally:
                room.close()


if __name__ == '__main__':
    unittest.main()
