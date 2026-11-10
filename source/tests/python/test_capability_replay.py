"""Executor/replay integration checks; no policy training or optimizer steps."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from rmuc2027_core.engine import Match
from rmuc2027_core.policy import EncodedObservation
from rmuc2027_core.replay import verify_replay
from rmuc2027_research.rl.capabilities import CapabilitySampler
from rmuc2027_research.rl.session import resource
from rmuc2027_research.strategies.battle import Battle


LINEUPS = {team: {'1': 2711, '3': 2731, '4': 2732, '6': 2761, '7': 2773}
           for team in ('red', 'blue')}


def profiles():
    value = CapabilitySampler(731).sample(LINEUPS, 'red')
    value['blue:3']['aim'].update(delay_s=.15, sigma_deg=2.5, tau_s=.15)
    return value


class PulseController:
    disabled = False
    control_hz = 20
    is_learning_controller = False
    consecutive_timeouts = 0
    logs = ()

    def __init__(self, team, slot):
        self.team, self.slot = team, slot
        self.calls = 0

    def decide(self, observation):
        if isinstance(observation, EncodedObservation): observation = json.loads(observation.payload)
        self.calls += 1
        if self.team == 'blue' and self.slot == 3:
            command = {'aim': .55, 'pitch': .1, 'fire': True} if self.calls == 1 else {'fire': True}
        else:
            command = {}
        return {str(self.slot): command}, None

    def close(self): self.disabled = True


class CapabilityReplayTests(unittest.TestCase):
    def test_hold_fire_only_advances_noise_without_contaminating_raw_reference(self):
        caps = profiles()
        with Match(seed=732, duration=2, training_capabilities=caps,
                   battle_settings={'drones_enabled': False}) as match:
            first = {'blue': {'3': {'aim': .55, 'pitch': .1, 'fire': True}}}
            unchanged = copy.deepcopy(first)
            match.step(first)
            self.assertEqual(first, unchanged)
            executor = match._aim_executors['blue:3']
            self.assertFalse(executor.last_result['ready'])
            previous_noise = tuple(executor.noise)
            for i in range(8):
                hold = {'blue': {'3': {'fire': True}}} if i % 2 else {}
                expected = copy.deepcopy(hold)
                match.step(hold)
                self.assertEqual(hold, expected)
                self.assertEqual(match._aim_references['blue:3'], [.55, .1])
                self.assertNotEqual(previous_noise, tuple(executor.noise))
                previous_noise = tuple(executor.noise)
            self.assertTrue(executor.last_result['ready'])
            self.assertAlmostEqual(executor.last_time, .4)
            self.assertTrue(all(tuple(row[1:]) == (.55, .1) for row in executor.queue))

    def test_pre_noise_battle_replay_is_exact_across_capability_drift(self):
        source = resource('examples/claude_opus55_xhigh_2027_squad')
        overrides = {(team, slot): PulseController(team, slot)
                     for team in ('red', 'blue') for slot in (1, 3, 4, 6, 7)}
        initial = profiles()
        with tempfile.TemporaryDirectory() as directory:
            replay_path = Path(directory)/'replay.json'
            with Battle(source, source, seed=733, duration=15.2,
                        sandbox='trusted-local', side_lineups=LINEUPS,
                        controller_overrides=overrides, battle_settings={'drones_enabled': False},
                        observation_profile='sentry_referee_2027_v1',
                        training_capabilities=initial) as battle:
                battle.run()
                self.assertEqual(battle.metadata['training_capabilities'], initial)
                self.assertIn('spec', battle.metadata['capability_drift'])
                self.assertNotEqual(battle.match.training_capabilities['blue:3']['mobility'], initial['blue:3']['mobility'])
                self.assertEqual(battle.actions[0]['blue']['3']['aim'], .55)
                self.assertNotIn('aim', battle.actions[1]['blue']['3'])
                self.assertTrue(battle.actions[0]['blue']['3']['fire'])
                self.assertEqual(battle.match._aim_references['blue:3'], [.55, .1])
                self.assertFalse(battle.diagnostics)
                ticks = len(battle.actions)
                battle.save(replay_path)
            verified = verify_replay(replay_path)
            self.assertEqual(verified['verified_ticks'], ticks)


if __name__ == '__main__': unittest.main()
