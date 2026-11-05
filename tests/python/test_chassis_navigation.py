"""Real-engine traversal checks for the two special infantry chassis."""
import math
import unittest
from rmuc2027_core.engine import Match
from rmuc2027_core.sdk import navigate_action, action


class ChassisNavigationTests(unittest.TestCase):
    def test_both_teams_reach_central_ground(self):
        for model in (2731, 2732, 2773):
            for seed in (7, 42, 91):
                with self.subTest(model=model, seed=seed):
                    slot = 7 if model == 2773 else 3
                    lineup = {t: [{'slot': s, 'model': model if s == slot else v} for s, v in
                              [(1, 2711), (3, 2732), (4, 301), (6, 2761), (7, 2771)]]
                              for t in ('red', 'blue')}
                    states = {t: {} for t in lineup}
                    seen = {t: False for t in lineup}
                    reached = set()
                    tipped = set()
                    with Match(seed=seed, lineups=lineup) as match:
                        for tick in range(800):
                            commands = {}
                            for team in lineup:
                                obs = match.observation(team, [slot])
                                robot = next(r for r in obs['robots'] if r['slot'] == slot)
                                if robot.get('tipped'):
                                    tipped.add(team)
                                goal = (12, 7.5) if team == 'red' else (16, 7.5)
                                fields = navigate_action(robot, goal, obs, states[team])
                                seen[team] |= bool(fields.get('jump') if model == 2731 else
                                                   robot.get('transform_progress', 0) >= .99)
                                if math.dist((robot['x'], robot['y']), goal) < .25:
                                    reached.add(team)
                                commands[team] = {str(slot): action(**fields)}
                            if len(reached) == 2:
                                break
                            match.step(commands)
                        self.assertEqual(reached, {'red', 'blue'}, states)
                        self.assertTrue(all(seen.values()), seen)
                        self.assertFalse(tipped, f'{model}: tipped during traversal: {tipped}')

    def test_death_clears_manoeuvre_state(self):
        state = {'entry': {'phase': 'jump'}}
        fields = navigate_action({'alive': False}, (12, 7), {'arena': {}, 'time': 1}, state)
        self.assertEqual(state, {})
        self.assertEqual(fields['move'], (0, 0))
