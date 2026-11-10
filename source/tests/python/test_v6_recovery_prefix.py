"""Control-flow/state-only tests: no engine process, physics or optimization."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rmuc2027_research.rl import sentry_v6, v6_recovery as recovery


def observation(t=0., hp=300, *, free=False, paid=False, coins=800, local=0):
    robot = dict(team='red', slot=7, id='red-7', hp=hp, max_hp=300, alive=hp>0,
        x=5.2, y=6., z=.1, power_outputs=7 if hp==300 else 3 if hp else 0, rfid=[0]*41)
    robot['rfid'][2] = int(t == 0)
    sources = {c: dict(valid=True, age=0., source_time=t) for c in
               ('0x0201', '0x020D', '0x0209', '0x0101')}
    return dict(observation_profile=sentry_v6.PROFILE, team='red', time=t,
        tick=round(t*20), robots=[robot], referee=dict(sources=sources,
        ack=dict(local_ammo=local, remote_ammo=0, remote_heal=0,
                 paid_revive=paid, free_revive=free, paid_cost=100, ammo_quota=1000),
        coins=coins, events=[], control={}))


class FakeBattle:
    def __init__(self, schedule, *, fail=False):
        self.schedule, self.fail = schedule, fail
        self.match = SimpleNamespace(time=0., status='running', _frame={})
        self.original = SimpleNamespace(state={}, calls=0)
        self.policies = [('red', [7], self.original)]
        self._seat_policies = {('red', 7): self.original}
        self.held_commands = {'red': {'7': {'fire': True}}, 'blue': {}}
        self.diagnostics, self.metadata, self.navigation_observations = [], {}, {}
        self.actions = []
        self._set(0.)

    def _set(self, t):
        self.obs = self.schedule(t)
        robot = copy.deepcopy(self.obs['robots'][0])
        robot.update(weak=0 < robot['hp'] < 300, respawn_progress=0.,
            respawn_required=12., respawned_at=t if robot['hp']==30 else None)
        self.match._frame = dict(robots=[robot], events=[])

    def observation(self, team, slots): return copy.deepcopy(self.obs)
    def cancelled(self): return False

    def step(self):
        if self.fail: raise RuntimeError('simulated transport failure')
        action, error = self.policies[0][2].decide(copy.deepcopy(self.obs))
        self.actions.append(action)
        self.match.time = round(self.match.time+.05, 8)
        self._set(self.match.time)
        return copy.deepcopy(self.match._frame)


def session(battle):
    return SimpleNamespace(config=dict(team='red', slot=7, policy_version=6),
        controller=battle.original, battle=battle, obs=None)


class RecoveryPrefixTests(unittest.TestCase):
    def test_real_control_handoff_restores_original_and_preserves_only_legal_history(self):
        battle = FakeBattle(lambda t: observation(t, 300 if t<.1 else 0,
            free=t>=.2, paid=t>=.15))
        original_policies, original_seats = battle.policies, battle._seat_policies
        battle.original.state.update(goal=[9., 9.], revive_intent=2, navigation={'old': True})
        with patch.object(recovery.sdk, 'navigate_action', return_value={'move': [0., 0.]}):
            report = recovery.prepare_recovery_prefix(session(battle), 'free_ready')
        self.assertTrue(report['prepared']); self.assertEqual(report['simulation_seconds'], .2)
        self.assertIs(battle.policies, original_policies); self.assertIs(battle._seat_policies, original_seats)
        self.assertEqual(battle.held_commands, {'red': {}, 'blue': {}})
        self.assertEqual(battle.original.calls, 0)
        self.assertNotIn('goal', battle.original.state); self.assertNotIn('revive_intent', battle.original.state)
        self.assertEqual(battle.original.state['recovery_belief']['life'], ['dead'])
        self.assertTrue(all(a['7']['sentry_cmd'] & 3 == 0 for a in battle.actions))
        self.assertEqual(report['learning_samples'], 0)
        self.assertGreater(report['legal_history_count'], 1)
        json.dumps(report, allow_nan=False)

    def test_alive_weak_uses_real_free_intent_and_keeps_its_legal_history(self):
        battle = FakeBattle(lambda t: observation(t, 300 if t<.1 else 0 if t<.25 else 30))
        with patch.object(recovery.sdk, 'navigate_action', return_value={'move': [0., 0.]}):
            report = recovery.prepare_recovery_prefix(session(battle), 'alive_weak')
        self.assertTrue(report['prepared'])
        self.assertTrue(any(a['7']['sentry_cmd'] & 3 == 1 for a in battle.actions))
        self.assertFalse(any(a['7']['sentry_cmd'] & 3 == 2 for a in battle.actions))
        self.assertEqual(battle.original.state['recovery_belief']['revive_types'], ['normal'])
        self.assertEqual(report['validation_ground_truth']['weak'], True)

    def test_deadline_never_steps_and_failure_always_restores_frozen_policies(self):
        for fail in (False, True):
            battle = FakeBattle(lambda t: observation(t), fail=fail)
            original = battle.policies
            if fail:
                with self.assertRaisesRegex(RuntimeError, 'transport failure'):
                    recovery.prepare_recovery_prefix(session(battle), 'free_ready')
            else:
                report = recovery.prepare_recovery_prefix(session(battle), 'free_ready', deadline=0.)
                self.assertFalse(report['prepared']); self.assertEqual(report['reason'], 'wall_deadline')
                self.assertEqual(battle.actions, [])
            self.assertIs(battle.policies, original)

    def test_unavailable_is_not_only_wait_and_requires_current_death_ack(self):
        battle = FakeBattle(lambda t: observation(t))
        driver = recovery._Driver(session(battle), recovery.SPECS['unavailable'])
        driver.spent = True
        dead = observation(.2, 0, paid=False, free=False, coins=0)
        frame = dict(robots=[dict(dead['robots'][0], respawn_progress=0., respawn_required=12.)])
        driver.update(dead, frame)
        dead['referee']['sources']['0x020D'].update(source_time=0., age=.2)
        self.assertFalse(driver.prepared(dead, frame))
        dead['referee']['sources']['0x020D'].update(source_time=.2, age=0.)
        self.assertTrue(driver.prepared(dead, frame))
        self.assertEqual(sentry_v6.revive_mask(dead), [True, True, False])

    def test_spend_is_idempotent_and_does_not_assume_ack_or_deduct_state(self):
        battle = FakeBattle(lambda t: observation(t))
        driver = recovery._Driver(session(battle), recovery.SPECS['unavailable'])
        obs = observation()
        before = copy.deepcopy(obs)
        a = driver.word(obs, False); b = driver.word(obs, False)
        self.assertEqual(a, b); self.assertEqual(obs, before)
        self.assertEqual((a >> 2) & 2047, 800)
        driver.update(obs, battle.match._frame)
        self.assertFalse(driver.spent); self.assertEqual(driver.phase, 'spend')
        acknowledged = observation(.1, coins=0, local=800)
        driver.update(acknowledged, battle.match._frame)
        self.assertTrue(driver.spent)
        self.assertIn(driver.phase, ('travel', 'exposed'))

    def test_simulation_cap_reports_unprepared_without_manufacturing_death(self):
        battle = FakeBattle(lambda t: observation(t))
        before = copy.deepcopy(battle.match._frame['robots'])
        spec = recovery.RecoveryPrefixSpec('free_ready', .15, (5.2, 6.))
        with patch.dict(recovery.SPECS, free_ready=spec), patch.object(recovery.sdk, 'navigate_action', return_value={'move': [0., 0.]}):
            report = recovery.prepare_recovery_prefix(session(battle), 'free_ready')
        self.assertFalse(report['prepared']); self.assertEqual(report['reason'], 'simulation_limit')
        self.assertEqual(report['simulation_seconds'], .15)
        self.assertEqual(battle.match._frame['robots'][0]['hp'], before[0]['hp'])


class RecoveryPrefixV2Tests(unittest.TestCase):
    def controller_obs(self, *, team='blue', slot=3, target=True, alive=True, distance=5.):
        shooter=dict(team=team,slot=slot,id=team+'-'+str(slot),alive=True,
                     x=16.3,y=6.5,z=0.,hp=300)
        opponent='red' if team=='blue' else 'blue'
        enemy=dict(team=opponent,slot=7,id=opponent+'-7',alive=alive,
                   x=16.3-distance,y=6.5,z=0.,hp=400 if alive else 0)
        return dict(team=team,time=0.,robots=[shooter],enemies=[enemy] if target else [])

    def driver(self,team='red',phase='travel'):
        return SimpleNamespace(team=team,other='blue' if team=='red' else 'red',
            phase=phase,engage=False,spec=recovery.SPECS['free_ready'])

    def test_firing_no_longer_waits_for_exact_learner_waypoint(self):
        c=recovery._PrefixController(self.driver(),'blue',3,'privileged_aux_2027_v1')
        with patch.object(recovery.sdk,'navigate_action',return_value=dict(move=[0.,0.])), \
             patch.object(recovery.sdk,'can_fire',return_value=True), \
             patch.object(recovery.sdk,'ballistic_reachable',return_value=True), \
             patch.object(recovery.sdk,'aim_at',return_value=0.), \
             patch.object(recovery.sdk,'aim_pitch',return_value=0.):
            result,_=c.decide(self.controller_obs())
        self.assertTrue(result['3']['fire']);self.assertFalse(c.driver.engage)
        self.assertEqual(c.guard_counts['in_range'],1)
        self.assertEqual(c.last_diagnostic['goal'],[16.3,6.5])
        json.dumps(c.last_diagnostic,allow_nan=False)

    def test_actual_legality_guards_remain_required(self):
        for target,alive,distance,can,reach in [(False,True,5.,True,True),(True,False,5.,True,True),
                (True,True,7.01,True,True),(True,True,5.,False,True),(True,True,5.,True,False)]:
            c=recovery._PrefixController(self.driver(),'blue',3,'privileged_aux_2027_v1')
            with patch.object(recovery.sdk,'navigate_action',return_value=dict(move=[0.,0.])), \
                 patch.object(recovery.sdk,'can_fire',return_value=can), \
                 patch.object(recovery.sdk,'ballistic_reachable',return_value=reach), \
                 patch.object(recovery.sdk,'aim_at',return_value=0.), \
                 patch.object(recovery.sdk,'aim_pitch',return_value=0.):
                result,_=c.decide(self.controller_obs(target=target,alive=alive,distance=distance))
            self.assertFalse(result['3']['fire'])

    def test_caps_and_team_local_central_targets(self):
        expected={'free_ready':60.,'paid_beneficial':45.,'unavailable':45.,'alive_weak':60.}
        for kind,spec in recovery.SPECS.items():
            self.assertEqual(spec.max_simulation_seconds,expected[kind])
            self.assertTrue(10.5<spec.goal_red[0]<17.5 and 3<spec.goal_red[1]<12)
        for learner in ('red','blue'):
            shooter='blue' if learner=='red' else 'red'
            for slot in (3,4):
                c=recovery._PrefixController(self.driver(learner),shooter,slot,'privileged_aux_2027_v1')
                with patch.object(recovery.sdk,'navigate_action',return_value=dict(move=[0.,0.])):
                    c.decide(self.controller_obs(team=shooter,slot=slot,target=False))
                world=c.last_diagnostic['goal']
                local=recovery._mirror(world,shooter)
                self.assertTrue(10.5<local[0]<17.5 and 3<local[1]<12)

    def test_failed_report_has_bounded_per_seat_diagnostics(self):
        b=FakeBattle(lambda t: observation(t))
        with patch.dict(recovery.SPECS,free_ready=recovery.RecoveryPrefixSpec('free_ready',.15,(11.7,8.5))), \
             patch.object(recovery.sdk,'navigate_action',return_value={'move':[0.,0.]}):
            report=recovery.prepare_recovery_prefix(session(b),'free_ready')
        self.assertFalse(report['prepared']);self.assertEqual(report['version'],2)
        d=report['diagnostics']['seats']['red:7']
        self.assertEqual(d['last']['driver_phase'],'travel')
        self.assertEqual(d['last']['robot']['hp'],300)
        self.assertEqual(len(d['samples']),1)
        self.assertEqual(sum(d['navigation_phase_counts'].values()),3)
        self.assertEqual(report['learning_samples'],0)
        json.dumps(report,allow_nan=False)


if __name__ == '__main__': unittest.main()
