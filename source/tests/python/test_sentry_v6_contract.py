import copy
import json
import unittest
from unittest.mock import patch

from rmuc2027_research.rl import recovery_belief as belief, sentry_v4, sentry_v6 as v6


def observation(time=0., hp=300, *, free=False, paid=False, shooter=True):
    sources = {command: dict(valid=True, age=0., source_time=time, receipt_time=time)
               for command in ('0x0201', '0x0209', '0x0101', '0x020D')}
    ack = dict(local_ammo=0, remote_ammo=0, remote_heal=0, free_revive=free,
               paid_revive=paid, paid_cost=100, disengaged=True, ammo_quota=1000, rune_available=True)
    return dict(observation_profile=v6.PROFILE, time=time, tick=round(time*20), team='red',
        robots=[dict(slot=7, team='red', hp=hp, alive=hp>0, max_hp=300, ammo=100,
                     power_outputs=7 if hp and shooter else (3 if hp else 0), rfid=[0]*41)],
        referee=dict(sources=sources, ack=ack, coins=800, events=[], control=dict(red_outpost=0, blue_outpost=2)))


def prefix_features(obs):
    values = [0.]*739
    fields = dict(**{'self.hp': obs['robots'][0]['hp']/1000,
                     'ack.paid_revive': obs['referee']['ack']['paid_revive']})
    for name in ('0201', '020D'):
        packet = obs['referee']['sources']['0x'+name]
        fields['packet.'+name+'.valid'] = packet['valid']
        fields['packet.'+name+'.age'] = packet['age']/5
    for name, value in fields.items(): values[sentry_v4.FEATURE_INDEX[name]] = float(value)
    return values


class RecoveryBeliefTests(unittest.TestCase):
    def test_free_intent_is_legal_before_ready_and_paid_uses_1hz_validity(self):
        obs = observation(.1, 0, free=False, paid=True)
        self.assertEqual(v6.revive_mask(obs), [True, True, True])
        obs['referee']['sources']['0x020D'].update(age=1.5, source_time=-1.4)
        self.assertEqual(v6.revive_mask(obs), [True, True, True])
        obs['referee']['sources']['0x020D']['valid'] = False
        self.assertEqual(v6.revive_mask(obs), [True, True, False])
        obs['referee']['sources']['0x0201']['valid'] = False
        self.assertEqual(v6.revive_mask(obs), [True, False, False])

    def test_missing_hp_is_unknown_not_a_death_and_private_state_is_ignored(self):
        obs = observation(); state = belief.observe(obs)
        missing = observation(.3, 0); missing['referee']['sources']['0x0201']['valid'] = False
        changed = copy.deepcopy(missing); changed['robots'][0].update(weak=True, respawned_at=123, weak_until=1e9)
        a, b = belief.observe(missing, copy.deepcopy(state)), belief.observe(changed, copy.deepcopy(state))
        self.assertEqual(a, b); self.assertIsNone(a['death_interval']); self.assertEqual(set(a['life']), set(belief.LIFE))

    def test_natural_revival_interval_and_new_own_card_clear_evidence(self):
        state = belief.observe(observation())
        state = belief.observe(observation(.1, 0), state)
        belief.record_tx(state, .1, (3<<21)|1)
        revived = observation(.2, 30, shooter=False)
        state = belief.observe(revived, state)
        self.assertEqual(state['revive_interval'], [.1, .2]); self.assertEqual(state['revive_types'], ['normal'])
        self.assertEqual(state['life'], ['normal_weak', 'active'])
        stale = observation(.25, 30, shooter=False); stale['robots'][0]['rfid'][2] = 1
        stale['referee']['sources']['0x0209'].update(source_time=.05, age=.2)
        state = belief.observe(stale, state); self.assertIsNone(state['last_clear'])
        clear = observation(.3, 30, shooter=False); clear['robots'][0]['rfid'][2] = 1
        state = belief.observe(clear, state)
        self.assertIsNone(state['last_clear'])  # Could still retain a pre-death card.
        state = belief.observe(observation(.4, 30, shooter=False), state)
        clear = observation(.5, 30, shooter=False); clear['robots'][0]['rfid'][2] = 1
        state = belief.observe(clear, state)
        self.assertEqual(state['life'], ['active']); self.assertEqual(state['last_clear'], .5)

    def test_shooter_zero_does_not_imply_weak_and_outpost_is_not_proven(self):
        state = belief.observe(observation(4., 100, shooter=False))
        self.assertIn('active', state['life']); self.assertIn('normal_weak', state['life'])
        obs = observation(4.1, 100, shooter=False); obs['robots'][0]['rfid'][8] = 1
        state = belief.observe(obs, state)
        self.assertTrue(state['recovery']['red_outpost']['possible'])
        self.assertFalse(state['recovery']['red_outpost']['proven'])

    def test_same_frame_shot_does_not_overturn_confirmed_death(self):
        state = belief.observe(observation())
        obs = observation(.1, 0)
        obs['referee']['events'] = [dict(command=0x207, source_time=.1)]
        state = belief.observe(obs, state)
        self.assertEqual(state['life'], ['dead'])
        self.assertIsNone(state['last_normal'])

    def test_json_roundtrip_idempotence_reset_and_seat_isolation(self):
        obs = observation(); state = belief.observe(obs)
        frozen = copy.deepcopy(state); self.assertEqual(belief.observe(obs, state), frozen)
        self.assertEqual(belief.from_state_dict(json.loads(json.dumps(state))), state)
        self.assertEqual(len(belief.features(state, 1e12)), 36)
        self.assertTrue(all(0 <= v <= 1 for v in belief.features(state, 1e12)))
        state = belief.observe(observation(1.), state)
        self.assertEqual(belief.observe(observation(), state)['last_now'], 0.)
        other = observation(); other['team'] = 'blue'
        with self.assertRaises(ValueError): belief.observe(other, state)


class SentryV6ContractTests(unittest.TestCase):
    def test_schema_prefix_masks_and_skill_indices(self):
        obs = observation(.1, 0, paid=True); prefix = prefix_features(obs)
        with patch.object(sentry_v4, 'encode', return_value=prefix): result = v6.encode(obs, {})
        self.assertEqual(len(result), 775); self.assertEqual(result[:739], prefix)
        self.assertEqual(v6.SKILL_INDICES, sentry_v4.SKILL_INDICES)
        self.assertEqual(v6.v6_feature_masks(result), [True, True, True])
        self.assertEqual(v6.schema()['critic_size'], 454)

    def test_class_is_not_rounded_or_globally_clipped(self):
        self.assertEqual(v6.validate_action([2., -2., 0., 0., 0., 0., 2]), [1., -1., 0., 0., 0., 0., 2])
        for value in (True, 1.2, -1, 3, float('nan'), '1'):
            with self.assertRaises(ValueError): v6.validate_revive_action(value)

    def test_rl_owns_bits_even_when_free_is_ready_and_when_ack_is_lost(self):
        obs = observation(.1, 0, free=True, paid=True); state = {}
        self.assertEqual(v6.resource_command_v6(obs, state, 0) & 3, 0)
        self.assertEqual(v6.resource_command_v6(obs, state, 1) & 3, 1)
        self.assertEqual(v6.resource_command_v6(obs, state, 2) & 3, 2)
        obs['referee']['sources']['0x020D']['valid'] = False
        self.assertEqual(v6.resource_command_v6(obs, state, 0) & 3, 0)
        self.assertEqual(v6.resource_command_v6(obs, state, 2) & 3, 0)
        self.assertEqual(v6.resource_command_v6(obs, state, 1) & 3, 1)

    def test_repeated_word_never_increments_resource_counters(self):
        obs = observation(.1, 0, free=False, paid=True); state = {}
        one = v6.resource_command_v6(obs, state, 1)
        two = v6.resource_command_v6(obs, state, 1)
        self.assertEqual(one, two); self.assertEqual(state['recovery_belief']['tx_count'], 1)
        self.assertFalse(state['revive_execution']['operation_acknowledged'])

    def test_dead_sample_remains_continuous_noop_after_substep_revive_and_keeps_wire(self):
        obs = observation(.2, 30, shooter=False)
        state = dict(action_selected_dead=True, revive_intent=1, goal=[14., 7.5],
                     sentry_intent=dict(fold=0., jump=0., spin=0., fire=0.))
        original = copy.deepcopy(state)
        def executor(received, memory, navigator, config, resource_builder):
            self.assertIsNone(memory['goal'])
            self.assertEqual(set(memory['sentry_intent'].values()), {-1.})
            word = resource_builder(received['robots'][0], received, memory, False)
            self.assertEqual(word & 3, 0)  # Alive packets cancel the old free intent.
            memory['execution_trace'] = {}
            return {'7': dict(move=[0., 0.], fire=False, sentry_cmd=word)}
        with patch.object(v6.sentry_v3, 'execute', side_effect=executor):
            result = v6.execute(obs, state, None, {})
        self.assertEqual(result['7']['move'], [0., 0.])
        self.assertEqual(state['goal'], original['goal'])
        self.assertEqual(state['sentry_intent'], original['sentry_intent'])
        self.assertTrue(state['execution_trace']['continuous_waiting_for_actor'])
        self.assertIn('recovery_belief', state)


if __name__ == '__main__': unittest.main()
