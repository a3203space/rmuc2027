import copy
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rmuc2027_core import sdk
from rmuc2027_core.policy import validate_actions
from rmuc2027_research.rl import privileged_aux as aux
from rmuc2027_research.rl.aux_policy import AuxPolicy, AuxLayerNormPolicy, PortableAuxActor, export_actor
from rmuc2027_research.rl.aux_training import command_to_action, human_representation_batch, imitation_loss
from rmuc2027_research.strategies.battle import Battle


def snapshot(model=2732, slot=3):
    def robot(team, seat, design):
        d = sdk.DESIGNS[str(design)]
        return dict(id=f'{team}_{seat}', team=team, slot=seat, model_slot=design,
            role=d['role'], chassis_type=d['chassis_type'], transform_allowed=d.get('transform_allowed', False),
            arm_count=d.get('arm_count', 0), x=3. if team == 'red' else 25., y=5., z=.3,
            vx=0., vy=0., vz=0., heading=0., aim=0., pitch=0., alive=True, powered=True,
            grounded=True, upright=1., hp=250., max_hp=250., ammo=100, heat=0., max_heat=200,
            weak=False, tipped=False, power_limit=100., mass_kg=d['mass_kg'], radius=.3,
            launcher='installed' if d['role'] == 'heavy' else 'locked', carried_units=2,
            legs=[.2, .2] if d['chassis_type'] == 'wheel_legged' else [0., 0.])
    robots = [robot('red', slot, model), robot('red', 7, 2773),
              robot('blue', 3, 2731), robot('blue', 7, 2773)]
    if slot != 1: robots.append(robot('red', 1, 2711))
    return dict(time=0., tick=0, duration=420, robots=robots, arena=copy.deepcopy(sdk._REFERENCE_ARENA),
        teams={side: dict(resources=1000, completed_difficulties=[1, 2, 3], energy_units_left=5) for side in ('red', 'blue')},
        assemblies={side: dict(stage='idle', difficulty=1) for side in ('red', 'blue')},
        bases={side: dict(x=2 if side == 'red' else 26, y=7.5, hp=7500, max_hp=7500, kind='base') for side in ('red', 'blue')},
        outposts={side: dict(x=8 if side == 'red' else 20, y=7.5, hp=1500, max_hp=1500, kind='outpost', owner=side) for side in ('red', 'blue')},
        projectiles=[], zones={}, runes={}, positions={})


class AuxiliaryContractTests(unittest.TestCase):
    def test_layernorm_portable_and_bounded_endpoint_imitation(self):
        torch.manual_seed(14)
        obs = aux.build_observation(snapshot(), 'red', 3)
        encoded = {k: np.asarray(v, np.float32) for k, v in aux.encode(obs).items()}
        tensor = {k: torch.as_tensor(v).unsqueeze(0).repeat(3, 1) for k, v in encoded.items()}
        tensor['actor'][1] *= .5; tensor['actor'][2] *= 1.5
        policy = AuxLayerNormPolicy(aux.observation_space(), aux.action_space(), lambda _: 1e-4)
        payload = export_actor(policy, family='infantry')
        self.assertEqual(payload['format'], 'rmuc-aux-actor-v2')
        native = policy(tensor, deterministic=True)[0].detach().numpy()
        portable, _ = PortableAuxActor(payload).predict({k:v.numpy() for k,v in tensor.items()})
        np.testing.assert_allclose(native, portable, atol=1e-6, rtol=1e-5)
        sampled, _, log_prob = policy(tensor)
        self.assertTrue(torch.allclose(log_prob, policy.evaluate_actions(tensor, sampled)[1]))
        action, mask = command_to_action(obs, dict(move=[1., -1.], heading=-math.pi, power=100.))
        target = torch.tensor([action]*3, dtype=torch.float32)
        heads = torch.tensor([mask]*3, dtype=torch.float32)
        loss, _, _ = imitation_loss(policy, tensor, target, heads)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(policy.mean_net.weight.grad).all())
        self.assertIsNone(policy.log_std.grad)
        self.assertIsNone(policy.critic_net[0].weight.grad)
        self.assertEqual(len(policy.optimizer.state), 0)
        for row in portable:
            validate_actions({'3': aux.decode(row, obs)}, ['3'], robots=obs['robots'], team='red', observation_profile=aux.PROFILE)

    def test_privilege_is_single_seat_and_copy_isolated(self):
        frame = snapshot()
        with self.assertRaises(ValueError): aux.build_observation(frame, 'red', 7)
        obs = aux.build_observation(frame, 'red', 3)
        self.assertEqual(len(obs['enemies']), 2)
        obs['enemies'][0]['hp'] = -100
        self.assertGreater(frame['robots'][2]['hp'], 0)
        obs = aux.build_observation(frame, 'red', 3)
        encoded = aux.encode(obs)
        self.assertEqual(len(encoded['actor']), aux.schema()['obs_size'])
        self.assertEqual(len(encoded['action_mask']), aux.MASK_SIZE)
        self.assertTrue(np.isfinite(encoded['actor']).all())

    def test_infantry_role_mask_and_no_compatibility_noops(self):
        obs = aux.build_observation(snapshot(), 'red', 3)
        _, masks = aux.split_mask(aux.action_mask(obs))
        self.assertEqual(masks['arm_mode'], [1., 0., 0., 0.])
        self.assertEqual(masks['jump'], [1., 0.])
        self.assertEqual(masks['transform'], [1., 1., 1.])
        command = aux.decode(aux.neutral_action(), obs)
        self.assertFalse({'assembly', 'arm', 'posture', 'deploy', 'support'} & command.keys())
        validate_actions({'3': command}, ['3'], robots=obs['robots'], team='red', observation_profile=aux.PROFILE)

    def test_hybrid_samples_evaluate_identically_and_export_matches(self):
        torch.manual_seed(13)
        obs = aux.build_observation(snapshot(), 'red', 3)
        encoded = {k: np.asarray(v, np.float32) for k, v in aux.encode(obs).items()}
        tensor = {k: torch.as_tensor(v).unsqueeze(0).repeat(8, 1) for k, v in encoded.items()}
        policy = AuxPolicy(aux.observation_space(), aux.action_space(), lambda _: 1e-4)
        actions, _, log_prob = policy(tensor)
        self.assertTrue(torch.allclose(log_prob, policy.evaluate_actions(tensor, actions)[1]))
        self.assertTrue(torch.equal(actions[:, aux.N_CONT:], actions[:, aux.N_CONT:].round()))
        self.assertTrue(torch.isfinite(log_prob).all())
        payload = export_actor(policy, family='infantry')
        self.assertNotIn('critic', json.dumps(payload))
        native = policy(tensor, deterministic=True)[0][0].detach().numpy()
        portable, _ = PortableAuxActor(payload).predict(encoded)
        np.testing.assert_allclose(native, portable, rtol=1e-5, atol=1e-6)
        # PPO's likelihood is differentiable; no optimizer/training is run here.
        (-log_prob.mean()).backward()
        self.assertIsNotNone(policy.actor_net[0].weight.grad)
        self.assertIsNone(policy.critic_net[0].weight.grad)
        with patch.object(sdk, 'navigate_action', return_value={'move': [0., 0.]}):
            for action in actions.detach().numpy():
                command = aux.decode(action, obs)
                validate_actions({'3': command}, ['3'], robots=obs['robots'], team='red', observation_profile=aux.PROFILE)

    def test_heavy_native_teacher_roundtrip_and_no_opening_takeover(self):
        obs = aux.build_observation(snapshot(2712, 1), 'red', 1)
        source = dict(move=[.2, -.4], aim=.7, pitch=.15, fire=True, brake=True,
                      power=75., arm={'joints': [.3, .1, .4, .2, .1, -.4, .8]},
                      arm2={'stow': True})
        action, labels = command_to_action(obs, source)
        command = aux.decode(action, obs)
        np.testing.assert_allclose(command['move'], source['move'])
        np.testing.assert_allclose(command['arm']['joints'], source['arm']['joints'])
        self.assertAlmostEqual(command['aim'], source['aim'])
        self.assertEqual(command['fire'], True)
        validate_actions({'1': command}, ['1'], robots=obs['robots'], team='red', observation_profile=aux.PROFILE)
        self.assertTrue(labels[aux.ACTION_INDEX['arm_joint0']])
        controller = aux.AuxController({'team': 'red', 'slot': 1})
        controller.select(obs, action)
        self.assertEqual(controller.decide(obs)[0]['1'], command)
        self.assertEqual(controller.state['mode'], 'policy')

    def test_untrusted_override_actions_use_same_boundary(self):
        obs = aux.build_observation(snapshot(), 'red', 3)
        for action in ({'move': [math.nan, 0]}, {'arm': {'stow': True}},
                       {'messages': [{'to': 'all', 'data': {'enemy': [20, 5]}}]},
                       {'deploy': False}, {'posture': 'attack'}):
            with self.assertRaises(ValueError):
                validate_actions({'3': action}, ['3'], True, robots=obs['robots'], team='red', observation_profile=aux.PROFILE)

    def test_dead_revive_and_full_dual_arm_command_fit_protocol(self):
        obs = aux.build_observation(snapshot(2712, 1), 'red', 1)
        robot = aux.own_robot(obs)
        robot['alive'] = False
        action = aux.neutral_action(); action[aux.ACTION_INDEX['revive']] = 1
        self.assertEqual(aux.decode(action, obs), {'revive': True})
        robot['alive'] = True; robot['launcher'] = 'unlocked'
        obs['assemblies']['red'].update(stage='inserting', difficulty=4)
        action = [.812345678] * aux.N_CONT + [0.] * len(aux.CAT_NAMES)
        for name, value in dict(target=1, ammo=2, ammo_batches=3, remote_heal=1,
            arm_mode=2, arm2_mode=2, assembly_core=2, assembly_insert=1, assembly_lift=1,
            assembly_flip=1, assembly_confirm=1, install_launcher=1, power_mode=1, turn=2).items():
            action[aux.ACTION_INDEX[name]] = value
        command = aux.decode(action, obs)
        self.assertLessEqual(len(json.dumps(command).encode()), 1024)
        self.assertIn('arm2', command); self.assertIn('assembly_insert', command)
        validate_actions({'1': command}, ['1'], robots=obs['robots'], team='red', observation_profile=aux.PROFILE)

    def test_summary_uses_trusted_roster_without_losing_capability_checks(self):
        summary = [dict(team='red', slot=3, hp=200, alive=True)]
        roster = {('red', '3'): dict(role='infantry', chassis_type='wheel_legged',
                                    transform_allowed=False, arm_count=0)}
        validate_actions({'3': {'jump': .5, 'leg_height': .2}}, ['3'],
                         robots=summary, team='red', roster=roster)
        for command in ({'arm': {'stow': True}}, {'transform': True}, {'posture': 'attack'}):
            with self.assertRaises(ValueError):
                validate_actions({'3': command}, ['3'], robots=summary, team='red', roster=roster)
        self.assertNotIn('role', summary[0])

    def test_battle_direct_and_encoded_views_preserve_strict_seat(self):
        frame = snapshot(); calls = []
        def ordinary(team, slots):
            calls.append(tuple(slots)); return dict(team=team, controlled_ids=list(map(str, slots)),
                robots=[dict(slot=slots[0], team=team)], enemies=[], observation_profile='sentry_referee_2027_v1')
        battle = object.__new__(Battle)
        battle.match = SimpleNamespace(tick=0, snapshot=lambda: copy.deepcopy(frame), observation=ordinary,
            training_capabilities={'red:3': {'aim': {'sigma_deg': 2.5}}})
        battle._aux_snapshot_tick = None; battle._aux_snapshot = None
        controller = aux.AuxController({'slot': 3})
        battle._seat_policies = {('red', 3): controller}
        battle.observation_profile = 'sentry_referee_2027_v1'; battle.navigation_mode = 'legacy'
        battle.communication = SimpleNamespace(inboxes={'red': {'3': [{'data': 'blocked'}], '7': [{'data': 'blocked'}]}})
        privileged = battle.observation('red', [3]); restricted = battle.observation('red', [7])
        self.assertEqual(calls, [(7,)])
        self.assertEqual(privileged['observation_profile'], aux.PROFILE)
        self.assertNotIn('training_capabilities', restricted)
        self.assertEqual(restricted['enemies'], [])
        self.assertEqual(privileged['messages'], [])
        self.assertEqual(restricted['messages'], [])

    def test_drone_does_not_redirect_to_captured_own_outpost_or_base(self):
        frame = snapshot(); obs = dict(frame, team='red', controlled_ids=['6'])
        drone = dict(id='red_6', slot=6, team='red', role='aerial', drones=[dict(x=3, y=5, z=1, state='flying')])
        obs['robots'] = [drone]
        plate = {'center': [8, 7.5, 1], 'normal': [-1, 0, 0]}
        obs['outposts']['red'].update(owner='blue', armor_plates=[plate])
        obs['outposts']['blue']['hp'] = 0
        self.assertEqual(aux.OutpostDroneController().decide(obs)[0], {'6': {'drones': {}}})

    def test_human_representation_is_origin_invariant_and_missing_masked(self):
        n = 7; state = np.zeros((n, 16), np.float32); state[:, 4:7] = [80, 200, 250]
        data = dict(state=state, state_valid=np.ones_like(state, dtype=bool), role_id=np.full(n, 2),
            time_s=np.arange(n), game_id=np.ones(n), robot_id=np.full(n, 3), alive=np.ones(n, bool),
            discontinuity=np.zeros(n, bool), source_xy=np.column_stack((np.arange(n), np.zeros(n))).astype(np.float32),
            source_position_valid=np.ones(n, bool), future_source_xy=np.zeros((n, 3, 2), np.float32),
            future_source_mask=np.zeros((n, 3), bool))
        for k, horizon in enumerate((1, 3, 5)):
            data['future_source_xy'][1, k] = [1+horizon, 0]; data['future_source_mask'][1, k] = True
        x, y, m = human_representation_batch(data, [1], family='infantry')
        moved = copy.deepcopy(data); moved['source_xy'] += [41, -12]; moved['future_source_xy'] += [41, -12]
        xx, yy, mm = human_representation_batch(moved, [1], family='infantry')
        np.testing.assert_array_equal(x, xx); np.testing.assert_array_equal(y, yy); np.testing.assert_array_equal(m, mm)
        names = [name for name, _ in aux.FEATURES]
        self.assertEqual(x[0, names.index('valid.ally.3.x')], 0)
        self.assertEqual(x[0, names.index('valid.ally.3.power_w')], 1)
        with self.assertRaises(ValueError): human_representation_batch(data, [1], family='main2773')


if __name__ == '__main__': unittest.main()
