"""RL boundary regression tests; engine/runtime tests need no neural libraries."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from rmuc2027_core.engine import Match
from rmuc2027_research.strategies.battle import Battle
from rmuc2027_research.rl.portable import (GROUND_SLOTS, OBS_SIZE, OBS_VERSION, SLOTS, SdkNavigator,
    capability_bits, encode_observation, execute_goal, mlp_forward, select_goal)
from rmuc2027_research.rl.session import GoalController, load_config, provenance, resource
from rmuc2027_research.rl import session as rl_session
from rmuc2027_research.rl.terrain_witness import TerrainWitness
from rmuc2027_research.rl.exporting import export_layers
from rmuc2027_core.replay import verify_replay


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Match(seed=42, map_name='rmuc2027') as match:
            cls.observation = match.observation('red', [7])
        cls.config = load_config()

    def test_fixed_features_are_bounded_and_ignore_referee_state(self):
        original = encode_observation(self.observation, {})
        obs = copy.deepcopy(self.observation)
        obs['referee'] = {'robots': [{'x': float('nan'), 'hp': 12345}], 'reward': 1000}
        self.assertEqual(original, encode_observation(obs, {}))
        self.assertEqual(len(original), OBS_SIZE)
        self.assertTrue(all(math.isfinite(v) and -1 <= v <= 1 for v in original))
        obs['controlled_ids'] = ['1', '7']
        with self.assertRaises(ValueError):
            encode_observation(obs, {})

    def test_interrupted_transport_cannot_consume_a_late_frame_as_digest(self):
        with Match(seed=42, map_name='rmuc2027') as match:
            with patch.object(match, '_read_line', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    match.step({})
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                match.digest()

    @unittest.skipUnless(importlib.util.find_spec('gymnasium'), 'Optional Gymnasium dependency unavailable')
    def test_non_numeric_gym_action_closes_every_owned_child(self):
        from rmuc2027_research.rl.env import SentryEnv
        env = SentryEnv(self.config)
        try:
            env.reset(seed=42)
            pids = env.process_ids()
            with self.assertRaises(ValueError):
                env.step(['invalid', 'coordinates'])
            self.assertIsNone(env.session.battle)
            self.assertFalse([pid for pid in pids if Path(f'/proc/{pid}').exists()])
        finally:
            env.close()

    def test_coordinate_projection_rejects_nonfinite_and_avoids_obstacles(self):
        navigator = SdkNavigator(.25)
        state = {}
        for bad in ([float('nan'), 0], [float('inf'), 0], [0]):
            with self.assertRaises(ValueError):
                select_goal(self.observation, state, bad, navigator, self.config)
        select_goal(self.observation, state, [0, 0], navigator, self.config)
        self.assertEqual(state['proposed_goal'], [14.0, 7.5])
        self.assertTrue(all(math.isclose(v/.25, round(v/.25)) for v in state['goal']))
        self.assertNotEqual(state['goal'], state['proposed_goal'])

    def test_controller_owns_only_sentry_and_recovery_has_hysteresis(self):
        controller = GoalController(self.config)
        obs = copy.deepcopy(self.observation)
        robot = next(r for r in obs['robots'] if r['slot'] == 7)
        controller.select(obs, [0, 0])
        for hp, alive, weak, expected in ((.2, True, False, 'recover'), (.5, True, False, 'recover'),
                (.9, True, False, 'policy'), (0, False, False, 'dead'), (.9, True, True, 'recover'), (.9, True, False, 'policy')):
            robot.update(hp=robot['max_hp']*hp, alive=alive, weak=weak)
            actions, error = controller.decide(obs)
            self.assertIsNone(error)
            self.assertEqual(set(actions), {'7'})
            self.assertEqual(controller.state['mode'], expected)
            if not alive:
                self.assertEqual(actions['7']['move'], [0, 0])
        controller.close()

    def test_truncation_does_not_finish_game_or_award_win(self):
        from rmuc2027_research.rl.session import ArenaSession
        config = {**self.config, 'episode_seconds': 1}
        with ArenaSession(config) as session:
            session.reset(42)
            _, _, terminated, truncated, info = session.step([0, 0])
            self.assertFalse(terminated)
            self.assertTrue(truncated)
            self.assertEqual(session.battle.match.status, 'running')
            self.assertEqual(info['reward_terms']['outcome'], 0)
            self.assertAlmostEqual(info['sampled_simulation_seconds'],1.)
            with self.assertRaises(RuntimeError):
                session.step([0, 0])

    def test_natural_end_and_no_recording_keep_constant_frame_count(self):
        from rmuc2027_research.rl.session import ArenaSession
        with ArenaSession(self.config) as session:
            session.reset(42)
            session.battle.queue_referee({'kind': 'forfeit', 'team': 'blue'})
            _, reward, terminated, truncated, info = session.step([0, 0])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            # Read the weight instead of restating it: this assertion hard-coded
            # the old 10.0 and silently became a claim that the reward never
            # changed, which is not what the test is about. What it is about is
            # that a natural finish pays the win exactly once, in full.
            self.assertEqual(info['reward_terms']['outcome'], self.config['rewards']['outcome'])
            self.assertEqual(round(reward, 9), round(sum(info['reward_terms'].values()), 9))
            self.assertLess(info['sampled_simulation_seconds'],self.config['rl_seconds'])
            self.assertGreaterEqual(info['sampled_simulation_seconds'],0.)
            self.assertEqual(len(session.battle.frames), 1)
            self.assertEqual(session.battle.actions, [])
            with self.assertRaises(RuntimeError):
                session.battle.replay()

    def test_exported_stdlib_policy_runs_in_existing_worker_and_replays(self):
        # Test fixture only: actual trained-weight parity is checked by export_model.
        layers = [{'weight': [[0.0]*OBS_SIZE for _ in range(2)], 'bias': [0.0, 0.0]}]
        self.assertEqual(mlp_forward([0]*OBS_SIZE, layers), [0, 0])
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/'squad'
            metadata = export_layers(layers, provenance(self.config), target)
            self.assertLess(metadata['sentry_source_bytes'], 512*1024)
            with Battle(target, resource(self.config['opponent']), 42, 1, 'rmuc2027', 'trusted-local') as battle:
                result = battle.run()
                self.assertEqual(result['policy_errors'], 0, battle.diagnostics)
                replay = Path(directory)/'match.json'
                battle.save(replay)
            self.assertEqual(verify_replay(replay)['digest'], result['digest'])


class SummarySerialisationTests(unittest.TestCase):
    """The action_trace=false summary line must survive numpy observations."""

    def _callback(self):
        from rmuc2027_research.rl import learning
        return learning, learning.TraceCallback

    def test_numpy_payloads_are_encoded(self):
        # Regression: the summary branch used a bare json.dumps, so any run with
        # action_trace=false died with "Object of type ndarray is not JSON
        # serializable" after training had already been going for minutes.
        learning, TraceCallback = self._callback()
        payload = {'metrics': [{'obs': np.zeros(4, dtype=np.float32),
                                'scalar': np.float32(0.5),
                                'nested': {'flags': np.array([1, 0], dtype=np.int64)}}]}
        text = json.dumps(payload, default=learning._json_default)
        self.assertEqual(json.loads(text)['metrics'][0]['obs'], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(json.loads(text)['metrics'][0]['scalar'], 0.5)
        self.assertEqual(json.loads(text)['metrics'][0]['nested']['flags'], [1, 0])

    def test_unsupported_objects_still_raise(self):
        learning, _ = self._callback()
        with self.assertRaises(TypeError):
            json.dumps({'bad': object()}, default=learning._json_default)

    def test_summary_written_when_details_disabled(self):
        learning, TraceCallback = self._callback()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'actions.jsonl'
            callback = TraceCallback(path, detailed=False)
            callback.model = type('M', (), {'n_steps': 64, '_n_updates': 4,
                                            'num_timesteps': 1024})()
            # training_env is a read-only property on BaseCallback.
            with patch.object(type(callback), 'training_env',
                              new_callable=unittest.mock.PropertyMock,
                              return_value=type('E', (), {'num_envs': 2})()):
                callback.last_progress = 0.0
                callback.stream = path.open('w')
                callback.rollout_behavior = {'samples': 0, 'moving_commands': 0, 'modes': {}}
                callback.locals = {'infos': [{'obs': np.zeros(3, dtype=np.float32)}], 'dones': [False, False]}
                try:
                    callback._on_step()
                finally:
                    callback.stream.close()
            lines = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            self.assertTrue(any(line.get('kind') == 'summary' for line in lines),
                            f'summary line missing: {lines}')


class ResumeDirectoryTests(unittest.TestCase):
    """A resumed run must reopen its own directory, not demand a new one."""

    def test_resume_reopens_existing_run_directory(self):
        from rmuc2027_research.rl import learning

        captured = {}

        def fake_dependencies(config):
            return {'config': config, 'sources': {}, 'sdk_hash': 'x', 'obs_version': 1,
                    'obs_size': 2, 'simulation': {}, 'navigation': None,
                    'runtime_hash': 'r', 'engine_binary_sha256': 'e', 'rules_sha256': 'u'}

        def fake_vector_env(config, count):
            raise RuntimeError('stop after directory handling')

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'run'
            run.mkdir()
            (run/'experiment.json').write_text(json.dumps(fake_dependencies({'seed': 1})))
            checkpoint = run/'interrupted.zip'
            checkpoint.write_bytes(b'placeholder')
            captured['run'] = run
            with patch.object(learning, 'experiment', fake_dependencies), \
                 patch.object(learning, 'vector_env', fake_vector_env), \
                 patch.object(learning, 'device_name', lambda *_: None):
                # The failure has to come from vector_env, i.e. after mkdir: had
                # mkdir kept exist_ok=False this would raise FileExistsError.
                with self.assertRaisesRegex(RuntimeError, 'stop after directory handling'):
                    learning.train({'seed': 1}, run, 1024, 2, 'cpu', resume=checkpoint)
            # Provenance is preserved rather than rewritten from the live config.
            self.assertEqual(json.loads((run/'experiment.json').read_text())['sdk_hash'], 'x')

    def test_init_from_still_requires_a_fresh_directory(self):
        from rmuc2027_research.rl import learning

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)/'run'
            run.mkdir()
            with patch.object(learning, 'device_name', lambda *_: None):
                with self.assertRaises(FileExistsError):
                    learning.train({'seed': 1}, run, 1024, 2, 'cpu', init_from=run/'model.zip')


class EnemyCapabilityTests(unittest.TestCase):
    """The enemy block reports what an opponent can do, not which wheels it has.

    chassis_type cannot separate 2732 (transforms, drives through the tunnel)
    from 4 (mecanum, no transformation): both say "mecanum". Rulebook table
    5-27 makes the distinction behavioural - driving through the tunnel grants
    1.5x cooling for 120 s while crossing it over the top grants 25% defence -
    so the encoder reports tunnel/climb/omni as three independent flags.
    """

    def _observation(self, enemies, model_by_slot):
        robots = [{'slot': 7, 'x': 2.0, 'y': 3.0, 'vx': 0.1, 'vy': 0.2, 'hp': 300, 'max_hp': 400,
                   'alive': True, 'heading': 0.3}]
        robots += [{'slot': s, 'x': 1.0, 'y': 1.0, 'vx': 0.0, 'vy': 0.0, 'hp': 100, 'max_hp': 100,
                    'alive': True} for s in (1, 3, 4, 6)]
        payload = [{'slot': s, 'model_slot': model_by_slot[s], 'x': 5.0+s, 'y': 5.0, 'vx': 0.0, 'vy': 0.0,
                    'hp': 400, 'max_hp': 400, 'alive': True} for s in enemies]
        return {'controlled_ids': ['7'], 'robots': robots, 'enemies': payload, 'team': 'red',
                'arena': {'width': 28.0, 'height': 15.0}, 'remaining': 300.0, 'time': 120.0,
                'teams': {'red': {'control': 100, 'resources': 50},
                          'blue': {'control': 80, 'resources': 40}},
                'bases': {'red': {'hp': 10000, 'max_hp': 10000},
                          'blue': {'hp': 9000, 'max_hp': 10000}},
                'positions': {f'{t}_{k}': {'controller': t, 'control': 'full', 'progress': 2}
                              for t in ('red', 'blue') for k in ('outpost', 'base')}}

    def _traits(self, encoded, slot):
        offset = 18 + 4*7 + SLOTS.index(slot)*11 + 8
        return encoded[offset:offset+3]

    def test_control_features_use_the_configured_divisor(self):
        # The two control features sit right after the public match block
        # (remaining, control[team], control[other], ...).
        from rmuc2027_research.rl.portable import CONTROL_SCALE
        obs = self._observation(SLOTS, {1: 2711, 3: 2731, 4: 2732, 6: 2761, 7: 2771})
        base = 18 + 4*7 + 5*11
        default = encode_observation(obs, {}, 7, 0.5)
        self.assertAlmostEqual(default[base+1], 100/CONTROL_SCALE, places=6)
        self.assertAlmostEqual(default[base+2], 80/CONTROL_SCALE, places=6)
        # The divisor is threaded, not hard-coded, because export bakes the
        # training value into the submission: an actor trained against one
        # divisor and shipped with another sees a rescaled input.
        scaled = encode_observation(obs, {}, 7, 0.5, None, 200.0)
        self.assertAlmostEqual(scaled[base+1], 0.5, places=6)
        self.assertAlmostEqual(scaled[base+2], 0.4, places=6)
        self.assertEqual(len(scaled), OBS_SIZE)

    def test_the_control_divisor_is_finite_and_positive_in_the_default_config(self):
        # A zero or negative divisor would divide by zero or flip the sign, and
        # both survive `math.isfinite(rewards)`. Assert the shipped value too,
        # so a stray edit to sentry.json fails here rather than in training.
        config = load_config()
        self.assertGreater(config['control_scale'], 0)
        self.assertTrue(math.isfinite(config['control_scale']))
        # The measured red-side peak across the 512-game pool was 863; a divisor
        # below that would clip live games and above it wastes range.
        self.assertGreaterEqual(config['control_scale'], 863)
        for broken in (0, -1.0, float('nan'), float('inf'), '1000', True):
            bad = copy.deepcopy(json.loads(rl_session.DEFAULT_CONFIG.read_text()))
            bad['control_scale'] = broken
            with self.assertRaises(ValueError, msg=f'control_scale={broken!r} accepted'):
                load_config(bad)

    def test_the_control_divisor_travels_into_the_export(self):
        # export_layers copies a fixed key list into the submission's _CONFIG.
        # Omitting control_scale there is invisible until a policy is swapped
        # in, which is exactly the train/serve skew this guards.
        source = (Path(rl_session.__file__).parent/'exporting.py').read_text()
        self.assertIn("'control_scale'", source)
        self.assertIn("_CONFIG['control_scale']", source)

    def test_design_table_flags_are_orthogonal(self):
        # 2732 transforms into a tunnel profile but cannot jump the 200 mm
        # terrace; 2731 is the exact mirror; 2713 is omni and neither.
        self.assertEqual(capability_bits({'model_slot': 2732, 'slot': 3}), [1.0, 0.0, 0.0])
        self.assertEqual(capability_bits({'model_slot': 2731, 'slot': 3}), [0.0, 1.0, 0.0])
        self.assertEqual(capability_bits({'model_slot': 2713, 'slot': 1}), [0.0, 0.0, 1.0])
        # 4 and 2732 share the same chassis_type but not the same ability, which
        # is precisely why the encoder cannot key off chassis_type alone.
        self.assertEqual(capability_bits({'model_slot': 4, 'slot': 3}), [0.0, 0.0, 0.0])

    def test_model_slot_falls_back_to_the_slot_number(self):
        # Designs that keep their slot number omit model_slot entirely.
        self.assertEqual(capability_bits({'slot': 3}), capability_bits({'model_slot': 3, 'slot': 3}))

    def test_enemy_block_carries_the_traits(self):
        models = {1: 2711, 3: 2731, 4: 2732, 6: 2761, 7: 2771}
        encoded = encode_observation(self._observation(SLOTS, models),
                                     {'goal': (1.0, 1.0), 'mode': 'policy', 'goal_time': 119.0}, 7, 0.5)
        self.assertEqual(len(encoded), OBS_SIZE)
        self.assertEqual(OBS_VERSION, 2)
        self.assertEqual(self._traits(encoded, 1), [0.0, 0.0, 0.0])   # heavy, plain mecanum
        self.assertEqual(self._traits(encoded, 3), [0.0, 1.0, 0.0])   # jump only
        self.assertEqual(self._traits(encoded, 4), [1.0, 0.0, 0.0])   # tunnel only
        self.assertEqual(self._traits(encoded, 7), [0.0, 0.0, 1.0])   # omni only
        # The aerial slot parks in a hangar and is never a ground mark target.
        self.assertEqual(self._traits(encoded, 6), [0.0, 0.0, 0.0])
        self.assertNotIn(6, GROUND_SLOTS)

    def test_hidden_slot_stays_zero(self):
        models = {1: 2711, 3: 2731, 6: 2761, 7: 2771}   # slot 4 never seen
        encoded = encode_observation(self._observation(models, models),
                                     {'goal': (1.0, 1.0), 'mode': 'policy', 'goal_time': 119.0}, 7, 0.5)
        offset = 18 + 4*7 + SLOTS.index(4)*11
        self.assertEqual(encoded[offset:offset+11], [0.0]*11)


class TerrainWitnessTests(unittest.TestCase):
    """The ability bits come from watching the opponent, not from a design lookup.

    The radar carries coordinates only (protocol p.67), so an encoder that wants
    to know whether an opponent can take the tunnel shortcut has to infer it.
    Rulebook table 5-27 gives the two crossings different rewards - 25% defence
    for 30 s over the top, 1.5x cooling for 120 s through the middle - and only a
    folding chassis can reach the card under the roof, so passing the inside card
    is the observable proof.
    """

    INSIDE = 'red_near_tunnel_1'
    END = 'red_near_tunnel_0'

    def _obs(self, t, enemies, zones=None):
        return {'controlled_ids': ['7'], 'team': 'red', 'time': t,
                'robots': [{'slot': 7, 'x': 2.0, 'y': 3.0, 'vx': 0.0, 'vy': 0.0,
                            'hp': 300, 'max_hp': 400, 'alive': True, 'heading': 0.0}],
                'enemies': enemies, 'arena': {'width': 28.0, 'height': 15.0},
                'zones': zones or {},
                'remaining': 300.0, 'teams': {'red': {'control': 100, 'resources': 50},
                                              'blue': {'control': 80, 'resources': 40}},
                'bases': {'red': {'hp': 10000, 'max_hp': 10000},
                          'blue': {'hp': 9000, 'max_hp': 10000}},
                'positions': {f'{t}_{k}': {'controller': t, 'control': 'full', 'progress': 2}
                              for t in ('red', 'blue') for k in ('outpost', 'base')}}

    def _enemy(self, slot, x, y, z=0.16, model=2761):
        # 2761 (hangar) has all three design-table bits clear, so a test that
        # does not name a model is stating "this slot's prior is zero" and any
        # non-zero bit it then reads must have come from the witness. 2771 would
        # be a poor default here: its omni prior is [0, 0, 1].
        return {'slot': slot, 'id': f'blue-{slot}', 'model_slot': model,
                'x': x, 'y': y, 'z': z,
                'vx': 0.0, 'vy': 0.0, 'hp': 400, 'max_hp': 400, 'alive': True}

    def test_inside_card_contact_proves_the_tunnel(self):
        witness = TerrainWitness()
        # Slot 4 parks on the card under the roof. The ends alone would not do:
        # crossing over the top also touches both of those.
        witness.observe(self._obs(10.0, [self._enemy(4, 14.92, 13.475)],
                                  {self.INSIDE: {'occupants': ['blue-4']},
                                   self.END: {'occupants': ['blue-4']}}), 7)
        self.assertEqual(witness.bits(4), [1.0, 0.0, 0.0])

    def test_end_cards_alone_prove_nothing(self):
        witness = TerrainWitness()
        witness.observe(self._obs(10.0, [self._enemy(1, 14.36, 13.475)],
                                  {self.END: {'occupants': ['blue-1']},
                                   'red_near_tunnel_2': {'occupants': ['blue-1']}}), 7)
        self.assertEqual(witness.bits(1), [0.0, 0.0, 0.0])

    def test_unseen_occupant_is_not_credited(self):
        # The card readout is match-scoped, but the witness only counts what this
        # robot could see: slot 4 is not in obs['enemies'], so it earns nothing.
        witness = TerrainWitness()
        witness.observe(self._obs(10.0, [],
                                  {self.INSIDE: {'occupants': ['blue-4']}}), 7)
        self.assertEqual(witness.bits(4), [0.0, 0.0, 0.0])

    def test_two_rises_away_from_a_tunnel_prove_a_climb(self):
        witness = TerrainWitness()
        # 200 mm terrace (rulebook 4.3.1): two 0.12 m+ rises inside the window.
        witness.observe(self._obs(10.0, [self._enemy(3, 6.0, 5.0, z=0.16)]), 7)
        witness.observe(self._obs(10.5, [self._enemy(3, 6.1, 5.0, z=0.30)]), 7)
        witness.observe(self._obs(11.0, [self._enemy(3, 6.2, 5.0, z=0.44)]), 7)
        self.assertEqual(witness.bits(3), [0.0, 1.0, 0.0])

    def test_one_rise_is_a_bump_not_a_climb(self):
        witness = TerrainWitness()
        witness.observe(self._obs(10.0, [self._enemy(3, 6.0, 5.0, z=0.16)]), 7)
        witness.observe(self._obs(10.5, [self._enemy(3, 6.1, 5.0, z=0.30)]), 7)
        self.assertEqual(witness.bits(3), [0.0, 0.0, 0.0])

    def test_a_height_change_inside_the_tunnel_is_not_a_climb(self):
        # The roof is the other way z jumps by this much, so a rise measured on
        # the tunnel footprint must not be read as climbing the terrace.
        witness = TerrainWitness()
        witness.observe(self._obs(10.0, [self._enemy(3, 14.93, 13.475, z=0.16)]), 7)
        witness.observe(self._obs(10.5, [self._enemy(3, 14.93, 13.475, z=0.50)]), 7)
        witness.observe(self._obs(11.0, [self._enemy(3, 14.93, 13.475, z=0.84)]), 7)
        self.assertEqual(witness.bits(3), [0.0, 0.0, 0.0])

    def test_omni_stays_on_the_prior_because_nothing_observes_it(self):
        # Strafe ability leaves no trace in a position track, so not even a
        # tunnel crossing can change that bit: it stays on the prior, which is 1
        # for the omni 2771 and 0 for everything else.
        witness = TerrainWitness()
        omni = self._enemy(7, 5.0, 5.0, model=2771)
        witness.observe(self._obs(10.0, [omni],
                                  {self.INSIDE: {'occupants': ['blue-7']}}), 7)
        self.assertEqual(witness.bits(7, omni), [1.0, 0.0, 1.0])
        plain = self._enemy(7, 5.0, 5.0, model=2732)
        self.assertEqual(witness.bits(7, plain), [1.0, 0.0, 0.0])

    def test_flags_latch_for_the_rest_of_the_episode(self):
        witness = TerrainWitness()
        witness.observe(self._obs(10.0, [self._enemy(4, 14.92, 13.475)],
                                  {self.INSIDE: {'occupants': ['blue-4']}}), 7)
        witness.observe(self._obs(20.0, [self._enemy(4, 5.0, 5.0)]), 7)
        self.assertEqual(witness.bits(4), [1.0, 0.0, 0.0])

    def test_evidence_overrides_the_prior_bit_by_bit(self):
        # 2731 jumps (prior climb=1) but was never seen crossing a tunnel (prior
        # tunnel=0). Watching it ride the inside card has to raise the tunnel bit
        # and leave the climb bit exactly where the table put it.
        witness = TerrainWitness()
        enemy = self._enemy(3, 14.92, 13.475, model=2731)
        witness.observe(self._obs(10.0, [enemy],
                                  {self.INSIDE: {'occupants': ['blue-3']}}), 7)
        self.assertEqual(witness.bits(3, enemy), [1.0, 1.0, 0.0])

    def test_an_unwatched_slot_costs_the_prior_not_a_fabricated_zero(self):
        # Nothing has been seen of slot 4, so the only honest answer is the
        # public table: a folding robot could still take the tunnel later. This
        # is the same three numbers an exported submission computes.
        witness = TerrainWitness()
        enemy = self._enemy(4, 14.92, 13.475, model=2732)
        self.assertEqual(witness.bits(4, enemy), [1.0, 0.0, 0.0])
        # And with no observation at all there is no prior to lean on.
        self.assertEqual(witness.bits(4), [0.0, 0.0, 0.0])

    def test_encoder_prefers_the_witness_over_the_design_table(self):
        # 2732 could fold, but nobody saw it do so: the witnessed encoder still
        # reports the prior here, and the unwitnessed one must agree bit for bit
        # so that training and the exported actor read the same numbers.
        witness = TerrainWitness()
        models = {1: 2732, 3: 2731, 4: 2732, 6: 2761, 7: 2771}
        payload = [{'slot': s, 'model_slot': models[s], 'x': 5.0+s, 'y': 5.0, 'vx': 0.0, 'vy': 0.0,
                    'hp': 400, 'max_hp': 400, 'alive': True} for s in SLOTS]
        state = {'goal': (1.0, 1.0), 'mode': 'policy', 'goal_time': 119.0}
        encoded = encode_observation(self._obs(10.0, payload), state, 7, 0.5, witness)
        self.assertEqual(len(encoded), OBS_SIZE)
        base = 18 + 4*7
        # Nothing watched yet: the two paths agree on every enemy bit.
        plain = encode_observation(self._obs(10.0, payload), state, 7, 0.5)
        for slot in GROUND_SLOTS:
            offset = base + SLOTS.index(slot)*11 + 8
            self.assertEqual(encoded[offset:offset+3], plain[offset:offset+3],
                             f'slot {slot} disagrees before any evidence')
            self.assertEqual(encoded[offset:offset+3],
                             capability_bits({'slot': slot, 'model_slot': models[slot]}))

    def test_witnessed_evidence_moves_exactly_the_watched_bits(self):
        # The gate that keeps this from being a silent mismatch: once slot 4 is
        # seen riding the inside card, its tunnel bit must rise while every other
        # enemy bit stays exactly on its design-table prior.
        witness = TerrainWitness()
        models = {1: 2732, 3: 2731, 4: 2732, 6: 2761, 7: 2771}
        payload = [{'slot': s, 'model_slot': models[s], 'x': 5.0+s, 'y': 5.0, 'vx': 0.0, 'vy': 0.0,
                    'hp': 400, 'max_hp': 400, 'alive': True} for s in SLOTS]
        state = {'goal': (1.0, 1.0), 'mode': 'policy', 'goal_time': 119.0}
        witness.observe(self._obs(10.0, payload,
                                  {self.INSIDE: {'occupants': ['blue-4']}}), 7)
        base = 18 + 4*7
        watched = encode_observation(self._obs(10.0, payload), state, 7, 0.5, witness)
        plain = encode_observation(self._obs(10.0, payload), state, 7, 0.5)
        for slot in GROUND_SLOTS:
            offset = base + SLOTS.index(slot)*11 + 8
            prior = capability_bits({'slot': slot, 'model_slot': models[slot]})
            if slot != 4:
                # Untouched: bit-for-bit the prior the exported actor would read.
                self.assertEqual(watched[offset:offset+3], plain[offset:offset+3])
                self.assertEqual(watched[offset:offset+3], prior, f'slot {slot} collateral change')
        offset = base + SLOTS.index(4)*11 + 8
        self.assertEqual(watched[offset], 1.0, 'slot 4 inside card not credited')
        self.assertEqual(watched[offset:offset+3], [1.0, 0.0, 0.0])


class RewardTermTests(unittest.TestCase):
    """Every term `_step` pays must be declared, and none may pay for nothing.

    `outcome` used to look unpaid because all 512 recorded games ended on the
    rulebook timeout rather than by base destruction - it was in fact paid in
    full, at 55.9% of the total. The real defect was the opposite: team-level
    `control` and `outcome` dominate the sum while nothing paid the sentry for
    standing on ground or for using the terrain its chassis can reach.
    """

    def test_declared_terms_cover_every_term_paid(self):
        from rmuc2027_research.rl.session import REQUIRED_REWARD_TERMS
        config = load_config()
        self.assertEqual(set(config['rewards']), set(REQUIRED_REWARD_TERMS))
        for weight in config['rewards'].values():
            self.assertTrue(math.isfinite(float(weight)))
        # The two team-level weights must stay below the sentry-level ones they
        # compete with, or "stand still while the allies hold ground" wins.
        self.assertLess(config['rewards']['control'], config['rewards']['position_control'])
        self.assertLessEqual(config['rewards']['outcome'], 2.0)

    def test_omitting_a_term_falls_back_to_the_default_weight(self):
        # `rewards` is merged over the default file rather than replaced, so a
        # config that omits a term inherits the default weight instead of
        # silently dropping the signal. Assert the fallback is the default.
        default = json.loads(rl_session.DEFAULT_CONFIG.read_text())
        config = copy.deepcopy(default)
        del config['rewards']['terrain_buff']
        merged = load_config(config)
        self.assertEqual(merged['rewards']['terrain_buff'], default['rewards']['terrain_buff'])

    def test_a_default_file_that_lost_a_term_is_rejected(self):
        # The failure the check exists for: the *default* file dropping a term,
        # which no per-run merge can repair.
        default = json.loads(rl_session.DEFAULT_CONFIG.read_text())
        del default['rewards']['position_control']
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sentry.json'
            path.write_text(json.dumps(default))
            original = rl_session.DEFAULT_CONFIG
            rl_session.DEFAULT_CONFIG = path
            try:
                with self.assertRaises(ValueError):
                    load_config()
            finally:
                rl_session.DEFAULT_CONFIG = original

    def test_terrain_buff_is_not_paid_for_the_attack_posture(self):
        # Attack posture triples cooling on its own, so a sentry parked in
        # attack must not collect the tunnel reward. Standing still never
        # crosses a card, so the term has to stay at exactly zero.
        from rmuc2027_research.rl.session import ArenaSession
        config = {**load_config(), 'episode_seconds': 30, 'sim_backend': 'rapier'}
        with ArenaSession(config) as session:
            session.reset(42)
            while not session.done:
                session.step([0.0, 0.0])
            self.assertEqual(session.metrics['terrain_buff_seconds'], 0.0)
            self.assertEqual(session.reward_totals['terrain_buff'], 0.0)

    def test_the_control_zones_the_new_term_reads_actually_exist(self):
        # A rename in the arena config would silently zero position_control, so
        # assert the zone ids the session hard-codes are the ones published.
        from rmuc2027_research.rl.session import CONTROL_ZONES, ArenaSession
        config = {**load_config(), 'episode_seconds': 40, 'sim_backend': 'rapier'}
        with ArenaSession(config) as session:
            session.reset(42)
            session.step([0.0, 0.0])
            for zone in CONTROL_ZONES:
                self.assertIn(zone, session.obs['zones'])
            while not session.done:
                session.step([0.0, 0.0])
            self.assertGreaterEqual(session.reward_totals['position_control'], 0.0)
            self.assertGreaterEqual(session.metrics['position_control_seconds'], 0.0)


class OutcomeCriterionTests(unittest.TestCase):
    """Training and evaluation must score the same notion of winning.

    The two used to be written out separately: `_step` produced a signed reward
    from `match.winner`, and `evaluate` produced a win/draw/loss label from
    `info['winner']`. Two expressions of one rule drift - and drift here means a
    run is optimising something the report never measures. They now call one
    function, so these tests pin the sign and the label together.
    """

    def test_the_sign_and_the_label_agree_on_every_outcome(self):
        from rmuc2027_research.rl.session import result_sign, result_label
        cases = [('red', 'red', 1, 'win'), ('red', 'blue', -1, 'loss'),
                 ('blue', 'blue', 1, 'win'), ('blue', 'red', -1, 'loss'),
                 (None, 'red', 0, 'draw'), (None, 'blue', 0, 'draw')]
        for winner, side, sign, label in cases:
            self.assertEqual(result_sign(winner, side), sign)
            self.assertEqual(result_label(winner, side, True), label)

    def test_a_finished_match_without_a_winner_is_a_draw_not_a_loss(self):
        # The bug this guards: `winner is None` read as anything other than 0
        # turns every timeout draw into a loss, and a timeout is the only ending
        # the engine ever produced across the 512 recorded games.
        from rmuc2027_research.rl.session import result_sign
        self.assertEqual(result_sign(None, 'red'), 0)

    def test_an_unfinished_rollout_is_labelled_truncated_and_unpaid(self):
        from rmuc2027_research.rl.session import result_label
        self.assertEqual(result_label(None, 'red', False), 'truncated')
        # A truncated rollout has no winner to score, so `_step` must leave the
        # outcome term untouched - it only writes it under `if terminated`.
        config = {**load_config(), 'episode_seconds': 1, 'sim_backend': 'rapier'}
        from rmuc2027_research.rl.session import ArenaSession
        with ArenaSession(config) as session:
            session.reset(42)
            while not session.done:
                session.step([0.0, 0.0])
            self.assertEqual(session.reward_totals['outcome'], 0.0)

    def test_evaluation_follows_the_training_backend_over_the_old_hard_code(self):
        # evaluate() pinned 'rapier' regardless of what the run trained on. A
        # warp_kinematic run was then scored in a different simulator than the
        # one that shaped its behaviour - the reported win rate measured the
        # backend switch, not the policy.
        #
        # Read the source rather than importing it: `learning` imports torch,
        # which cannot map its shared object under the 4 GiB address-space cap,
        # so a test that imported it would only ever prove torch is missing.
        source = (Path(rl_session.__file__).parent/'learning.py').read_text()
        self.assertNotIn("sim_backend='rapier'", source)
        self.assertIn("backend or config['sim_backend']", source)
        self.assertIn('backend=None', source)
        self.assertIn('result_label(', source)


class ActionStdTests(unittest.TestCase):
    """The exploration width has to stay inside the action box it samples in.

    PPO samples N(mean, std) and the env clips at [-1, 1]. Once std reaches the
    bound, every other sample is clipped, so the gradient describes actions the
    environment never executed - and the loss curve looks *better* as that gets
    worse, because the policy can no longer distinguish clipping from
    preference. Recording std per update is what makes that visible.
    """

    def _source(self):
        return (Path(rl_session.__file__).parent/'learning.py').read_text()

    def test_the_update_log_records_the_exploration_width(self):
        source = self._source()
        self.assertIn("'action_std': action_std(self)", source)
        self.assertIn("'action_std': {'start': start_std, 'end': end_std", source)

    def test_an_exploration_width_beyond_the_bound_is_rejected(self):
        from rmuc2027_research.rl.preflight import check_action_std
        self.assertEqual(check_action_std({'ppo': {}}), 1.0)   # default sits on the bound
        self.assertEqual(check_action_std({'ppo': {'action_std': 0.5}}), 0.5)
        for bad in (1.01, 2.0, 0.0, -0.1, float('inf'), float('nan')):
            with self.assertRaises(ValueError, msg=f'action_std={bad!r} accepted'):
                check_action_std({'ppo': {'action_std': bad}})

    def test_the_validator_key_never_reaches_sb3(self):
        # SB3's PPO takes no `action_std` argument; leaking it turns a config
        # typo into a TypeError several seconds into setup.
        source = self._source()
        self.assertIn('**_ppo_kwargs(config)', source)
        self.assertIn("if key != 'action_std'", source)


class PreflightTests(unittest.TestCase):
    """Argument mistakes must fail before anything expensive is created.

    `--total-steps 100` used to be rejected only after N engine processes had
    spawned and --run-dir existed, so the operator cleaned up a half-built run
    directory to retry a typo. `mkdir(exist_ok=False)` raised FileExistsError,
    which is neither ValueError nor RuntimeError, so re-using a run directory
    escaped as a traceback instead of a message.

    The validators live in a torch-free module so a shell preflight can run
    them where `import torch` fails outright; that is not a detail, it is the
    property that lets the checks fire before the run directory exists.
    """

    def test_the_validators_import_without_torch(self):
        # Assert the import graph, not the installed torch: `learning` pulls
        # torch in at module scope, so anything importing it is unusable on a
        # machine whose address space cannot map libtorch.
        import subprocess, sys
        script = ('import sys; sys.modules["torch"] = None; '
                  'from rmuc2027_research.rl.preflight import rollout_block, check_steps, check_action_std; '
                  'print("ok")')
        done = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                              env={**__import__('os').environ,
                                   'PYTHONPATH': f'research:packages/core'})
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), 'ok')

    def test_a_step_budget_off_the_rollout_grid_is_rejected(self):
        from rmuc2027_research.rl.preflight import rollout_block, check_steps
        config = {'ppo': {'n_steps': 64}}
        self.assertEqual(rollout_block(config, 1), 64)
        self.assertEqual(rollout_block(config, 4), 256)
        check_steps(config, 1, 640)
        check_steps(config, 4, 1024)
        check_steps(config, 1, None)          # wall-clock runs have no step budget
        for steps, envs in ((100, 1), (63, 1), (1025, 4), (0, 1), (-64, 1)):
            with self.assertRaises(ValueError, msg=f'steps={steps} envs={envs} accepted'):
                check_steps(config, envs, steps)
        with self.assertRaises(ValueError):
            rollout_block(config, 0)

    def test_the_first_run_script_uses_the_torch_free_validators(self):
        # The script's own step check must not import `learning`: on a capped
        # address space that import dies with "failed to map segment", and the
        # operator would read that as a broken install instead of a bad flag.
        script = (rl_session.ROOT/'scripts/rl-first-131-run.sh').read_text()
        self.assertIn('rmuc2027_research.rl.preflight', script)
        self.assertNotIn('rmuc2027_research.rl.learning import', script)

    def test_preflight_runs_before_the_environment_exists(self):
        # Pin the ordering, not just the checks: `check_steps` has to be called
        # ahead of `vector_env`, or the fix is cosmetic.
        source = (Path(rl_session.__file__).parent/'learning.py').read_text()
        start = source.index('\ndef train(')
        body = source[start:source.index('\ndef ', start + 10)]
        self.assertIn('check_steps(', body)
        self.assertIn('vector_env(', body)
        self.assertLess(body.index('check_steps('), body.index('vector_env('))
        self.assertLess(body.index('check_action_std('), body.index('vector_env('))
        # The CLI must also reject a bad budget before train() is entered, so
        # --run-dir is never created for a flag that can never work.
        cli = Path(rl_session.__file__).parent/'__main__.py'
        cli_body = cli.read_text()
        self.assertLess(cli_body.index('check_steps(config'),
                        cli_body.index('train(config, args.run_dir'))

    def test_an_existing_run_directory_is_a_message_not_a_traceback(self):
        # FileExistsError is not caught by main()'s handlers, so mkdir must be
        # wrapped. The one remaining `exist_ok=False` has to be inside the
        # wrapper itself, and every fresh run directory must go through it.
        from rmuc2027_research.rl import __main__ as cli
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'run'
            self.assertEqual(cli.fresh_directory(path, 'train'), path)
            with self.assertRaises(ValueError) as caught:
                cli.fresh_directory(path, 'train')
        self.assertIn('already exists', str(caught.exception))
        self.assertIn(str(path), str(caught.exception))
        source = Path(cli.__file__).read_text()
        self.assertEqual(source.count('.mkdir(parents=True, exist_ok=False)'), 1)
        for command in ('random', 'benchmark'):
            self.assertIn(f'fresh_directory(output, \'{command}\')', source)


class TrainingClockTests(unittest.TestCase):
    """Two denominators for one run, and neither may drift into the other.

    `sampling_seconds` runs from the first `_on_rollout_start`, so it contains
    operator spawn, warp's lazy compile and the navigation mesh. Over 1988
    rollouts that is noise; over one it is the whole denominator, which is how
    a run sustaining 25 steps/s reported 12.57 and looked like a 2x regression.
    The fix splits the first rollout out rather than redefining the old field,
    because the 90,709 steps/h baseline in the plan doc was measured with the
    old definition.
    """

    def _source(self):
        return (Path(rl_session.__file__).parent/'learning.py').read_text()

    def _body(self, name):
        source = self._source()
        start = source.index(f'\n    def {name}(')
        # `_on_training_end` is the last method on the class, so the next
        # `def` is the module-level `_actor_parameters` - hence the two-space
        # indent in the search, falling back to the class boundary.
        ends = [index for index in (source.find('\n    def ', start + 10),
                                    source.find('\ndef ', start + 10),
                                    source.find('\nclass ', start + 10)) if index > 0]
        return source[start:min(ends)]

    def test_legacy_fields_keep_their_denominator(self):
        # The regression this guards is a one-word edit: swapping the divisor
        # for `sampling_seconds_excluding_first` would make the new 24h run
        # incomparable with the 22h one it is being compared against.
        source = self._source()
        self.assertIn("'rl_steps_per_second': (model.num_timesteps-previous_steps)"
                      '/max(callback.sampling_seconds, 1e-9)', source)
        self.assertIn("'simulation_seconds_per_second': (model.num_timesteps-previous_steps)"
                      "*config['rl_seconds']/max(callback.sampling_seconds, 1e-9)", source)
        self.assertNotIn("'rl_steps_per_second': (model.num_timesteps-previous_steps)/"
                         'callback.sampling_seconds_excluding_first', source)

    def test_first_rollout_is_recorded_as_init(self):
        body = self._body('_on_rollout_end')
        # Both buckets must be fed from the same measurement, or the split
        # silently loses seconds and the steady rate reads high.
        self.assertIn('self.sampling_seconds += seconds', body)
        self.assertIn('if not self.rollouts:', body)
        self.assertIn('self.init_seconds = seconds', body)
        self.assertIn('self.sampling_seconds_excluding_first += seconds', body)
        self.assertIn('self.rollouts += 1', body)
        # A half-closed rollout is still wall time; `--wall-seconds` can cut
        # the first rollout short, and init would otherwise stay at zero.
        end = self._body('_on_training_end')
        self.assertIn('self.init_seconds += seconds', end)

    def test_steady_rate_needs_two_rollouts(self):
        from rmuc2027_research.rl.preflight import steady_throughput
        # One rollout has no steady state, and inventing one is exactly the
        # number that started this investigation.
        self.assertEqual(steady_throughput(1024, 1024, 81.46, 81.46), (None, None))
        seconds, rate = steady_throughput(2048, 1024, 81.46, 81.46+40.55)
        self.assertAlmostEqual(seconds, 40.55)
        self.assertAlmostEqual(rate, 1024/40.55)
        # The 22h run reconstructs its own measured rate from its totals.
        _, long_rate = steady_throughput(2035712, 1024, 40.55, 80780.0)
        self.assertAlmostEqual(long_rate, 25.2, places=1)
        for bad in ((float('inf'), 1024, 1.0, 2.0), (1024, float('nan'), 1.0, 2.0),
                    (1024, 1024, float('inf'), 2.0), (1024, 1024, 1.0, float('nan'))):
            with self.assertRaises(ValueError, msg=f'{bad!r} accepted'):
                steady_throughput(*bad)
        with self.assertRaises(ValueError):
            steady_throughput(2048, 1024, -1.0, 2.0)

    def test_setup_seconds_covers_vector_env(self):
        # Spawning the operators is the expensive part that is not sampling;
        # if it were measured after `vector_env`, the field would report the
        # cost of building a callback and prove nothing.
        source = self._source()
        start = source.index('\ndef train(')
        body = source[start:source.index('\ndef ', start + 10)]
        self.assertLess(body.index('vec = vector_env('), body.index('setup_seconds = '))
        self.assertIn("'setup_seconds': setup_seconds", body)
        self.assertIn("'rl_rollouts': callback.rollouts", body)


class EpisodeSmokeScriptTests(unittest.TestCase):
    """`outcome` is paid only on a finished match, so a 32 s episode proves nothing.

    The one-rollout smoke runs 16 envs x 64 steps x 0.5 s = 32 simulated
    seconds inside a 420 s match, so every episode is truncated and the outcome
    term stays at zero by construction. Nothing on disk could distinguish "the
    win reward is not wired up" from "the match never ended". The episode
    script closes that hole by ending a real match through the referee.
    """

    def _script(self):
        return rl_session.ROOT/'scripts/rl-episode-smoke.py'

    def test_the_script_ends_the_match_through_the_referee(self):
        source = self._script().read_text()
        self.assertIn("queue_referee({'kind': 'forfeit', 'team': 'blue'})", source)
        # Truncating instead would leave the term unpaid and the script green.
        self.assertIn('terminated', source)
        self.assertIn('truncated', source)
        # Must not import learning.py: it imports torch at module scope, so on
        # a capped address space this script would die before doing anything.
        # Checked against the import block, since the docstring names both
        # modules to explain why they are avoided.
        code = '\n'.join(source.splitlines()[source.splitlines().index('import argparse'):])
        self.assertNotIn('rl.learning', code)
        self.assertNotIn('stable_baselines3', code)

    def test_the_smoke_script_runs_it_before_printing_the_long_run(self):
        script = (rl_session.ROOT/'scripts/rl-first-131-run.sh').read_text()
        self.assertIn('== 5/6', script)
        self.assertIn('== 6/6', script)
        self.assertEqual(script.count('== 5/5'), 0)
        self.assertEqual(script.count('== 6/6'), 1)
        # Order matters: the outcome proof has to sit between the PPO smoke and
        # the long-run instructions, or an operator reads the command first.
        self.assertLess(script.index('rl-episode-smoke.py'),
                        script.index('--wall-seconds 86400'))

    def test_a_finished_match_pays_the_win_once(self):
        from rmuc2027_research.rl.session import ArenaSession, load_config
        config = dict(load_config(rl_session.ROOT/'configs/rl/sentry.json'))
        config['episode_seconds'] = 32
        with ArenaSession(config) as session:
            session.reset(42)
            session.battle.queue_referee({'kind': 'forfeit', 'team': 'blue'})
            _, reward, terminated, truncated, info = session.step([0.0, 0.0])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(info['winner'], 'red')
            self.assertEqual(info['reason'], 'forfeit')
            self.assertEqual(info['reward_terms']['outcome'], config['rewards']['outcome'])
            self.assertEqual(round(reward, 9), round(sum(info['reward_terms'].values()), 9))


class ActionStdControlTests(unittest.TestCase):
    """`check_action_std` guards the configured width, not the learned one.

    The 22h run shipped with no `action_std` key, so the gate saw the 1.0
    default and passed - and then the very first update took the real width to
    1.969, ending at 2.219. All 440 updates sampled wider than the [-1, 1] box,
    clipping roughly half of every sample. A configured-value check cannot see
    that; it needs a control channel that reads the live parameter.
    """

    def _source(self):
        return (Path(rl_session.__file__).parent/'learning.py').read_text()

    def test_the_clamp_never_raises_the_std(self):
        source = self._source()
        start = source.index('\n    def apply_action_std_control(')
        body = source[start:source.index('\n    def ', start + 10)]
        # log_std is the parameter and the request is in std space: clamping to
        # `max=bound` instead of `max=math.log(bound)` would clamp ~0.0 to ~1.0
        # and do nothing at all, which looks like success and fixes nothing.
        self.assertIn('torch.clamp_(self.policy.log_std.data, max=math.log(bound))', body)
        self.assertNotIn('max=bound', body)
        # And it must be one-sided: a lower clamp would also pull up a policy
        # that had legitimately narrowed, which is not this channel's job.
        self.assertNotIn('min=', body)
        # The event has to carry both ends or the operator cannot tell a real
        # correction from a no-op.
        self.assertIn("'before': before, 'after': after", body)

    def test_the_control_reads_between_rollouts(self):
        source = self._source()
        # The model's own train() override, not the module-level train(): both
        # are named `train` and the module-level one comes later in the file.
        start = source.index('\n    def train(self):')
        body = source[start:source.index('super().train()', start)]
        # Same slot as the learning-rate channel: before the rollout loop
        # starts, so a clamp cannot land halfway through a buffer that is
        # already collecting samples at the old width.
        self.assertIn('self.apply_action_std_control()', body)
        self.assertIn('self.apply_learning_rate_control()', body)
        # Both must land before the rollout loop; their order relative to each
        # other is immaterial, so only the boundary is asserted.
        self.assertLess(body.index('self.apply_action_std_control()'), len(body))
        # The path has to be wired in train(), or the channel silently no-ops
        # for every run that nobody configured.
        self.assertIn("model.action_std_control_path = str((output/'action-std.json').resolve())",
                      source)

    def test_an_out_of_range_bound_is_rejected(self):
        # Guard the arithmetic the channel depends on without importing torch.
        import math
        for bad in (0.0, -0.5, 1.5, float('nan'), float('inf')):
            self.assertFalse(0 < bad <= 1.0 and math.isfinite(bad), f'{bad!r} accepted')
        self.assertTrue(0 < 1.0 <= 1.0 and math.isfinite(1.0))
        self.assertTrue(0 < 0.5 <= 1.0 and math.isfinite(0.5))
        # The rejection path must report, not raise: a malformed control file
        # arriving mid-run cannot be allowed to kill a 24h session.
        source = self._source()
        self.assertIn("'kind': 'action-std-control-error'", source)
        start = source.index('\n    def apply_action_std_control(')
        body = source[start:source.index('\n    def ', start + 10)]
        self.assertIn('except (OSError, ValueError, KeyError, TypeError)', body)
        self.assertNotIn('raise ', body.split('except')[1].split('with torch.no_grad()')[0])


if __name__ == '__main__':
    unittest.main()
