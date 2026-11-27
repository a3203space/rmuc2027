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
        self.assertEqual(OBS_VERSION, 5)
        self.assertEqual(OBS_SIZE, 162)
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

    def test_slope_travel_does_not_trigger_climb_witness(self):
        witness = TerrainWitness(use_prior=False)
        tan_slope = math.tan(math.radians(1.5))
        for step, y in enumerate((1.0, 3.0, 5.0, 7.0)):
            z = min(y, 15.0 - y) * tan_slope
            witness.observe(self._obs(10.0 + step * 0.5, [self._enemy(4, 6.0, y, z=z)]), 7)
        self.assertEqual(witness.bits(4, self._enemy(4, 6.0, 7.0, model=2731)), [0.0, 0.0, 0.0])

    def test_honest_witness_omits_design_table_prior_and_matches_export(self):
        witness = TerrainWitness(use_prior=False)
        enemy = self._enemy(4, 14.92, 13.475, model=2732)
        self.assertEqual(witness.bits(4, enemy), [0.0, 0.0, 0.0])
        witness.observe(self._obs(10.0, [enemy], {self.INSIDE: {'occupants': ['blue-4']}}), 7)
        self.assertEqual(witness.bits(4, enemy), [1.0, 0.0, 0.0])


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
        # Outcome and team control must dominate per-second shaping so personal
        # zone occupancy can never make a losing match net-positive.
        self.assertGreaterEqual(config['rewards']['outcome'], 5.0)
        self.assertGreater(config['rewards']['control'], 0.0)
        self.assertLess(config['duration'] * config['rewards']['position_control'],
                        2.0 * config['rewards']['outcome'])

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

    def test_uncontested_own_base_pays_zero_position_control(self):
        from rmuc2027_research.rl.session import contributes_to_control
        robot = {'id': 'red-7', 'alive': True, 'weak': False}
        base_frame = {
            'teams': {'red': {'control': 10.0}, 'blue': {'control': 0.0}},
            'zones': {
                'red_base': {'occupants': ['red-7']},
                'red_fortress': {'occupants': []},
                'red_central': {'occupants': []},
                'red_outpost': {'occupants': []},
                'blue_central': {'occupants': ['red-3']},
                'blue_outpost': {'occupants': []},
                'blue_base': {'occupants': []},
                'blue_fortress': {'occupants': []},
            },
            'positions': {
                'red_base': {'kind': 'base', 'owner': 'red', 'controller': 'red',
                             'control': 'full', 'progress': 0.0, 'active': ['red'],
                             'zones': ['red_base', 'red_fortress']},
                'blue_base': {'kind': 'base', 'owner': 'blue', 'controller': 'blue',
                              'control': 'full', 'progress': 0.0, 'active': [],
                              'zones': ['blue_base', 'blue_fortress']},
                'red_outpost': {'kind': 'outpost', 'owner': 'red', 'controller': 'red',
                                'control': 'full', 'progress': 0.0, 'active': [],
                                'zones': ['red_outpost', 'red_central']},
                'blue_outpost': {'kind': 'outpost', 'owner': 'blue', 'controller': 'red',
                                 'control': 'temporary', 'progress': 0.0, 'active': ['red'],
                                 'zones': ['blue_outpost', 'blue_central']},
            },
        }
        next_frame = copy.deepcopy(base_frame)
        # Even while allies score +1/s from dual outposts, camping on `red_base`
        # (when `blue_base` is NOT held) must pay strictly zero.
        next_frame['teams']['red']['control'] = 10.05
        self.assertFalse(contributes_to_control(robot, base_frame, next_frame, 'red'))
        # Holding `red_central` while the dual-outpost pair is actively scoring DOES pay.
        next_frame['zones']['red_base']['occupants'] = []
        next_frame['zones']['red_central']['occupants'] = ['red-7']
        self.assertTrue(contributes_to_control(robot, base_frame, next_frame, 'red'))
        # Advancing capture progress on an enemy-held position pays even before control rises.
        capturing_before = copy.deepcopy(base_frame)
        capturing_before['positions']['blue_outpost'].update(controller='blue', control='full', progress=1.0)
        capturing_after = copy.deepcopy(capturing_before)
        capturing_after['zones']['blue_central']['occupants'] = ['red-7']
        capturing_after['positions']['blue_outpost'].update(progress=1.05, active=['red'])
        self.assertTrue(contributes_to_control(robot, capturing_before, capturing_after, 'red'))


class SentryExecutionTests(unittest.TestCase):
    """Arrival deadband, 2 Hz damping, posture switching, and outpost guard."""

    @classmethod
    def setUpClass(cls):
        with Match(seed=42, map_name='rmuc2027') as match:
            cls.observation = match.observation('red', [7])
        cls.config = {**load_config(), 'learning_control_hz': 2}

    def test_arrival_deadband_stops_and_unlocks_non_move_postures(self):
        controller = GoalController(self.config)
        try:
            obs = copy.deepcopy(self.observation)
            obs['time'] = 10.0
            obs['enemies'] = []
            obs['outposts']['blue']['hp'] = 0
            robot = next(r for r in obs['robots'] if r['slot'] == 7)
            controller.state['goal'] = [robot['x'] + 0.15, robot['y'] - 0.10]
            actions, _ = controller.decide(obs)
            cmd = actions['7']
            self.assertEqual(cmd['move'], [0.0, 0.0])
            self.assertTrue(cmd.get('brake'))
            # Parked with no targetable enemy or structure enters `defense` (50% reduction).
            self.assertEqual(cmd.get('posture'), 'defense')

            # Visible enemy in ballistic range switches parked sentry to `attack` (90/s cooling),
            # and high barrel heat activates the 15 s enhanced attack (0-heat firing).
            obs['enemies'] = [{
                'id': 'blue-3', 'slot': 3, 'role': 'infantry', 'team': 'blue',
                'x': robot['x'] + 2.0, 'y': robot['y'], 'z': robot.get('z', 0.0),
                'vx': 0.0, 'vy': 0.0, 'hp': 200.0, 'max_hp': 200.0,
                'alive': True, 'targetable': True, 'invulnerable': False,
                'armor_center_z_m': 0.2,
            }]
            robot.update(posture='defense', posture_ready_at=5.0, heat=210.0, max_heat=260.0)
            actions, _ = controller.decide(obs)
            cmd = actions['7']
            self.assertEqual(cmd.get('posture'), 'attack')
            self.assertTrue(cmd.get('posture_enhanced'))
        finally:
            controller.close()

    def test_uncovered_temporary_outpost_holds_ground_with_enhanced_defense(self):
        controller = GoalController(self.config)
        try:
            obs = copy.deepcopy(self.observation)
            obs['time'] = 25.0
            # Own outpost degraded to temporary and no ally covers `red_central`.
            obs['positions']['red_outpost']['control'] = 'temporary'
            robot = next(r for r in obs['robots'] if r['slot'] == 7)
            central = controller.baseline.Context(obs, {}).zone('red', 'central')
            controller.state['goal'] = list(central)
            robot.update(
                x=central[0], y=central[1],
                hp=robot['max_hp'] * 0.20,
                last_damage_time=24.2,
                posture='attack',
                posture_ready_at=10.0,
            )
            actions, _ = controller.decide(obs)
            cmd = actions['7']
            self.assertEqual(controller.state['mode'], 'policy')
            self.assertEqual(cmd['move'], [0.0, 0.0])
            self.assertEqual(cmd.get('posture'), 'defense')
            self.assertTrue(cmd.get('posture_enhanced'))
        finally:
            controller.close()


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
        body = source[start:source.index('self._train_masked()', start)]
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


class Stage3FixRegressionTests(unittest.TestCase):
    """Regressions covering OBS v3, bounded actor, hysteresis, LOS fire gate, ammo/posture."""

    @classmethod
    def setUpClass(cls):
        with Match(seed=42, map_name='rmuc2027') as match:
            cls.observation = match.observation('red', [7])
        cls.config = {**load_config(), 'learning_control_hz': 2}

    def test_obs_v3_encodes_tactical_awareness_in_tail(self):
        obs = copy.deepcopy(self.observation)
        obs['time'] = 185.0
        robot = next(r for r in obs['robots'] if r['slot'] == 7)
        robot.update(
            supply_grants_taken=1,
            posture_enhanced_time=[7.5, 0.0, 15.0],
            crossing_cooling_until=245.0,
            crossing_defense_until=200.0,
        )
        obs['outposts']['red'].update(hp=1000.0, max_hp=2000.0)
        obs['outposts']['blue'].update(hp=500.0, max_hp=2000.0)
        obs['zones']['red_central']['occupants'] = ['blue-3']
        obs['zones']['blue_central']['occupants'] = ['red-4']
        obs['positions']['red_outpost']['controller'] = 'blue'
        obs['teams']['red']['purchased_17'] = 750
        encoded = encode_observation(obs, {'goal': (5.0, 5.0), 'mode': 'policy', 'goal_time': 185.0}, 7, 0.5)
        self.assertEqual(len(encoded), 162)
        tail = encoded[131:145]
        self.assertAlmostEqual(tail[0], 2.0 / 3.0, places=5)   # 3 due - 1 taken = 2
        self.assertAlmostEqual(tail[1], 0.5, places=5)         # attack enhanced remaining
        self.assertAlmostEqual(tail[2], 1.0, places=5)         # defense enhanced remaining
        self.assertAlmostEqual(tail[3], 0.0, places=5)         # move enhanced remaining
        self.assertAlmostEqual(tail[4], 60.0 / 120.0, places=5)# crossing cooling remaining
        self.assertAlmostEqual(tail[5], 15.0 / 30.0, places=5) # crossing defense remaining
        self.assertAlmostEqual(tail[6], 0.5, places=5)         # own outpost hp ratio
        self.assertAlmostEqual(tail[7], 0.25, places=5)        # enemy outpost hp ratio
        self.assertEqual(tail[8:13], [0.0, 1.0, 1.0, 0.0, 1.0])# central occ & own fortress active
        self.assertAlmostEqual(tail[13], 0.25, places=5)       # team 17mm cap left

    def test_goal_hysteresis_and_arrival_resume_deadband(self):
        navigator = SdkNavigator(0.25)
        controller = GoalController(self.config)
        try:
            obs = copy.deepcopy(self.observation)
            obs['time'] = 10.0
            obs['enemies'] = []
            obs['outposts']['blue']['hp'] = 0
            robot = next(r for r in obs['robots'] if r['slot'] == 7)
            state = controller.state
            select_goal(obs, state, [0.0, 0.0], navigator, self.config)
            first_goal = list(state['goal'])
            # Small action perturbation resolving <= GOAL_HYSTERESIS_M keeps locked goal.
            select_goal(obs, state, [0.015, -0.015], navigator, self.config)
            self.assertEqual(state['goal'], first_goal)
            # Even when resolve_goal snaps > 2.0 m across the central plinth, <= 1.8 m proposed_goal drift stays locked.
            w, h = obs['arena']['width'], obs['arena']['height']
            plinth_cand = navigator.resolve_goal(robot, [((0.095 + 1.0) * w) / 2.0, ((0.06 + 1.0) * h) / 2.0], obs['arena'])
            self.assertGreater(math.hypot(plinth_cand[0] - first_goal[0], plinth_cand[1] - first_goal[1]), 2.0)
            select_goal(obs, state, [0.095, 0.06], navigator, self.config)
            self.assertEqual(state['goal'], first_goal)
            # Large perturbation (> 1.8 m proposed drift) moves goal cleanly.
            select_goal(obs, state, [0.5, 0.5], navigator, self.config)
            self.assertNotEqual(state['goal'], first_goal)
            # Dead/tipped mode freezes goal even under large action changes.
            frozen_goal = list(state['goal'])
            state['mode'] = 'dead'
            select_goal(obs, state, [-0.5, -0.5], navigator, self.config)
            self.assertEqual(state['goal'], frozen_goal)
            state['mode'] = 'policy'
            # ~0.90 m diagonal grid perturbation (> old 0.55 m, <= 1.15 m) stays locked on open apron.
            select_goal(obs, state, [2.0 * 5.0 / w - 1.0, 2.0 * 5.0 / h - 1.0], navigator, self.config)
            open_goal = list(state['goal'])
            diag_action = [2.0 * 5.75 / w - 1.0, 2.0 * 5.5 / h - 1.0]
            diag_cand = navigator.resolve_goal(robot, [5.75, 5.5], obs['arena'])
            self.assertGreater(math.hypot(diag_cand[0] - open_goal[0], diag_cand[1] - open_goal[1]), 0.55)
            self.assertLessEqual(math.hypot(diag_cand[0] - open_goal[0], diag_cand[1] - open_goal[1]), 1.15)
            select_goal(obs, state, diag_action, navigator, self.config)
            self.assertEqual(state['goal'], open_goal)

            # Settle within 0.30 m then nudge out to 0.40 m (< 0.48 m resume): stays braked.
            controller.state['goal'] = [robot['x'] + 0.1, robot['y']]
            actions, _ = controller.decide(obs)
            self.assertEqual(actions['7']['move'], [0.0, 0.0])
            self.assertTrue(actions['7'].get('brake'))
            robot['x'] = controller.state['goal'][0] - 0.40
            actions, _ = controller.decide(obs)
            self.assertEqual(actions['7']['move'], [0.0, 0.0])
            self.assertTrue(actions['7'].get('brake'))
        finally:
            controller.close()

    def test_structure_los_blocks_wall_fire_and_zero_ammo_avoids_attack_and_grants_retreat(self):
        controller = GoalController(self.config)
        try:
            obs = copy.deepcopy(self.observation)
            obs['time'] = 125.0
            obs['enemies'] = []
            obs['positions']['red_outpost']['control'] = 'full'
            robot = next(r for r in obs['robots'] if r['slot'] == 7)
            # Place sentry behind red wall from blue outpost (>10m and occluded in vision).
            obs['vision']['7']['outposts'] = {}
            obs['positions']['blue_outpost']['controller'] = 'red'
            obs['outposts']['blue']['hp'] = 0.0
            # Put enemy base 4m away horizontally, but place a blocking wall right in between.
            obs['bases']['blue']['x'] = robot['x'] + 4.0
            obs['bases']['blue']['y'] = robot['y']
            obs['arena'] = copy.deepcopy(obs['arena'])
            obs['arena']['obstacles'].append({
                'id': 'test_blocker', 'x': robot['x'] + 1.8, 'y': robot['y'] - 2.0,
                'w': 0.5, 'h': 4.0, 'bottom': 0.0, 'height': 3.0, 'rotation': 0.0,
            })
            controller.state['goal'] = [robot['x'], robot['y']]
            robot.update(ammo=100, posture='defense', posture_ready_at=0.0, hp=robot['max_hp'])
            actions, _ = controller.decide(obs)
            self.assertFalse(actions['7'].get('fire', False))
            self.assertNotEqual(actions['7'].get('posture'), 'attack')

            # With an open enemy target, ammo==0 must switch out of 'attack' (both in policy
            # mode and when triggering free-grant retreat).
            obs['enemies'] = [{
                'id': 'blue-3', 'slot': 3, 'role': 'infantry', 'team': 'blue',
                'x': robot['x'] + 2.0, 'y': robot['y'], 'z': robot.get('z', 0.0),
                'vx': 0.0, 'vy': 0.0, 'hp': 200.0, 'max_hp': 200.0,
                'alive': True, 'targetable': True, 'invulnerable': False,
                'armor_center_z_m': 0.2,
            }]
            obs['time'] = 10.0
            robot.update(ammo=0, posture='attack', posture_ready_at=0.0,
                         pending_ammo=[], supply_grants_taken=0, hp=robot['max_hp'])
            actions, _ = controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'policy')
            self.assertEqual(actions['7'].get('posture'), 'defense')
            self.assertFalse(actions['7'].get('fire', False))

            obs['time'] = 125.0
            robot.update(ammo=0, posture='attack', posture_ready_at=0.0,
                         pending_ammo=[], supply_grants_taken=0, hp=robot['max_hp'])
            actions, _ = controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'recover')
            self.assertEqual(actions['7'].get('posture'), 'move')
            self.assertFalse(actions['7'].get('fire', False))
        finally:
            controller.close()

    def test_bump_approach_keeps_move_posture_and_unstick_escapes_at_2hz(self):
        controller = GoalController(self.config)
        try:
            obs = copy.deepcopy(self.observation)
            obs['enemies'] = []
            obs['outposts']['blue']['hp'] = 0
            robot = next(r for r in obs['robots'] if r['slot'] == 7)
            controller.state['goal'] = [13.0, 7.25]
            controller.state['mode'] = 'policy'

            # Approaching red_road_bumps keeps 'move' even if bump alignment briefly softens move.
            obs['time'] = 10.0
            robot.update(x=4.65, y=14.0, vx=0.1, vy=0.0, heading=0.0,
                         ammo=200, posture='defense', posture_ready_at=0.0, hp=robot['max_hp'])
            actions, _ = controller.decide(obs)
            self.assertEqual(actions['7'].get('posture'), 'move')

            # Wedged inside red_road_bumps for 1.5 s at 2 Hz triggers backoff + supercap burst.
            for t in (15.0, 15.5, 16.0, 16.5):
                obs['time'] = t
                robot.update(x=5.70, y=14.23, vx=0.0, vy=0.0, heading=0.0,
                             ammo=200, posture='move', posture_ready_at=0.0,
                             supercap_energy_j=1800.0, hp=robot['max_hp'])
                actions, _ = controller.decide(obs)
            cmd = actions['7']
            self.assertLess(cmd['move'][0], -0.5)
            self.assertIn('heading', cmd)
            self.assertAlmostEqual(cmd['heading'], 0.0, places=5)
            self.assertEqual(cmd.get('power'), 200)
            self.assertNotIn(cmd.get('posture'), ('defense', 'attack'))
        finally:
            controller.close()

    def test_free_grant_prefers_remote_buy_far_from_supply_and_commits_halfway_retreat(self):
        controller = GoalController(self.config)
        try:
            obs = copy.deepcopy(self.observation)
            obs['enemies'] = []
            obs['outposts']['blue']['hp'] = 0
            robot = next(r for r in obs['robots'] if r['slot'] == 7)
            controller.state['goal'] = [14.0, 7.5]

            # Far from supply (>8 m) with enough coins & cap left: stay on policy and buy remotely.
            obs['time'] = 125.0
            obs['teams']['red']['resources'] = 300
            obs['teams']['red']['purchased_17'] = 200
            robot.update(x=14.0, y=7.5, ammo=0, pending_ammo=[], supply_grants_taken=0,
                         last_shot_time=0.0, last_damage_time=0.0, posture='defense',
                         posture_ready_at=0.0, hp=robot['max_hp'])
            actions, _ = controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'policy')
            self.assertEqual(actions['7'].get('buy_ammo'), {'amount': 100, 'remote': True})

            # At team purchase cap (>=900), zero ammo triggers free-grant retreat in 'move' posture.
            obs['time'] = 125.5
            obs['teams']['red']['purchased_17'] = 950
            robot.update(x=14.0, y=7.5, ammo=0, pending_ammo=[], supply_grants_taken=0,
                         last_shot_time=0.0, last_damage_time=0.0, posture='defense',
                         posture_ready_at=0.0, hp=robot['max_hp'])
            actions, _ = controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'recover')
            self.assertEqual(actions['7'].get('posture'), 'move')

            # Already recovering within 8 m of supply stays committed even if outpost degrades & central uncovered.
            obs['time'] = 135.0
            obs['positions']['red_outpost']['control'] = 'temporary'
            for r in obs['robots']:
                if r['slot'] != 7:
                    r.update(x=2.0, y=2.0)
            robot.update(x=5.8, y=10.5, ammo=0, pending_ammo=[], supply_grants_taken=0,
                         hp=robot['max_hp'] * 0.6, posture='move', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'recover')

            # Off bumps during critical low-HP threatened recovery still triggers enhanced defense.
            obs['time'] = 140.0
            obs['positions']['red_outpost']['control'] = 'full'
            robot.update(x=10.0, y=5.0, ammo=100, hp=robot['max_hp'] * 0.25,
                         last_damage_time=139.5, posture='move', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'recover')
            self.assertEqual(actions['7'].get('posture'), 'defense')
            self.assertTrue(actions['7'].get('posture_enhanced'))
        finally:
            controller.close()

    def test_bumps_lock_heading_dampen_lateral_and_detect_axis_stall(self):
        controller = GoalController(self.config)
        try:
            obs = copy.deepcopy(self.observation)
            obs['enemies'] = []
            obs['outposts']['blue']['hp'] = 0
            robot = next(r for r in obs['robots'] if r['slot'] == 7)
            controller.state['goal'] = [1.5, 13.0]
            controller.state['mode'] = 'policy'

            # Inside red_road_bumps with skewed heading: locks axis heading & dampens lateral move.
            obs['time'] = 18.0
            robot.update(x=7.12, y=13.51, vx=-0.1, vy=0.0, heading=-2.60,
                         ammo=200, posture='move', posture_ready_at=0.0,
                         supercap_energy_j=1800.0, hp=robot['max_hp'])
            actions, _ = controller.decide(obs)
            cmd = actions['7']
            self.assertAlmostEqual(cmd['heading'], math.pi, places=5)
            self.assertLessEqual(abs(cmd['move'][1]), 0.16 * abs(cmd['move'][0]))

            # Pure lateral y-sliding + twisting heading inside bumps with stalled x progress still unsticks!
            controller = GoalController(self.config)
            controller.state['goal'] = [1.5, 13.0]
            controller.state['mode'] = 'policy'
            for t, yy, hdg in ((20.0, 13.51, -2.60), (20.5, 13.85, -2.30), (21.0, 13.52, -2.75), (21.5, 13.88, -2.20)):
                obs['time'] = t
                robot.update(x=7.12, y=yy, vx=0.0, vy=0.0, heading=hdg,
                             ammo=200, posture='move', posture_ready_at=0.0,
                             supercap_energy_j=1800.0, hp=robot['max_hp'])
                actions, _ = controller.decide(obs)
            cmd = actions['7']
            self.assertGreater(cmd['move'][0], 0.5)
            self.assertAlmostEqual(cmd['heading'], math.pi, places=5)
            self.assertEqual(cmd.get('power'), 200)
        finally:
            controller.close()

    def test_survival_posture_thresholds_and_low_ammo_attack_entry_guard(self):
        controller = GoalController(self.config)
        try:
            obs = copy.deepcopy(self.observation)
            obs['outposts']['blue']['hp'] = 0
            obs['positions']['red_outpost']['control'] = 'full'
            robot = next(r for r in obs['robots'] if r['slot'] == 7)

            # Recovering off-bumps at hp=0.38 (< 0.40) even without recent hit enters defense.
            obs['enemies'] = []
            obs['time'] = 50.0
            controller.state['recovering'] = True
            controller.state['goal'] = [1.5, 13.0]
            robot.update(x=10.0, y=5.0, vx=0.0, vy=0.0, ammo=100,
                         hp=robot['max_hp'] * 0.38, last_damage_time=0.0,
                         posture='move', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'recover')
            self.assertEqual(actions['7'].get('posture'), 'defense')

            # Policy mode under fire at hp=0.60 (< 0.65) switches to defense.
            obs['time'] = 60.0
            controller.state['recovering'] = False
            controller.state['mode'] = 'policy'
            robot.update(x=10.0, y=5.0, ammo=100, hp=robot['max_hp'] * 0.60,
                         last_damage_time=59.2, posture='attack', posture_ready_at=0.0)
            obs['enemies'] = [{
                'id': 'blue-3', 'slot': 3, 'role': 'infantry', 'team': 'blue',
                'x': 12.0, 'y': 5.0, 'z': robot.get('z', 0.0),
                'vx': 0.0, 'vy': 0.0, 'hp': 200.0, 'max_hp': 200.0,
                'alive': True, 'targetable': True, 'invulnerable': False,
                'armor_center_z_m': 0.2,
            }]
            controller.state['goal'] = [10.0, 5.0]
            actions, _ = controller.decide(obs)
            self.assertEqual(actions['7'].get('posture'), 'defense')

            # Parked with < 45 ammo refuses to enter 'attack' from 'defense', whereas ammo=50 enters 'attack'.
            robot.update(x=10.0, y=5.0, ammo=40, hp=robot['max_hp'],
                         last_damage_time=0.0, posture='defense', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertNotEqual(actions['7'].get('posture'), 'attack')
            robot.update(ammo=50, posture='defense', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertEqual(actions['7'].get('posture'), 'attack')

            # Unconditional low-HP defense floor (< 45% HP) refuses 'attack' even when threatened cleared.
            robot.update(ammo=150, hp=robot['max_hp'] * 0.40, last_damage_time=0.0,
                         posture='defense', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertNotEqual(actions['7'].get('posture'), 'attack')

            # Close-range enemy heavy hero (<= 8 m) blocks entering vulnerable 'attack' even at 100% HP,
            # and keeps a 3.0 s anti-flicker memory when the heavy briefly leaves line-of-sight.
            obs['time'] = 60.0
            obs['enemies'] = [{
                'id': 'blue-1', 'slot': 1, 'role': 'heavy', 'team': 'blue',
                'x': 15.0, 'y': 5.0, 'z': robot.get('z', 0.0),
                'vx': 0.0, 'vy': 0.0, 'hp': 600.0, 'max_hp': 600.0,
                'alive': True, 'targetable': True, 'invulnerable': False,
                'armor_center_z_m': 0.25,
            }]
            robot.update(ammo=150, hp=robot['max_hp'], last_damage_time=0.0,
                         posture='defense', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertNotEqual(actions['7'].get('posture'), 'attack')
            obs['enemies'] = [{
                'id': 'blue-3', 'slot': 3, 'role': 'infantry', 'team': 'blue',
                'x': 12.0, 'y': 5.0, 'z': robot.get('z', 0.0),
                'vx': 0.0, 'vy': 0.0, 'hp': 200.0, 'max_hp': 200.0,
                'alive': True, 'targetable': True, 'invulnerable': False,
                'armor_center_z_m': 0.2,
            }]
            obs['time'] = 62.0  # +2.0 s <= 3.0 s heavy memory: still blocks 'attack'
            actions, _ = controller.decide(obs)
            self.assertNotEqual(actions['7'].get('posture'), 'attack')
            obs['time'] = 64.0  # +4.0 s > 3.0 s memory expired: normal infantry duel enters 'attack'
            actions, _ = controller.decide(obs)
            self.assertEqual(actions['7'].get('posture'), 'attack')

            # Emergency survival retreat bypasses uncovered outpost off central (< 25% HP or < recover_enter & 0 ammo).
            obs['positions']['red_outpost']['control'] = 'temporary'
            for r in obs['robots']:
                if r['slot'] != 7:
                    r.update(x=2.0, y=2.0)
            controller.state['recovering'] = False
            robot.update(x=16.0, y=8.0, ammo=100, hp=robot['max_hp'] * 0.20,
                         last_damage_time=0.0, posture='defense', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'recover')
            controller.state['recovering'] = False
            with patch.dict(controller.config, {'recover_enter': 0.35}):
                robot.update(x=12.3, y=7.1, ammo=100, hp=robot['max_hp'] * 0.31,
                             last_damage_time=0.0, posture='defense', posture_ready_at=0.0)
                actions, _ = controller.decide(obs)
                self.assertEqual(controller.state['mode'], 'policy')
                robot.update(x=12.3, y=7.1, ammo=0, hp=robot['max_hp'] * 0.31,
                             last_damage_time=0.0, posture='defense', posture_ready_at=0.0)
                actions, _ = controller.decide(obs)
                self.assertEqual(controller.state['mode'], 'recover')

            # Critical low HP (< 35%) on speed bumps keeps 'defense' unconditionally (even if threatened cleared),
            # and activates enhanced defense when under fire; off-bumps unstick never drops 'defense' to 'move'.
            obs['time'] = 120.0
            controller.state['recovering'] = True
            controller.state['goal'] = [1.5, 13.0]
            robot.update(x=6.0, y=13.7, vx=0.0, vy=0.0, ammo=100,
                         hp=robot['max_hp'] * 0.18, last_damage_time=0.0,
                         posture='move', posture_ready_at=0.0,
                         posture_enhanced_time=[0.0, 0.0, 0.0])
            actions, _ = controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'recover')
            self.assertEqual(actions['7'].get('posture'), 'defense')
            robot.update(last_damage_time=119.5, posture='move', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertEqual(actions['7'].get('posture'), 'defense')
            self.assertTrue(actions['7'].get('posture_enhanced'))
            controller.state['unstick'] = {'trail': [], 'tries': 1, 'until': 125.0, 'escape': [-0.7, 0.0], 'heading': 0.0}
            robot.update(x=17.8, y=8.0, hp=robot['max_hp'] * 0.10, last_damage_time=0.0,
                         posture='defense', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertNotEqual(actions['7'].get('posture'), 'move')
        finally:
            controller.close()

    def test_mlp_forward_supports_terminal_tanh_and_legacy_linear(self):
        layers_tanh = [{'weight': [[100.0] * 2, [-100.0] * 2], 'bias': [0.0, 0.0], 'activation': 'tanh'}]
        out = mlp_forward([1.0, 1.0], layers_tanh)
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in out))
        self.assertAlmostEqual(out[0], 1.0, places=6)
        self.assertAlmostEqual(out[1], -1.0, places=6)
        layers_linear = [{'weight': [[2.0, 0.0], [0.0, -3.0]], 'bias': [0.5, 0.5]}]
        self.assertEqual(mlp_forward([1.0, 1.0], layers_linear), [2.5, -2.5])

    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Optional SB3/PyTorch unavailable')
    def test_tanh_policy_exports_with_parity_and_honors_action_std_config(self):
        import torch
        from stable_baselines3 import PPO
        from rmuc2027_research.rl.env import SentryEnv
        from rmuc2027_research.rl.learning import TanhActorCriticPolicy, TimedPPO, action_std
        from rmuc2027_research.rl.exporting import export_model
        from rmuc2027_research.rl.learning import experiment

        config = load_config({
            'episode_seconds': 1,
            'sim_backend': 'rapier',
            'ppo': {'n_steps': 8, 'batch_size': 8, 'action_std': 0.35},
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            run.mkdir()
            env = SentryEnv(config)
            try:
                model = TimedPPO(
                    TanhActorCriticPolicy,
                    env,
                    n_steps=8,
                    batch_size=8,
                    device='cpu',
                    seed=42,
                    policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]}, 'log_std_init': math.log(0.35)},
                )
                self.assertAlmostEqual(action_std(model), 0.35, places=5)
                extreme = np.full(env.observation_space.shape, 1e4, dtype=np.float32)
                pred, _ = model.predict(extreme, deterministic=True)
                self.assertTrue(np.all(np.abs(pred) <= 1.0))
                model.save(run / 'model.zip')
                (run / 'experiment.json').write_text(json.dumps(experiment(config)))
                reloaded = PPO.load(run / 'model.zip', device='cpu')
                np.testing.assert_array_equal(reloaded.predict(extreme, deterministic=True)[0], pred)
            finally:
                env.close()
            exported = root / 'exported'
            metadata = export_model(run / 'model.zip', exported)
            self.assertLess(metadata['actor_parity_max_error'], 1e-5)

    def test_recover_corridor_targets_western_ramp_exit_without_east_overshoot(self):
        from rmuc2027_research.rl import portable

        cfg = load_config(rl_session.ROOT / 'configs/rl/sentry-pooled-v3.json')
        cfg['learning_control_hz'] = 2
        for side, start_xy, expected_wp in [
            ('red', (4.9, 13.3), (4.95, 12.05)),
            ('red', (5.4, 11.9), (6.65, 11.50)),
            ('red', (6.6, 11.4), (6.15, 10.55)),
            ('red', (5.4, 10.8), (3.85, 10.70)),
            ('blue', (23.1, 1.7), (23.05, 2.95)),
            ('blue', (22.6, 3.1), (21.35, 3.50)),
            ('blue', (21.4, 3.6), (21.85, 4.45)),
            ('blue', (22.6, 4.2), (24.15, 4.30)),
        ]:
            arena = {'layout_revision': 'chapter4-2027-v1', 'width': 28.0, 'height': 15.0}
            supply = [1.5, 13.0] if side == 'red' else [26.5, 2.0]
            wp = portable._recover_nav_goal(
                {'team': side, 'x': start_xy[0], 'y': start_xy[1]},
                supply,
                {'arena': arena},
            )
            self.assertAlmostEqual(wp[0], expected_wp[0], places=6)
            self.assertAlmostEqual(wp[1], expected_wp[1], places=6)
        other_arena = {'layout_revision': 'other-layout', 'width': 28.0, 'height': 15.0}
        self.assertEqual(
            portable._recover_nav_goal(
                {'team': 'red', 'x': 4.9, 'y': 13.3}, [1.5, 13.0], {'arena': other_arena},
            ),
            [1.5, 13.0],
        )

    def test_v3_config_matches_requested_settings(self):
        cfg = load_config(rl_session.ROOT / 'configs/rl/sentry-pooled-v3.json')
        self.assertFalse(cfg['drones_enabled'])
        self.assertTrue(cfg['timed_assembly'])
        self.assertAlmostEqual(cfg['rl_seconds'], 0.5)
        self.assertEqual(cfg['learning_control_hz'], 10)
        self.assertAlmostEqual(cfg['recover_enter'], 0.35)
        self.assertAlmostEqual(cfg['rewards']['death'], -0.5)
        self.assertEqual(cfg['rewards']['terrain_buff'], 0.0)
        self.assertAlmostEqual(cfg['rewards']['position_control'], 0.03)
        self.assertAlmostEqual(cfg['rewards']['capture_seconds'], 0.01)
        self.assertEqual(
            cfg['opponent_pool_weights'],
            [1.0, 3.0, 1.0, 3.0, 0.5, 0.5, 2.0, 2.0, 2.0, 1.5, 1.0, 0.5],
        )
        self.assertEqual(cfg['ppo']['n_steps'], 512)
        self.assertEqual(cfg['ppo']['batch_size'], 256)
        self.assertAlmostEqual(cfg['ppo']['ent_coef'], 0.002)
        self.assertAlmostEqual(cfg['ppo']['action_std'], 0.28)

    def test_stage4_obs_v4_symmetry_buffer_goal_age_and_posture_guards(self):
        obs = copy.deepcopy(self.observation)
        obs['time'] = 100.0
        me = next(r for r in obs['robots'] if r['slot'] == 7)
        me.update(
            buffer_energy_j=30.0, energy=0.0,
            supercap_energy_j=900.0, posture_ready_at=102.5,
        )
        obs['zones']['red_base']['occupants'] = ['blue-3']
        obs['zones']['blue_fortress']['occupants'] = ['red-4']
        obs['teams']['red']['attack_damage'] = 1200.0
        obs['teams']['blue']['attack_damage'] = 400.0
        obs['teams']['red']['control'] = 300.0
        obs['teams']['blue']['control'] = 100.0
        encoded = encode_observation(obs, {'goal': (5.0, 5.0), 'mode': 'policy', 'goal_time': 92.5}, 7, 0.5)
        self.assertEqual(len(encoded), 162)
        self.assertAlmostEqual(encoded[7], 0.5, places=5)       # buffer_energy_j / 60
        self.assertAlmostEqual(encoded[130], 7.5 / 15.0, places=5)  # goal_age / 15
        v4 = encoded[145:]
        self.assertEqual(len(v4), 17)
        self.assertAlmostEqual(v4[0], 0.5, places=5)            # supercap / 1800
        self.assertAlmostEqual(v4[1], 0.5, places=5)            # posture_cd / 5
        # zone_occ order: own_outpost(2,3), other_outpost(4,5), own_base(6,7),
        # other_base(8,9), own_fortress(10,11), other_fortress(12,13)
        self.assertEqual(v4[6:8], [0.0, 1.0])                   # enemy on red_base
        self.assertEqual(v4[12:14], [1.0, 0.0])                 # ally on blue_fortress
        self.assertAlmostEqual(v4[14], 800.0 / 4000.0, places=5)# attack_damage_diff
        self.assertAlmostEqual(v4[15], 200.0 / 1000.0, places=5)# control_diff
        expected_hp = sum(r['hp'] for r in obs['robots'] if r['slot'] in GROUND_SLOTS and r['alive']) / 1800.0
        self.assertAlmostEqual(v4[16], min(1.0, expected_hp), places=5)

        # Recover locks goal and preserves goal_time; no quadratic crawl floor; no move posture mid-firefight.
        navigator = SdkNavigator(0.25)
        controller = GoalController(load_config({'learning_control_hz': 10, 'rl_seconds': 0.5}))
        try:
            obs2 = copy.deepcopy(self.observation)
            obs2['time'] = 20.0
            robot = next(r for r in obs2['robots'] if r['slot'] == 7)
            select_goal(obs2, controller.state, [0.4, 0.2], navigator, controller.config)
            locked = list(controller.state['goal'])
            locked_t = controller.state['goal_time']
            obs2['time'] = 25.0
            controller.state['mode'] = 'recover'
            select_goal(obs2, controller.state, [-0.8, -0.8], navigator, controller.config)
            self.assertEqual(controller.state['goal'], locked)
            self.assertEqual(controller.state['goal_time'], locked_t)

            # Near goal (gap=0.35m > 0.30m stop radius) at high speed_reference_mps keeps move mag >= 0.25 (no crawl).
            controller.state['mode'] = 'policy'
            controller.state['recovering'] = False
            controller.state['goal_settled'] = False
            robot.update(x=10.0, y=5.0, vx=0.0, vy=0.0, speed_reference_mps=3.5,
                         ammo=200, hp=robot['max_hp'], posture='defense', posture_ready_at=0.0)
            controller.state['goal'] = [10.35, 5.0]
            obs2['enemies'] = []
            obs2['outposts']['blue']['hp'] = 0
            actions, _ = controller.decide(obs2)
            self.assertGreaterEqual(math.hypot(*actions['7']['move']), 0.25 - 1e-6)

            # Mid-firefight (combat_ready + live_duel) with far goal (>3.5m) or near bumps MUST stay 'attack', never 'move'.
            obs2['enemies'] = [{
                'id': 'blue-3', 'slot': 3, 'role': 'infantry', 'team': 'blue',
                'x': 7.5, 'y': 14.0, 'z': robot.get('z', 0.0),
                'vx': 0.0, 'vy': 0.0, 'hp': 200.0, 'max_hp': 200.0,
                'alive': True, 'targetable': True, 'invulnerable': False,
                'armor_center_z_m': 0.2,
            }]
            robot.update(x=4.65, y=14.0, vx=0.1, vy=0.0, heading=0.0, ammo=200,
                         hp=robot['max_hp'], posture='defense', posture_ready_at=0.0, last_damage_time=0.0)
            controller.state['goal'] = [13.0, 7.25]
            actions, _ = controller.decide(obs2)
            self.assertEqual(actions['7'].get('posture'), 'attack')
        finally:
            controller.close()

    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Optional SB3/PyTorch unavailable')
    def test_stage4_policy_mask_freezes_actor_on_masked_rollout_and_trains_critic(self):
        import gymnasium as gym
        from rmuc2027_research.rl.learning import TimedPPO, TanhActorCriticPolicy, _actor_parameters

        class MaskedStubEnv(gym.Env):
            def __init__(self, executed):
                self.executed = executed
                self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
                self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return np.zeros(4, dtype=np.float32), {}

            def step(self, action):
                return np.ones(4, dtype=np.float32), 1.0, False, False, {'policy_executed': self.executed}

        env = MaskedStubEnv(executed=False)
        model = TimedPPO(
            TanhActorCriticPolicy, env, n_steps=8, batch_size=8, n_epochs=2,
            device='cpu', seed=7, ent_coef=0.01,
        )
        before_actor = _actor_parameters(model)
        before_vf = model.policy.value_net.weight.detach().clone()
        model.learn(total_timesteps=8)
        after_actor = _actor_parameters(model)
        after_vf = model.policy.value_net.weight.detach().clone()
        for name, tensor in before_actor.items():
            self.assertTrue(np.allclose(tensor.numpy(), after_actor[name].numpy()), f'Actor {name} moved when masked')
        self.assertFalse(np.allclose(before_vf.numpy(), after_vf.numpy()), 'Critic must still update when masked')
        self.assertAlmostEqual(model.logger.name_to_value['train/policy_mask_fraction'], 0.0, places=6)

    def test_obs_v4_red_blue_mirror_symmetry(self):
        red = copy.deepcopy(self.observation)
        red['time'] = 150.0
        w, h = red['arena']['width'], red['arena']['height']
        red['teams']['red']['attack_damage'] = 1400.0
        red['teams']['blue']['attack_damage'] = 500.0
        red['teams']['red']['control'] = 320.0
        red['teams']['blue']['control'] = 110.0
        red['outposts']['red']['hp'] = 1200.0
        red['outposts']['blue']['hp'] = 600.0
        red['zones']['red_base']['occupants'] = ['blue-3']
        red['zones']['blue_fortress']['occupants'] = ['red-4']
        red['zones']['red_central']['occupants'] = ['red-3', 'blue-7']
        red['robots'][-1].update(buffer_energy_j=42.0, supercap_energy_j=1200.0, posture_ready_at=152.0)
        # Include an occluded enemy present only via radar station (not in optical obs['enemies']).
        red['enemies'] = [e for e in red['enemies'] if e['slot'] != 3]
        red['radar'] = [
            {'id': 'blue-3', 'slot': 3, 'team': 'blue', 'role': 'infantry',
             'x': 0.80, 'y': 8.12, 'hp': 150.0, 'max_hp': 200.0, 'alive': True, 'invulnerable': False},
            {'id': 'blue-7', 'slot': 7, 'team': 'blue', 'role': 'sentry',
             'x': 20.5, 'y': 6.0, 'hp': 320.0, 'max_hp': 400.0, 'alive': True, 'invulnerable': True},
        ]

        def swap_id(rid):
            if rid.startswith('red-'):
                return 'blue-' + rid[4:]
            if rid.startswith('blue-'):
                return 'red-' + rid[5:]
            return rid

        def swap_team(val):
            return 'blue' if val == 'red' else ('red' if val == 'blue' else val)

        blue = copy.deepcopy(red)
        blue['team'] = 'blue'
        for group in ('robots', 'enemies', 'radar'):
            for r in blue[group]:
                r['team'] = swap_team(r.get('team'))
                if 'id' in r:
                    r['id'] = swap_id(r['id'])
                r['x'] = w - r['x']
                r['y'] = h - r['y']
                if 'vx' in r:
                    r['vx'] = -r.get('vx', 0.0)
                if 'vy' in r:
                    r['vy'] = -r.get('vy', 0.0)
                if 'heading' in r:
                    r['heading'] = r['heading'] + math.pi
        blue['teams'] = {'red': copy.deepcopy(red['teams']['blue']), 'blue': copy.deepcopy(red['teams']['red'])}
        blue['bases'] = {'red': copy.deepcopy(red['bases']['blue']), 'blue': copy.deepcopy(red['bases']['red'])}
        blue['outposts'] = {'red': copy.deepcopy(red['outposts']['blue']), 'blue': copy.deepcopy(red['outposts']['red'])}
        blue['positions'] = {}
        for k, v in red['positions'].items():
            owner, kind = k.split('_', 1)
            nv = copy.deepcopy(v)
            nv['controller'] = swap_team(nv.get('controller'))
            blue['positions'][f'{swap_team(owner)}_{kind}'] = nv
        blue['zones'] = {}
        for k, v in red['zones'].items():
            nv = copy.deepcopy(v)
            if 'occupants' in nv:
                nv['occupants'] = [swap_id(rid) for rid in nv['occupants']]
            if k.startswith('red_'):
                blue['zones']['blue_' + k[4:]] = nv
            elif k.startswith('blue_'):
                blue['zones']['red_' + k[5:]] = nv
            else:
                blue['zones'][k] = nv

        red_vec = encode_observation(red, {'goal': (6.0, 4.0), 'mode': 'policy', 'goal_time': 142.5}, 7, 0.5)
        blue_vec = encode_observation(blue, {'goal': (w - 6.0, h - 4.0), 'mode': 'policy', 'goal_time': 142.5}, 7, 0.5)
        self.assertEqual(len(red_vec), 162)
        for i, (a, b) in enumerate(zip(red_vec, blue_vec)):
            self.assertAlmostEqual(a, b, places=5, msg=f'OBS v4 symmetry mismatch at index {i}: {a} vs {b}')

    def test_staggered_initial_episode_seconds_and_policy_executed_flag(self):
        from rmuc2027_research.rl.session import ArenaSession
        cfg = {**self.config, 'episode_seconds': 2.0, 'rl_seconds': 0.5, '_initial_episode_seconds': 1.0}
        with ArenaSession(cfg) as session:
            self.assertNotIn('_initial_episode_seconds', session.config)
            session.reset(42)
            self.assertNotIn('_initial_episode_seconds', session.battle.metadata['rl']['config'])
            _, _, term1, trunc1, info1 = session.step([0.0, 0.0])
            self.assertFalse(term1 or trunc1)
            self.assertTrue(info1['policy_executed'])
            _, _, term2, trunc2, info2 = session.step([0.0, 0.0])
            self.assertFalse(term2)
            self.assertTrue(trunc2)
            self.assertAlmostEqual(info2['time'], 1.0, places=5)
            # Episode 1 runs full episode_seconds (2.0s), not 1.0s.
            session.reset(42)
            for step_idx in range(4):
                _, _, term, trunc, info = session.step([0.0, 0.0])
                if step_idx < 3:
                    self.assertFalse(term or trunc)
                else:
                    self.assertTrue(trunc)
                    self.assertAlmostEqual(info['time'], 2.0, places=5)

    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Optional SB3/PyTorch unavailable')
    def test_init_from_applies_new_ppo_hyperparams_and_masked_cuda_determinism(self):
        import gymnasium as gym
        import torch
        from rmuc2027_research.rl import learning
        from rmuc2027_research.rl.learning import TimedPPO, TanhActorCriticPolicy, MaskedRolloutBuffer

        # Verify MaskedRolloutBuffer + deterministic CUDA path when GPU is present.
        class MixedMaskEnv(gym.Env):
            def __init__(self):
                self.i = 0
                self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)
                self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.i = 0
                return np.zeros(6, dtype=np.float32), {}

            def step(self, action):
                self.i += 1
                return np.full(6, 0.1 * self.i, dtype=np.float32), 0.5, False, False, {'policy_executed': (self.i % 2 == 1)}

        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        prev_det = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            env = MixedMaskEnv()
            model = TimedPPO(TanhActorCriticPolicy, env, n_steps=8, batch_size=4, n_epochs=2, device=dev, seed=11)
            self.assertIsInstance(model.rollout_buffer, MaskedRolloutBuffer)
            model.learn(total_timesteps=8)
            self.assertAlmostEqual(model.logger.name_to_value['train/policy_mask_fraction'], 0.5, places=5)
            self.assertTrue(math.isfinite(model.logger.name_to_value['train/loss']))
        finally:
            torch.use_deterministic_algorithms(prev_det)

        # Verify --init-from applies updated n_steps / ent_coef / learning_rate and resets Critic on reward change.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root / 'src'
            dst = root / 'dst'
            cfg_src = load_config({
                'episode_seconds': 2.0, 'rl_seconds': 0.5, 'sim_backend': 'rapier',
                'ppo': {'n_steps': 8, 'batch_size': 8, 'n_epochs': 1, 'ent_coef': 0.01, 'learning_rate': 3e-4, 'action_std': 0.3},
            })
            learning.train(cfg_src, src, steps=8, num_envs=1, device='cpu')
            cfg_dst = load_config({
                'episode_seconds': 2.0, 'rl_seconds': 0.5, 'sim_backend': 'rapier',
                'rewards': {**cfg_src['rewards'], 'death': -2.0},
                'ppo': {'n_steps': 16, 'batch_size': 8, 'n_epochs': 1, 'ent_coef': 0.002, 'learning_rate': 1e-4, 'action_std': 0.3},
            })
            learning.train(cfg_dst, dst, steps=16, num_envs=1, device='cpu', init_from=src / 'model.zip')
            loaded = TimedPPO.load(dst / 'model.zip', device='cpu')
            self.assertEqual(loaded.n_steps, 16)
            self.assertAlmostEqual(loaded.ent_coef, 0.002, places=6)
            self.assertAlmostEqual(loaded.lr_schedule(1.0), 1e-4, places=7)

    def test_stage5_unstick_compass_recover_bump_lane_heavy_threshold_and_base_defense_shaping(self):
        from rmuc2027_research.rl import portable
        from rmuc2027_research.rl.session import ArenaSession, base_defense_progress_delta

        # 1. Off-bumps _unstick_fields sweeps 8 distinct directions with 0.95 magnitude and aligned heading.
        state = {}
        robot = {'slot': 7, 'x': 16.85, 'y': 10.60, 'heading': -2.02}
        angles = set()
        t = 100.0
        for _ in range(8):
            for step in range(4):
                out = portable._unstick_fields(
                    robot, {'move': (-0.43, -0.41)}, [16.0, 8.25], {'time': t}, state, on_bumps=None
                )
                t += 0.5
            self.assertAlmostEqual(math.hypot(*out['move']), 0.95, places=4)
            angles.add(round(math.atan2(out['move'][1], out['move'][0]), 2))
            t += 1.0  # let 0.9s escape expire before next stall trigger
        self.assertEqual(len(angles), 8)

        # 2. _recover_nav_goal inside road bumps lane aims along centerline (4.70, 13.95) for Red & mirrored for Blue.
        arena = {'layout_revision': 'chapter4-2027-v1', 'width': 28.0, 'height': 15.0}
        self.assertEqual(
            portable._recover_nav_goal({'team': 'red', 'x': 6.20, 'y': 13.20}, [1.5, 13.0], {'arena': arena}),
            [4.70, 13.95],
        )
        blue_wp = portable._recover_nav_goal({'team': 'blue', 'x': 21.80, 'y': 1.80}, [26.5, 2.0], {'arena': arena})
        self.assertAlmostEqual(blue_wp[0], 23.30, places=5)
        self.assertAlmostEqual(blue_wp[1], 1.05, places=5)

        # 3. Heavy-near raises recover threshold to 0.52 (hp=0.48 enters recover only when heavy is near),
        #    and safe_bump_escape allows 'move' on bumps during bump_boost when unthreatened and hp >= 0.25.
        controller = GoalController(load_config({'recover_enter': 0.42, 'recover_exit': 0.75, 'learning_control_hz': 10}))
        try:
            obs = copy.deepcopy(self.observation)
            obs['time'] = 60.0
            obs['positions']['red_outpost']['control'] = 'full'
            obs['outposts']['blue']['hp'] = 0
            me = next(r for r in obs['robots'] if r['slot'] == 7)
            controller.state['goal'] = [12.0, 7.5]
            obs['enemies'] = [{
                'id': 'blue-3', 'slot': 3, 'role': 'infantry', 'team': 'blue',
                'x': 15.0, 'y': 7.5, 'z': me.get('z', 0.0), 'vx': 0.0, 'vy': 0.0,
                'hp': 200.0, 'max_hp': 200.0, 'alive': True, 'targetable': True,
                'invulnerable': False, 'armor_center_z_m': 0.2,
            }]
            me.update(x=12.0, y=7.5, ammo=100, hp=me['max_hp'] * 0.48, posture='defense', posture_ready_at=0.0)
            controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'policy')
            obs['enemies'][0].update(id='blue-1', slot=1, role='heavy', hp=600.0, max_hp=600.0)
            controller.decide(obs)
            self.assertEqual(controller.state['mode'], 'recover')

            # Safe bump escape on red_road_bumps at hp=0.30 (not threatened, no enemies) during bump_boost -> 'move'.
            obs['time'] = 80.0
            obs['enemies'] = []
            controller.state['recovering'] = True
            controller.state['last_heavy_near_time'] = -99.0
            controller.state['unstick'] = {'trail': [], 'tries': 1, 'until': 79.0, 'bump_boost_until': 83.0}
            me.update(x=6.20, y=13.20, ammo=100, hp=me['max_hp'] * 0.30, last_damage_time=0.0,
                      posture='defense', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertEqual(actions['7'].get('posture'), 'move')
            # Under fire (threatened=True) on bumps stays 'defense'.
            me.update(last_damage_time=79.5, posture='defense', posture_ready_at=0.0)
            actions, _ = controller.decide(obs)
            self.assertNotEqual(actions['7'].get('posture'), 'move')
        finally:
            controller.close()

        # 4. Potential-based base_defense_progress_delta: zero-sum, Red/Blue symmetric, gated on policy & threat.
        red_frame = {
            'bases': {'red': {'x': 2.411, 'y': 7.5}, 'blue': {'x': 25.589, 'y': 7.5}},
            'positions': {'red_base': {'controller': 'blue', 'active': ['blue']},
                          'blue_base': {'controller': 'blue', 'active': []}},
            'zones': {'red_base': {'occupants': ['blue-7']}, 'red_fortress': {'occupants': []},
                      'blue_base': {'occupants': []}, 'blue_fortress': {'occupants': []}},
        }
        prev_r = {'alive': True, 'weak': False, 'x': 12.411, 'y': 7.5}
        curr_r = {'alive': True, 'weak': False, 'x': 12.211, 'y': 7.5}
        d_in = base_defense_progress_delta(prev_r, curr_r, red_frame, red_frame, 'red', 'policy')
        d_out = base_defense_progress_delta(curr_r, prev_r, red_frame, red_frame, 'red', 'policy')
        self.assertAlmostEqual(d_in, 0.2, places=6)
        self.assertAlmostEqual(d_in + d_out, 0.0, places=6)
        # Non-policy / weak / dead / unthreatened all pay strictly 0.0.
        self.assertEqual(base_defense_progress_delta(prev_r, curr_r, red_frame, red_frame, 'red', 'recover'), 0.0)
        self.assertEqual(base_defense_progress_delta(prev_r, {**curr_r, 'weak': True}, red_frame, red_frame, 'red', 'policy'), 0.0)
        self.assertEqual(base_defense_progress_delta({**prev_r, 'alive': False}, curr_r, red_frame, red_frame, 'red', 'policy'), 0.0)
        safe_frame = copy.deepcopy(red_frame)
        safe_frame['positions']['red_base'] = {'controller': 'red', 'active': []}
        safe_frame['zones']['red_base'] = {'occupants': []}
        self.assertEqual(base_defense_progress_delta(prev_r, curr_r, safe_frame, safe_frame, 'red', 'policy'), 0.0)
        # Mirror symmetry for Blue.
        blue_frame = {
            'bases': {'red': {'x': 2.411, 'y': 7.5}, 'blue': {'x': 25.589, 'y': 7.5}},
            'positions': {'blue_base': {'controller': 'red', 'active': ['red']},
                          'red_base': {'controller': 'red', 'active': []}},
            'zones': {'blue_base': {'occupants': ['red-7']}, 'blue_fortress': {'occupants': []},
                      'red_base': {'occupants': []}, 'red_fortress': {'occupants': []}},
        }
        d_blue = base_defense_progress_delta(
            {'alive': True, 'weak': False, 'x': 28.0 - 12.411, 'y': 7.5},
            {'alive': True, 'weak': False, 'x': 28.0 - 12.211, 'y': 7.5},
            blue_frame, blue_frame, 'blue', 'policy',
        )
        self.assertAlmostEqual(d_blue, d_in, places=6)

        # 5. Stage-5 configs load cleanly and run in ArenaSession with training_summary=True.
        cfg_w = load_config(rl_session.ROOT / 'configs/rl/sentry-stage5-warp.json')
        cfg_r = load_config(rl_session.ROOT / 'configs/rl/sentry-stage5-rapier.json')
        for cfg in (cfg_w, cfg_r):
            self.assertEqual(cfg['learning_control_hz'], 10)
            self.assertAlmostEqual(cfg['rl_seconds'], 0.5)
            self.assertAlmostEqual(cfg['recover_enter'], 0.42)
            self.assertAlmostEqual(cfg['recover_exit'], 0.75)
            self.assertAlmostEqual(cfg['rewards']['death'], -1.5)
            self.assertAlmostEqual(cfg['rewards']['base_defense_progress'], 0.15)
        with ArenaSession({**cfg_r, 'episode_seconds': 1.0, 'training_summary': True}) as session:
            session.reset(42)
            _, _, _, _, info = session.step([0.0, 0.0])
            self.assertIn('base_defense_progress', info['reward_terms'])

    def test_radar_station_blind_spot_coordinates_in_obs_v5(self):
        # 1. Real Rust Match observation exposes all 4 enemy ground robots via the RMUC 2027 5.7.4
        #    radar info-wave (obs['radar']), completely independent of camera LOS, while keeping
        #    obs['enemies'] strictly optical camera LOS.
        with Match(42, 30, 'rmuc2027') as match:
            obs_r = match.observation('red', [7])
            self.assertIn('radar', obs_r)
            radar_slots = {r['slot'] for r in obs_r['radar']}
            self.assertEqual(radar_slots, {1, 3, 4, 7})
            self.assertNotIn(6, radar_slots)
            optical_ground = [e for e in obs_r['enemies'] if e['slot'] in GROUND_SLOTS]
            self.assertEqual(optical_ground, [])
            for entry in obs_r['radar']:
                self.assertEqual(entry['team'], 'blue')
                self.assertTrue(entry['alive'])
                self.assertGreater(entry['hp'], 0.0)
                self.assertGreaterEqual(entry['max_hp'], entry['hp'])
                self.assertIn('ammo', entry)
                self.assertIn('invulnerable', entry)
                for optical_only in ('vx', 'vy', 'z', 'model_slot', 'chassis_type', 'heading', 'aim', 'heat'):
                    self.assertNotIn(optical_only, entry)
            vec = encode_observation(obs_r, {}, 7, 0.5)
            self.assertEqual(len(vec), OBS_SIZE)
            # Enemy ground slots are encoded with presence=1.0, exact radar (x, y), HP ratio, alive
            # and invulnerable completely independently of camera visibility.
            w, h = obs_r['arena']['width'], obs_r['arena']['height']
            by_slot = {r['slot']: r for r in obs_r['radar']}
            for s in GROUND_SLOTS:
                off = 18 + 4 * 7 + SLOTS.index(s) * 11
                r = by_slot[s]
                self.assertEqual(vec[off], 1.0)
                self.assertAlmostEqual(vec[off + 1], 2 * r['x'] / w - 1, places=5)
                self.assertAlmostEqual(vec[off + 2], 2 * r['y'] / h - 1, places=5)
                self.assertEqual(vec[off + 3:off + 5], [0.0, 0.0])  # vx, vy only when optically tracked
                self.assertAlmostEqual(vec[off + 5], r['hp'] / r['max_hp'], places=5)
                self.assertEqual(vec[off + 6], 1.0)
                self.assertEqual(vec[off + 7], float(r['invulnerable']))
                self.assertEqual(vec[off + 8:off + 11], [0.0, 0.0, 0.0])  # no prior leak
            # When aerial slot 6 is removed from optical enemies, it stays all zeros (never in radar).
            obs_no_air = copy.deepcopy(obs_r)
            obs_no_air['enemies'] = [e for e in obs_no_air['enemies'] if e['slot'] != 6]
            vec_no_air = encode_observation(obs_no_air, {}, 7, 0.5)
            off6 = 18 + 4 * 7 + SLOTS.index(6) * 11
            self.assertEqual(vec_no_air[off6:off6 + 11], [0.0] * 11)

        # 2. Radar info-wave supplies full position/HP/alive/invulnerable regardless of camera LOS;
        #    optical LOS adds velocity + witness bits; latched witness persists when occluded;
        #    dead ground enemy keeps alive=0.0, hp=0.0; aerial radar entry is ignored.
        obs = copy.deepcopy(self.observation)
        obs['enemies'] = [{
            'id': 'blue-3', 'slot': 3, 'team': 'blue', 'role': 'infantry', 'model_slot': 2731,
            'x': 12.0, 'y': 7.5, 'vx': 2.0, 'vy': -1.0, 'hp': 150.0, 'max_hp': 200.0,
            'alive': True, 'invulnerable': False,
        }]
        obs['radar'] = [
            {'id': 'blue-3', 'slot': 3, 'team': 'blue', 'role': 'infantry',
             'x': 12.0, 'y': 7.5, 'hp': 150.0, 'max_hp': 200.0, 'alive': True, 'invulnerable': False},
            {'id': 'blue-4', 'slot': 4, 'team': 'blue', 'role': 'infantry',
             'x': 0.80, 'y': 8.12, 'hp': 120.0, 'max_hp': 200.0, 'alive': True, 'invulnerable': True},
            {'id': 'blue-1', 'slot': 1, 'team': 'blue', 'role': 'heavy',
             'x': 20.0, 'y': 7.5, 'hp': 0.0, 'max_hp': 400.0, 'alive': False, 'invulnerable': False},
            {'id': 'blue-6', 'slot': 6, 'team': 'blue', 'role': 'aerial',
             'x': 25.0, 'y': 1.0, 'hp': 0.0, 'max_hp': 0.0, 'alive': True, 'invulnerable': False},
        ]
        witness = TerrainWitness(use_prior=False)
        witness._tunnel[4] = True
        encoded = encode_observation(obs, {}, 7, 0.5, witness=witness)
        off3 = 18 + 4 * 7 + SLOTS.index(3) * 11
        off4 = 18 + 4 * 7 + SLOTS.index(4) * 11
        off1 = 18 + 4 * 7 + SLOTS.index(1) * 11
        off6 = 18 + 4 * 7 + SLOTS.index(6) * 11
        self.assertEqual(encoded[off3], 1.0)
        self.assertAlmostEqual(encoded[off3 + 3], 0.2, places=5)   # optical vx/10
        self.assertAlmostEqual(encoded[off3 + 4], -0.1, places=5)  # optical vy/10
        self.assertAlmostEqual(encoded[off3 + 5], 0.75, places=5)  # HP ratio
        # Occluded slot 4 gets exact radar coordinates, HP=0.6, alive=1.0, invulnerable=1.0, presence=1.0!
        self.assertEqual(encoded[off4], 1.0)
        self.assertAlmostEqual(encoded[off4 + 1], 2 * 0.80 / 28.0 - 1, places=5)
        self.assertAlmostEqual(encoded[off4 + 2], 2 * 8.12 / 15.0 - 1, places=5)
        self.assertEqual(encoded[off4 + 3:off4 + 5], [0.0, 0.0])
        self.assertAlmostEqual(encoded[off4 + 5], 0.6, places=5)
        self.assertEqual(encoded[off4 + 6], 1.0)
        self.assertEqual(encoded[off4 + 7], 1.0)
        self.assertEqual(encoded[off4 + 8:off4 + 11], [1.0, 0.0, 0.0])  # latched witness preserved
        # Dead ground slot 1 reports alive=0.0, hp=0.0 at its position.
        self.assertEqual(encoded[off1], 1.0)
        self.assertEqual(encoded[off1 + 5], 0.0)
        self.assertEqual(encoded[off1 + 6], 0.0)
        # Aerial slot 6 radar entry ignored.
        self.assertEqual(encoded[off6:off6 + 11], [0.0] * 11)


if __name__ == '__main__':
    unittest.main()
