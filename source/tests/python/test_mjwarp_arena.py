"""Actual Rust scene conversion and full-match training interface regressions."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import unittest

from rmuc2027_core.engine import binary_path
from rmuc2027_core.paths import ROOT

AVAILABLE = all(importlib.util.find_spec(n) for n in ('mujoco', 'mujoco_warp', 'warp', 'numpy'))


def scene(lineups=None):
    process = subprocess.Popen([str(binary_path())], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, env={**os.environ, 'RMUC_EXTERNAL_PHYSICS': '1'})
    try:
        process.stdin.write(json.dumps(dict(op='init', seed=42, duration=420, map='rmuc2027', lineups=lineups))+'\n')
        process.stdin.flush()
        init = json.loads(process.stdout.readline())
        if 'error' in init:
            raise ValueError(init['error'])
        process.stdin.write('{"op":"step","commands":{}}\n'); process.stdin.flush()
        request = json.loads(process.stdout.readline())
        assert request['op'] == 'external_physics'
        return request
    finally:
        process.kill(); process.wait()
        process.stdin.close(); process.stdout.close()


@unittest.skipUnless(AVAILABLE, 'Optional MJWarp dependencies unavailable')
class SceneTests(unittest.TestCase):
    def test_loaded_body_constraint_weights_match_mujoco_recomputation(self):
        import mujoco
        import numpy as np
        from types import SimpleNamespace
        from rmuc2027_research.rl.mjwarp.arena import ArenaPhysics,MODEL_FIELDS
        physics=ArenaPhysics('cpu')
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="b1" simple="false"><freejoint/><geom type="box" size=".3 .2 .1" mass="10"/></body></worldbody></mujoco>')
        physics.runtime=SimpleNamespace(cpu_model=model)
        physics.host_models=[{n:np.array(getattr(model,n),copy=True) for n in MODEL_FIELDS}]
        physics.host_data=[mujoco.MjData(model)];physics.prepared_geometry=[{}];physics.body_ids={'b1':1}
        try:
            inertia=model.body_inertia[1].copy()
            for mass in (20.,12.,30.):
                body=dict(id=1,dynamic=True,mass=mass,pose=dict(pos=[0.,0.,1.],quat=[1.,0.,0.,0.]),
                          velocity=[0.,0.,0.],angular=[0.,0.,0.],force=[0.,0.,0.],torque=[0.,0.,0.],
                          mass_properties=dict(local_com=[0.,0.,.1],principal_inertia_local_frame=[0.,0.,0.,1.],
                                               inv_principal_inertia_sqrt=(1/np.sqrt(inertia*mass/10)).tolist()))
                physics.prepare(dict(bodies=[body]),[])
                actual=model.body_invweight0[1].copy()
                mujoco.mj_setConst(model,physics.host_data[0])
                np.testing.assert_allclose(actual,model.body_invweight0[1],rtol=1e-12)
        finally:
            physics.runtime=None;physics.close()

    def test_all_models_compile_with_real_collision_geometry(self):
        import mujoco
        from rmuc2027_research.rl.mjwarp.arena import ArenaPhysics
        physics = ArenaPhysics('cpu')
        designs = json.loads((ROOT/'configs/rules.json').read_text())['robot_designs']
        self.assertGreaterEqual(len(designs), 18)
        for name, design in designs.items():
            with self.subTest(model=name):
                slot = {'heavy': 1, 'infantry': 3, 'aerial': 6, 'sentry': 7}[design['role']]
                lineup = [{'slot': s, 'model': int(name) if s == slot else m}
                          for s, m in ((1, 2711), (3, 2731), (4, 2732), (6, 2761), (7, 2771))]
                request = scene({'red': lineup, 'blue': lineup})
                model = mujoco.MjModel.from_xml_string(physics.build_xml(request, physics.geometry(request)))
                self.assertGreater(model.ngeom, 200)
                self.assertGreater(model.nv, 40)
                self.assertTrue(all(b['mass'] > 0 for b in request['bodies'] if b['dynamic']))
        physics.close()

    def test_contact_export_uses_canonical_ids_when_legacy_fields_contain_garbage(self):
        import mujoco
        from rmuc2027_research.rl.mjwarp.arena import contact_geometry_ids
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="plane" size="1 1 .1"/><body pos="0 0 .05"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.assertGreater(data.ncon, 0)
        contact = data.contact[0]
        expected = tuple(int(g) for g in contact.geom)
        contact.geom1 = 1816213317
        contact.geom2 = -123
        self.assertEqual(contact_geometry_ids(contact, model.ngeom), expected)
        contact.geom[0] = model.ngeom
        with self.assertRaisesRegex(RuntimeError, 'contact.geom'):
            contact_geometry_ids(contact, model.ngeom)

    def test_batched_contact_forces_match_full_mujoco_export(self):
        import mujoco
        import numpy as np
        from rmuc2027_research.rl.mjwarp.arena import ArenaPhysics
        from rmuc2027_research.rl.mjwarp.runtime import BatchPhysics
        xml = '<mujoco><worldbody><geom type="plane" size="10 10 .1"/><body name="b10" pos="0 0 .04"><freejoint/><geom type="sphere" size=".1" mass="100"/></body></worldbody></mujoco>'
        physics = ArenaPhysics('cpu')
        try:
            physics.runtime = BatchPhysics(xml, 2, 'cpu', nconmax=64, njmax=128)
            runtime = physics.runtime
            actual = physics.read_contacts()
            for world in range(2):
                cpu = mujoco.MjData(runtime.cpu_model)
                runtime.mjw.get_data_into(cpu, runtime.cpu_model, runtime.data, world_id=world)
                expected = []
                for c in range(cpu.ncon):
                    force = np.zeros(6)
                    mujoco.mj_contactForce(runtime.cpu_model, cpu, c, force)
                    impulse = cpu.contact[c].frame.reshape(3, 3).T @ force[:3] * runtime.timestep
                    if np.linalg.norm(impulse[:2]) >= 3:
                        expected.append(float(np.linalg.norm(impulse)))
                np.testing.assert_allclose([v['impulse'] for v in actual[world]], expected, rtol=1e-5)
            # Vertical floor forces must not produce horizontal-impact events.
            self.assertEqual(actual, [[], []])
        finally:
            physics.close()

    def test_interleaved_worlds_keep_independent_geometry_cache(self):
        from copy import deepcopy
        from unittest.mock import patch
        from rmuc2027_research.rl.mjwarp.arena import ArenaPhysics
        physics = ArenaPhysics('cpu')
        first = {'id': 1, 'body': None, 'sensor': False, 'pose': {'pos': [0, 0, 0], 'quat': [1, 0, 0, 0]}, 'shape': {'Cuboid': {'half_extents': [1, 1, 1]}}}
        second = deepcopy(first)
        second['shape']['Cuboid']['half_extents'] = [2, 2, 2]
        with patch.object(physics, 'flatten', wraps=physics.flatten) as flatten:
            for _ in range(10):
                a = physics.geometry({'colliders': [first]})
                b = physics.geometry({'colliders': [second]})
            self.assertEqual(flatten.call_count, 2)
            self.assertEqual(a[0][2][3], [1, 1, 1])
            self.assertEqual(b[0][2][3], [2, 2, 2])
            changed = deepcopy(first)
            changed['shape']['Cuboid']['half_extents'] = [3, 3, 3]
            self.assertEqual(physics.geometry({'colliders': [changed]})[0][2][3], [3, 3, 3])
            self.assertEqual(flatten.call_count, 3)
        physics.geometry_cache_limit = 4
        for _ in range(20):
            physics.geometry({'colliders': [deepcopy(first)]})
        # LRU eviction keeps memory bounded even when descriptors change each step.
        self.assertLessEqual(len(physics.geometry_cache), 4)
        physics.close()
        self.assertEqual(len(physics.geometry_cache), 0)

    def test_cached_preparation_restores_each_world_geometry(self):
        import mujoco
        import numpy as np
        from types import SimpleNamespace
        from copy import deepcopy
        from rmuc2027_research.rl.mjwarp.arena import ArenaPhysics, MODEL_FIELDS
        physics = ArenaPhysics('cpu')
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="c1_0" type="box" size="1 1 1"/></worldbody></mujoco>')
        physics.runtime = SimpleNamespace(cpu_model=model)
        physics.host_models = [{n: np.array(getattr(model, n), copy=True) for n in MODEL_FIELDS} for _ in range(2)]
        physics.host_data = [mujoco.MjData(model) for _ in range(2)]
        physics.prepared_geometry = [{}, {}]
        physics.geom_ids = {'c1_0': 0}
        first = {'id': 1, 'body': None, 'enabled': True, 'sensor': False, 'pose': {'pos': [0, 0, 0], 'quat': [1, 0, 0, 0]}, 'shape': {'Cuboid': {'half_extents': [1, 1, 1]}}}
        second = deepcopy(first)
        second['shape']['Cuboid']['half_extents'] = [2, 2, 2]
        for _ in range(3):
            for world, collider in enumerate((first, second)):
                request = {'colliders': [collider], 'bodies': []}
                cpu = physics.prepare(request, physics.geometry(request), world)
                self.assertIs(cpu, physics.host_data[world])
                np.testing.assert_array_equal(model.geom_size[0], [world+1]*3)
        from collections import OrderedDict
        physics.runtime.num_worlds = 1
        physics.initial_model = {n: np.array(getattr(model, n), copy=True) for n in MODEL_FIELDS}
        physics.host_states = OrderedDict()
        keys = [object(), object()]
        from unittest.mock import patch
        with patch.object(physics.mj, 'MjData', wraps=mujoco.MjData) as allocate:
            for _ in range(3):
                for source, collider in enumerate((first, second)):
                    request = {'colliders': [collider], 'bodies': [], '_world_key': keys[source]}
                    physics.prepare(request, physics.geometry(request), 0)
                    np.testing.assert_array_equal(model.geom_size[0], [source+1]*3)
            self.assertEqual(allocate.call_count, 2)
        self.assertEqual(ArenaPhysics.signature({'colliders': [first]}), ArenaPhysics.signature({'colliders': [second]}))
        changed = deepcopy(second)
        changed['enabled'] = False
        self.assertNotEqual(ArenaPhysics.signature({'colliders': [first]}), ArenaPhysics.signature({'colliders': [changed]}))
        physics.runtime = None
        physics.close()

    def test_unsupported_shape_is_rejected(self):
        from rmuc2027_research.rl.mjwarp.arena import ArenaPhysics
        physics = ArenaPhysics('cpu')
        with self.assertRaises(ValueError):
            list(physics.flatten({'UnknownShape': {}}, [0, 0, 0], [1, 0, 0, 0]))

    @unittest.skipUnless(importlib.util.find_spec('stable_baselines3'), 'Optional PPO dependencies unavailable')
    def test_two_complete_matches_step_reset_and_close(self):
        import numpy as np
        from rmuc2027_research.rl.learning import vector_env
        from rmuc2027_research.rl.session import load_config
        from rmuc2027_research.rl.mjwarp import batching
        config = load_config({'sim_backend': 'mjwarp', 'sim_device': 'cpu', 'rl_seconds': .1,
                              'episode_seconds': .1})
        vec = vector_env(config, 2)
        try:
            obs = vec.reset()
            self.assertEqual(obs.shape, (2, 116))
            obs, reward, done, infos = vec.step(np.array([[.5, .5], [-.5, -.5]]))
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(reward).all())
            self.assertTrue(done.all())
            for info in infos:
                self.assertEqual(info['terminal_observation'].shape, (116,))
                self.assertTrue(info['TimeLimit.truncated'])
                self.assertEqual(set(info['bases']), {'red', 'blue'})
                self.assertEqual(info['metrics']['policy_errors'], 0)
            self.assertGreaterEqual(vec.physics.max_batch, 1)
            self.assertEqual(sum(vec.physics.batch_histogram.values()), vec.physics.batches)
            self.assertEqual(sum(size*count for size, count in vec.physics.batch_histogram.items()), vec.physics.world_steps)
            self.assertIn('step_and_sync', vec.physics.phase_seconds)
            self.assertTrue(all(value >= 0 for value in vec.physics.phase_seconds.values()))
            self.assertLessEqual(sum(vec.physics.phase_seconds.values()), vec.physics.integration_seconds)
            vec.physics.reset_metrics()
            self.assertEqual(vec.physics.world_steps, 0)
            self.assertEqual(dict(vec.physics.phase_seconds), {})
        finally:
            vec.close()
        self.assertIsNone(batching.active_service)
