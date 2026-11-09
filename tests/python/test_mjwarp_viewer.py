"""Rendering consumes integrated state without a second CPU physics step."""
import importlib.util
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch, MagicMock


@unittest.skipUnless(importlib.util.find_spec('mujoco'), 'Optional MuJoCo unavailable')
class NativeViewerTests(unittest.TestCase):
    def test_draw_only_updates_kinematics_and_reuses_window(self):
        import mujoco
        import numpy as np
        from rmuc2027_research.rl.mjwarp.viewer import NativeView
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body pos="0 0 1"><freejoint/><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        qpos = data.qpos.copy(); qpos[2] = 2
        physics = SimpleNamespace(runtime=SimpleNamespace(cpu_model=model), host_data=[data],
            last_state={'qpos': qpos[None, :], 'qvel': data.qvel[None, :].copy(), 'time': np.array([.5])})
        handle = MagicMock()
        handle.user_scn = mujoco.MjvScene(model, maxgeom=32)
        handle.lock.side_effect = lambda: nullcontext()
        handle.is_running.return_value = True
        with patch('mujoco.viewer.launch_passive', return_value=handle) as launch, patch('mujoco.mj_step', side_effect=AssertionError('CPU integration forbidden')):
            view = NativeView()
            view.draw(physics)
            self.assertEqual(data.qpos[2], 2)
            self.assertAlmostEqual(data.geom_xpos[0, 2], 2)
            self.assertEqual(data.time, .5)
            view.last_draw = 0
            view.draw(physics)
            self.assertEqual(launch.call_count, 1)
            view.key(32)
            self.assertTrue(view.paused)
            view.close()
            handle.close.assert_called_once()


    def test_label_lifecycle_and_scene_capacity(self):
        import mujoco
        from rmuc2027_research.rl.mjwarp.viewer import NativeView
        model = mujoco.MjModel.from_xml_string('<mujoco/>')
        view = NativeView()
        view.handle = SimpleNamespace(user_scn=mujoco.MjvScene(model, maxgeom=16))
        frame = {'robots':[dict(team='red',slot=7,x=1,y=2,z=.2,hp=350,max_hp=400,alive=True)],
                 'bases':{'blue':dict(x=25,y=7,z=0,hp=9500,max_hp=10000)},
                 'outposts':{'red':dict(owner='blue',x=10,y=11,z=0,hp=0,max_hp=1000)}}
        view.update_labels(frame)
        view._draw_labels()
        scene = view.handle.user_scn
        self.assertEqual(scene.ngeom, 3)
        self.assertEqual(scene.geoms[0].label,'RED 7 | HP 350/400')
        self.assertEqual(scene.geoms[2].label,'BLUE OUTPOST | HP 0/1000 | DEAD')
        self.assertGreater(scene.geoms[0].rgba[0],scene.geoms[0].rgba[2])
        self.assertGreater(scene.geoms[2].rgba[2],scene.geoms[2].rgba[0])
        view.rl_team='red'
        view.update_labels(frame)
        view._draw_labels()
        self.assertTrue(scene.geoms[0].label.endswith(' | RL'))
        self.assertNotIn(' | RL',scene.geoms[1].label)
        frame['robots'][0].update(x=3,hp=0,alive=False)
        view.update_labels(frame)
        view._draw_labels()
        self.assertEqual(scene.ngeom,3)
        self.assertEqual(scene.geoms[0].pos[0],3)
        self.assertIn('DEAD',scene.geoms[0].label)
        frame['robots'][0].update(hp=400,alive=True)
        view.update_labels(frame)
        view._draw_labels()
        self.assertNotIn('DEAD',scene.geoms[0].label)
        view.labels_enabled=False
        view.update_labels(frame)
        view._draw_labels()
        self.assertEqual(scene.ngeom,0)
