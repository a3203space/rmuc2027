"""Native MuJoCo window showing live MJWarp physics and full Arena rules."""
import argparse
from contextlib import nullcontext
import math
import os
import time


class NativeView:
    def __init__(self, fps=30, labels=True):
        self.handle = None
        self.model = None
        self.last_draw = 0.
        self.interval = 1/fps
        self.paused = False
        self.labels_enabled = labels
        self.labels = []

    def update_labels(self, frame):
        """Cache display-only facts from the last completed referee frame."""
        self.labels = []
        if not self.labels_enabled:
            return
        from rmuc2027_core.engine import _config
        designs = _config.get('robot_designs', {})
        for robot in frame.get('robots', []):
            design = designs.get(str(robot.get('model_slot', robot.get('slot'))), {})
            height = design.get('extended_m', [0, 0, .8])[2]
            self._label(robot, robot['team'], str(robot['slot']), height + .25)
        for kind, title in (('bases', 'BASE'), ('outposts', 'OUTPOST')):
            for site, building in frame.get(kind, {}).items():
                top = max((p['center'][2] + p.get('height', .15)/2
                           for p in building.get('armor_plates', [])),
                          default=building.get('z', 0) + 1.5)
                self._label(building, building.get('owner', site), title,
                            top-building.get('z', 0)+.4)

    def _label(self, unit, team, name, height):
        hp, maximum = unit.get('hp', 0), unit.get('max_hp', 0)
        dead = not unit.get('alive', hp > 0)
        text = f"{team.upper()} {name} | HP {max(0, hp):.0f}/{maximum:.0f}"
        if dead:
            text += ' | DEAD'
        color = (1., .3, .3, 1.) if team == 'red' else (.3, .65, 1., 1.)
        self.labels.append((text, (unit['x'], unit['y'], unit.get('z', 0)+height), color))

    def _draw_labels(self):
        import mujoco
        import numpy as np
        scene = self.handle.user_scn
        scene.ngeom = 0
        for text, position, color in self.labels[:scene.maxgeom]:
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL,
                               np.zeros(3), np.asarray(position), np.eye(3).ravel(),
                               np.asarray(color, dtype=np.float32))
            geom.label = text
            scene.ngeom += 1

    def key(self, code):
        if code == 32:
            self.paused = not self.paused

    def running(self):
        return self.handle is None or self.handle.is_running()

    def lock(self):
        return self.handle.lock() if self.handle is not None else nullcontext()

    def draw(self, physics):
        import mujoco
        import mujoco.viewer
        now = time.monotonic()
        runtime = physics.runtime
        if self.model is not runtime.cpu_model:
            self.close()
            self.model = runtime.cpu_model
            self.handle = mujoco.viewer.launch_passive(self.model, physics.host_data[0], key_callback=self.key)
            with self.handle.lock():
                self.handle.cam.lookat[:] = [14, 7.5, 0]
                self.handle.cam.distance = 35
                self.handle.cam.azimuth = 90
                self.handle.cam.elevation = -55
        if now-self.last_draw < self.interval or not self.handle.is_running():
            return
        # GPU result already downloaded by the rule bridge. Kinematics only:
        # do not integrate or solve the scene again on the CPU for rendering.
        data = physics.host_data[0]
        with self.handle.lock():
            data.qpos[:] = physics.last_state['qpos'][0]
            data.qvel[:] = physics.last_state['qvel'][0]
            data.time = float(physics.last_state['time'][0])
            mujoco.mj_kinematics(self.model, data)
            self._draw_labels()
        self.handle.sync(state_only=True)
        self.last_draw = now

    def close(self):
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.model = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/rl/sentry-mjwarp-hour.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--duration', type=int, default=420)
    parser.add_argument('--trace-dir', help='Record motion diagnostics to a new directory')
    parser.add_argument('--no-labels', action='store_true', help='Hide team and HP labels')
    parser.add_argument('--speed', type=float, default=1.)
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error('speed must be finite and positive')
    if not 1 <= args.duration <= 420:
        parser.error('duration must be in [1, 420]')
    if not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY')):
        parser.error('Run from an Ubuntu desktop terminal with DISPLAY or WAYLAND_DISPLAY available')
    from ..session import load_config, resource
    from ...strategies.battle import Battle
    from rmuc2027_core.engine import DT
    cfg = load_config(args.config)
    sources = (resource(cfg['friendly']), resource(cfg['opponent']))
    red, blue = sources if cfg['team'] == 'red' else sources[::-1]
    print('Native MJWarp viewer: full 5v5 scripted AI; space pauses, mouse rotates/zooms. No RL policy is loaded.', flush=True)
    view = NativeView(labels=not args.no_labels)
    with Battle(red, blue, args.seed, args.duration, cfg['map'], sandbox='trusted-local', side_lineups=cfg.get('lineups'),
                timeout=cfg['policy_timeout_s'], record_replay=False,
                navigation_mode=cfg['navigation_mode'], battle_settings={'drones_enabled': cfg['drones_enabled'], 'timed_assembly': cfg['timed_assembly']},
                fixed_policy_hz=cfg['fixed_policy_hz'],
                physics_backend='mjwarp', sim_device=args.device) as battle:
        from .diagnose import MotionTrace
        trace = MotionTrace(args.trace_dir, battle, cfg) if args.trace_dir else None
        view.update_labels(battle.frames[0])
        physics = battle.match.external_physics
        original_step = physics.step
        def visible_step(request):
            with view.lock():
                result = original_step(request)
            if trace is not None:
                trace.physics(request, result)
            view.draw(physics)
            return result
        physics.step = visible_step
        last_tick = -1
        try:
            while view.running() and battle.match.status != 'finished':
                if view.paused:
                    time.sleep(.02)
                    continue
                started = time.monotonic()
                frame = battle.step()
                if trace is not None:
                    trace.record(frame)
                view.update_labels(frame)
                view.draw(physics)
                if battle.diagnostics:
                    raise RuntimeError(f'Strategy error: {battle.diagnostics[-1]}')
                if int(frame['time']) != last_tick:
                    last_tick = int(frame['time'])
                    print(f"sim={frame['time']:.2f}s red_base={frame['bases']['red']['hp']:.0f} blue_base={frame['bases']['blue']['hp']:.0f}", flush=True)
                time.sleep(max(0, DT/args.speed-(time.monotonic()-started)))
        finally:
            if trace is not None:
                trace.close()
            view.close()


if __name__ == '__main__':
    main()
