"""Native MuJoCo window showing live MJWarp physics and full Arena rules."""
import argparse
from contextlib import ExitStack, nullcontext
import json
import math
import os
from pathlib import Path
import tempfile
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
        self.rl_team = None

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
        if team == self.rl_team and name == '7':
            text += ' | RL'
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


def prepare_match(args, stack):
    """Resolve scripted teams and, optionally, one exported frozen sentry actor."""
    from ..session import load_config, resource
    from rmuc2027_core.submissions import load_submission
    metadata = None
    if args.rl_model:
        model = Path(args.rl_model).expanduser().resolve()
        if not model.is_file():
            raise ValueError(f'RL checkpoint does not exist: {model}')
        metadata = json.loads((model.parent/'experiment.json').read_text())
        if metadata['config']['slot'] != 7:
            raise ValueError('--rl-model must be a sentry (slot 7) checkpoint')
    source = args.config or (metadata['config'] if metadata else 'configs/rl/sentry-mjwarp-hour.json')
    cfg = load_config(source)
    original_team = cfg['team']
    side = args.rl_team or original_team
    if args.rl_team and not args.rl_model:
        raise ValueError('--rl-team requires --rl-model')
    if metadata and getattr(args, side):
        raise ValueError(f'--{side} cannot accompany an RL model for {side}; its other robots come from the training squad')
    sources = {side: resource(cfg['friendly']), 'blue' if side == 'red' else 'red': resource(cfg['opponent'])}
    for team in ('red', 'blue'):
        if getattr(args, team):
            sources[team] = resource(getattr(args, team))
    if metadata:
        from ..exporting import export_model
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='rmuc-viewer-rl-')))/'squad'
        exported = export_model(model, directory)
        if exported['learned_slots'] != [7]:
            raise ValueError('Native sentry viewing requires exactly one learned slot: 7')
        sources[side] = directory
        other = 'blue' if side == 'red' else 'red'
        other_metadata = sources[other]/'rl-metadata.json'
        if other_metadata.exists() and json.loads(other_metadata.read_text()).get('learned_slots'):
            raise ValueError('The opposing team must be scripted when --rl-model selects a single RL sentry')
        # Preserve the learned action/control cadence even when a separate
        # display configuration is supplied. The exported actor also latches it.
        cfg['learning_control_hz'] = metadata['config'].get('learning_control_hz', 10)
        cfg['rl_seconds'] = metadata['config']['rl_seconds']
        if cfg.get('lineups') and side != original_team:
            cfg['lineups'] = {'red': cfg['lineups']['blue'], 'blue': cfg['lineups']['red']}
        if cfg.get('lineups'):
            from rmuc2027_core.models import normalize_team_lineup
            trained = normalize_team_lineup(load_submission(directory)[0].get('team_lineup'), '2027')
            if cfg['lineups'][side] != trained:
                raise ValueError('The RL side lineup must match its checkpoint; use the checkpoint configuration')
    # The viewer is a development match and can compare registered asymmetric
    # lineups. Without explicit lineups, honor each chosen submission's roster.
    if not cfg.get('lineups'):
        from rmuc2027_core.models import normalize_team_lineup
        cfg['lineups'] = {team: normalize_team_lineup(load_submission(path)[0].get('team_lineup'), '2027')
                         for team, path in sources.items()}
    cfg['team'] = side
    cfg['sim_backend'] = 'mjwarp'
    cfg['sim_device'] = args.device
    learned_side = side if metadata else None
    for team, path in sources.items():
        info = path/'rl-metadata.json'
        slots = json.loads(info.read_text()).get('learned_slots', []) if info.exists() else []
        if slots:
            if slots != [7] or (learned_side is not None and learned_side != team):
                raise ValueError('This viewer mode supports only one RL sentry on one team')
            learned_side = team
    cfg['team'] = learned_side or side
    if getattr(args,'scenario_id',None):
        if cfg.get('policy_version')!=4:raise ValueError('--scenario-id requires a v4 checkpoint')
        from ..scenarios import scenario_registry
        case=next((case for case in scenario_registry() if case.scenario_id==args.scenario_id),None)
        if case is None:raise ValueError('Unknown scenario id; use the v4 scenario registry')
        if case.learner_side!=cfg['team']:raise ValueError('Scenario side must match --rl-team')
        cfg['lineups']=case.lineups();cfg['scenario']=case.to_dict()
    return cfg, sources, learned_side


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='Defaults to checkpoint configuration with --rl-model, otherwise sentry-mjwarp-hour.json')
    parser.add_argument('--red', help='Red scripted strategy submission directory')
    parser.add_argument('--blue', help='Blue scripted strategy submission directory')
    parser.add_argument('--rl-model', help='Frozen sentry PPO checkpoint (.zip); no training occurs')
    parser.add_argument('--rl-team', choices=('red', 'blue'), help='RL sentry side; defaults to training/config side')
    parser.add_argument('--scenario-id',help='One registered v4 hardware scenario, including learner colour')
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
    with ExitStack() as stack:
        try:
            cfg, sources, rl_team = prepare_match(args, stack)
        except (ValueError, OSError, KeyError) as exc:
            parser.error(str(exc))
        description = (f'{rl_team}:7 frozen RL; auxiliary pool={list(cfg.get("auxiliary_pool",{}))}'
                       if rl_team else 'scripted AI; no RL model loaded')
        print(f'Native MJWarp viewer: {description}; space pauses, mouse rotates/zooms.', flush=True)
        print(f"RED={sources['red']} BLUE={sources['blue']}", flush=True)
        run_view(args, cfg, sources, rl_team)


def run_view(args, cfg, sources, rl_team=None):
    from ...strategies.battle import Battle
    from rmuc2027_core.engine import DT
    view = NativeView(labels=not args.no_labels)
    view.rl_team = rl_team
    overrides={};capabilities={}
    if cfg.get('auxiliary_pool'):
        from ..aux_policy import load_portable
        from ..privileged_aux import AuxController
        from ..session import resource
        for side in ('red','blue'):
            for slot in (1,3,4):
                role='heavy' if slot==1 else 'infantry'
                if role in cfg['auxiliary_pool']:
                    overrides[(side,slot)]=AuxController(dict(cfg,team=side,slot=slot),load_portable(resource(cfg['auxiliary_pool'][role])))
    if cfg.get('capability_sampling'):
        from ..capabilities import CapabilitySampler
        capabilities=CapabilitySampler(args.seed).sample(cfg['lineups'],cfg['team'])
    classifier=json.loads(Path(cfg['radar_classifier_path']).read_text()) if cfg.get('radar_classifier_path') else None
    with Battle(sources['red'], sources['blue'], args.seed, args.duration, cfg['map'], sandbox='trusted-local', side_lineups=cfg.get('lineups'),
                timeout=cfg['policy_timeout_s'], record_replay=False,
                navigation_mode=cfg['navigation_mode'], battle_settings={'drones_enabled': cfg['drones_enabled'], 'timed_assembly': cfg['timed_assembly']},
                fixed_policy_hz=cfg['fixed_policy_hz'], learning_control_hz=cfg['learning_control_hz'],
                physics_backend='mjwarp', sim_device=args.device,observation_profile=cfg.get('observation_profile'),
                controller_overrides=overrides,training_capabilities=capabilities,radar_classifier=classifier) as battle:
        if rl_team:
            battle.metadata['frozen_rl'] = {'team': rl_team, 'slot': 7,
                'submission': str(sources[rl_team]),
                'model': str(Path(args.rl_model).resolve()) if args.rl_model else None}
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
