"""SB3 vector adapter: CPU policies concurrently feed batched GPU workers."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import multiprocessing as mp
import os
import signal
import traceback
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv
from . import batching


class WarpVecEnv(DummyVecEnv):
    def __init__(self, factories):
        if batching.active_service is not None:
            raise RuntimeError('Only one kinematic vector environment may be active per process')
        self.physics = batching.PhysicsService(len(factories))
        self.pool = ThreadPoolExecutor(max_workers=len(factories))
        batching.active_service = self.physics
        try:
            super().__init__(factories)
        except BaseException:
            self.pool.shutdown(wait=True)
            self.physics.close()
            batching.active_service = None
            raise

    def _step_one(self,i):
        try:
            return self.envs[i].step(self.actions[i])
        finally:
            self.physics.worker_done()

    def step_wait(self):
        self.physics.begin_round(len(self.envs))
        futures = [self.pool.submit(self._step_one,i) for i in range(len(self.envs))]
        results = [f.result() for f in futures]
        resets={i:self.pool.submit(self.envs[i].reset)
                for i,(_,_,terminated,truncated,_) in enumerate(results) if terminated or truncated}
        for i, (obs, reward, terminated, truncated, info) in enumerate(results):
            self.buf_rews[i] = reward
            self.buf_dones[i] = terminated or truncated
            self.buf_infos[i] = info
            self.buf_infos[i]['TimeLimit.truncated'] = truncated and not terminated
            if self.buf_dones[i]:
                self.buf_infos[i]['terminal_observation'] = obs
                obs, self.reset_infos[i] = resets[i].result()
            self._save_obs(i, obs)
        return self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos)

    def reset(self):
        futures=[self.pool.submit(env.reset,seed=self._seeds[i],options=self._options[i])
                 for i,env in enumerate(self.envs)]
        for i,future in enumerate(futures):
            obs,self.reset_infos[i]=future.result()
            self._save_obs(i,obs)
        self._reset_seeds();self._reset_options()
        return self._obs_from_buf()

    def close(self):
        try:
            self.pool.shutdown(wait=True)
            super().close()
        finally:
            self.physics.close()
            batching.active_service = None


def _shard_worker(conn, factories):
    # Parent handles SIGINT and sends an explicit close command so checkpoints
    # and subprocess cleanup run deterministically.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    vec = None
    try:
        vec = WarpVecEnv(factories)
        conn.send(('ready', (vec.observation_space, vec.action_space)))
        while True:
            try:
                cmd, data = conn.recv()
            except EOFError:
                break
            if cmd == 'step':
                conn.send(('ok', vec.step(data)))
            elif cmd == 'reset':
                obs = vec.reset()
                conn.send(('ok', (obs, list(vec.reset_infos))))
            elif cmd == 'seed':
                conn.send(('ok', vec.seed(data)))
            elif cmd == 'get_attr':
                attr_name, indices = data
                conn.send(('ok', vec.get_attr(attr_name, indices)))
            elif cmd == 'set_attr':
                attr_name, value, indices = data
                vec.set_attr(attr_name, value, indices)
                conn.send(('ok', None))
            elif cmd == 'env_method':
                method_name, indices, args, kwargs = data
                conn.send(('ok', vec.env_method(method_name, *args, indices=indices, **kwargs)))
            elif cmd == 'env_is_wrapped':
                wrapper_class, indices = data
                conn.send(('ok', vec.env_is_wrapped(wrapper_class, indices)))
            elif cmd == 'physics_metrics':
                service = vec.physics
                conn.send(('ok', {
                    'batches': service.batches,
                    'world_steps': service.world_steps,
                    'max_batch': service.max_batch,
                    'integration_seconds': service.integration_seconds,
                    'batch_histogram': dict(service.batch_histogram),
                    'phase_seconds': dict(service.phase_seconds),
                }))
            elif cmd == 'reset_physics_metrics':
                vec.physics.reset_metrics()
                conn.send(('ok', None))
            elif cmd == 'process_ids':
                conn.send(('ok', [pid for env in vec.envs for pid in env.process_ids()]))
            elif cmd == 'warmup_prep':
                conn.send(('ok', [getattr(env.session.controller.navigator, 'preparation_seconds', 0.)
                                  for env in vec.envs]))
            elif cmd == 'close':
                break
            else:
                raise ValueError(f'Unknown shard command: {cmd}')
    except SystemExit:
        pass
    except BaseException:
        try:
            conn.send(('error', traceback.format_exc()))
        except OSError:
            pass
    finally:
        if vec is not None:
            try:
                vec.close()
            except BaseException:
                pass
        try:
            conn.close()
        except OSError:
            pass


class _ShardedPhysicsProxy:
    """Aggregates PhysicsService counters across shard processes for benchmarking."""

    def __init__(self, owner):
        self._owner = owner

    def _snapshot(self):
        parts = self._owner._broadcast('physics_metrics', None)
        batches = sum(p['batches'] for p in parts)
        world_steps = sum(p['world_steps'] for p in parts)
        max_batch = max((p['max_batch'] for p in parts), default=0)
        integration_seconds = sum(p['integration_seconds'] for p in parts)
        histogram = defaultdict(int)
        phases = defaultdict(float)
        for p in parts:
            for size, count in p['batch_histogram'].items():
                histogram[size] += count
            for name, seconds in p['phase_seconds'].items():
                phases[name] += seconds
        return {
            'batches': batches,
            'world_steps': world_steps,
            'max_batch': max_batch,
            'integration_seconds': integration_seconds,
            'batch_histogram': dict(histogram),
            'phase_seconds': dict(phases),
        }

    def reset_metrics(self):
        self._owner._broadcast('reset_physics_metrics', None)

    @property
    def batches(self):
        return self._snapshot()['batches']

    @property
    def world_steps(self):
        return self._snapshot()['world_steps']

    @property
    def max_batch(self):
        return self._snapshot()['max_batch']

    @property
    def integration_seconds(self):
        return self._snapshot()['integration_seconds']

    @property
    def batch_histogram(self):
        return self._snapshot()['batch_histogram']

    @property
    def phase_seconds(self):
        return self._snapshot()['phase_seconds']


class ShardedWarpVecEnv(VecEnv):
    """Spreads WarpVecEnv groups across multiple spawned Python processes to bypass the GIL."""

    def __init__(self, factories, shards):
        count = len(factories)
        if shards < 1:
            raise ValueError('sim-shards must be positive')
        shards = min(shards, count)
        self.num_shards = shards
        self.closed = False
        self.waiting = False
        self.remotes = []
        self.processes = []
        self.slices = []
        base, rem = divmod(count, shards)
        ctx = mp.get_context('spawn')
        offset = 0
        try:
            for shard_idx in range(shards):
                size = base + (1 if shard_idx < rem else 0)
                shard_factories = factories[offset:offset + size]
                self.slices.append((offset, offset + size))
                offset += size
                parent_conn, child_conn = ctx.Pipe()
                proc = ctx.Process(target=_shard_worker, args=(child_conn, shard_factories),
                                   name=f'warp-shard-{shard_idx}', daemon=False)
                proc.start()
                child_conn.close()
                self.remotes.append(parent_conn)
                self.processes.append(proc)
            spaces = []
            for remote in self.remotes:
                status, payload = remote.recv()
                if status == 'error':
                    raise RuntimeError(f'Warp shard initialization failed:\n{payload}')
                if status != 'ready':
                    raise RuntimeError(f'Unexpected shard handshake: {status}')
                spaces.append(payload)
            obs_space, act_space = spaces[0]
            super().__init__(count, obs_space, act_space)
            self.physics = _ShardedPhysicsProxy(self)
        except BaseException:
            self.close()
            raise

    def _recv_one(self, remote):
        try:
            status, payload = remote.recv()
        except EOFError as exc:
            raise RuntimeError('Warp shard process exited unexpectedly') from exc
        if status == 'error':
            raise RuntimeError(f'Warp shard error:\n{payload}')
        return payload

    def _broadcast(self, cmd, data):
        for remote in self.remotes:
            remote.send((cmd, data))
        return [self._recv_one(remote) for remote in self.remotes]

    def seed(self, seed=None):
        seeds = []
        for remote, (start, _end) in zip(self.remotes, self.slices):
            remote.send(('seed', None if seed is None else seed + start))
        for remote in self.remotes:
            seeds.extend(self._recv_one(remote))
        return seeds

    def reset(self):
        results = self._broadcast('reset', None)
        obs = np.concatenate([item[0] for item in results], axis=0)
        self.reset_infos = [info for item in results for info in item[1]]
        return obs

    def step_async(self, actions):
        actions = np.asarray(actions)
        for remote, (start, end) in zip(self.remotes, self.slices):
            remote.send(('step', actions[start:end]))
        self.waiting = True

    def step_wait(self):
        results = [self._recv_one(remote) for remote in self.remotes]
        self.waiting = False
        obs = np.concatenate([item[0] for item in results], axis=0)
        rews = np.concatenate([item[1] for item in results], axis=0)
        dones = np.concatenate([item[2] for item in results], axis=0)
        infos = [info for item in results for info in item[3]]
        return obs, rews, dones, infos

    def _group_indices(self, indices):
        target = list(self._get_indices(indices))
        grouped = defaultdict(list)
        positions = defaultdict(list)
        for pos, global_idx in enumerate(target):
            for shard_idx, (start, end) in enumerate(self.slices):
                if start <= global_idx < end:
                    grouped[shard_idx].append(global_idx - start)
                    positions[shard_idx].append(pos)
                    break
        return len(target), grouped, positions

    def get_attr(self, attr_name, indices=None):
        total, grouped, positions = self._group_indices(indices)
        out = [None] * total
        for shard_idx, local_indices in grouped.items():
            self.remotes[shard_idx].send(('get_attr', (attr_name, local_indices)))
        for shard_idx in grouped:
            values = self._recv_one(self.remotes[shard_idx])
            for pos, value in zip(positions[shard_idx], values):
                out[pos] = value
        return out

    def set_attr(self, attr_name, value, indices=None):
        _, grouped, _ = self._group_indices(indices)
        for shard_idx, local_indices in grouped.items():
            self.remotes[shard_idx].send(('set_attr', (attr_name, value, local_indices)))
        for shard_idx in grouped:
            self._recv_one(self.remotes[shard_idx])

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        total, grouped, positions = self._group_indices(indices)
        out = [None] * total
        for shard_idx, local_indices in grouped.items():
            self.remotes[shard_idx].send(('env_method', (method_name, local_indices, method_args, method_kwargs)))
        for shard_idx in grouped:
            values = self._recv_one(self.remotes[shard_idx])
            for pos, value in zip(positions[shard_idx], values):
                out[pos] = value
        return out

    def env_is_wrapped(self, wrapper_class, indices=None):
        total, grouped, positions = self._group_indices(indices)
        out = [False] * total
        for shard_idx, local_indices in grouped.items():
            self.remotes[shard_idx].send(('env_is_wrapped', (wrapper_class, local_indices)))
        for shard_idx in grouped:
            values = self._recv_one(self.remotes[shard_idx])
            for pos, value in zip(positions[shard_idx], values):
                out[pos] = value
        return out

    def get_images(self):
        return [None] * self.num_envs

    def process_ids(self):
        """Return all descendant simulator and policy PIDs across shards."""
        return [pid for group in self._broadcast('process_ids', None) for pid in group]

    def warmup_preparation_seconds(self):
        return [value for group in self._broadcast('warmup_prep', None) for value in group]

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                try:
                    if remote.poll(0.5):
                        remote.recv()
                except (EOFError, OSError):
                    pass
            self.waiting = False
        for remote in self.remotes:
            try:
                remote.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for proc in self.processes:
            proc.join(timeout=15)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
        for remote in self.remotes:
            try:
                remote.close()
            except OSError:
                pass
