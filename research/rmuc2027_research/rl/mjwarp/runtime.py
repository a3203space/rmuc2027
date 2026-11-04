"""MJWarp device ownership and batched world lifecycle, independent of PPO."""
import hashlib
import importlib.metadata
import math
import time


def dependencies():
    try:
        import mujoco
        import mujoco_warp
        import numpy
        import warp
    except ImportError as exc:
        raise RuntimeError('MJWarp runtime unavailable; install research/requirements-mjwarp.txt in the training environment') from exc
    return mujoco, mujoco_warp, numpy, warp


class BatchPhysics:
    """Own one MJCF topology and N independent worlds on an explicit device.

    Controls and state are batch arrays. No automatic CPU fallback is permitted.
    Callers own rule state and must synchronize it at each physics substep.
    """
    def __init__(self, xml, num_worlds, device='cuda:0', timestep=1/120, nconmax=None, njmax=None):
        if isinstance(num_worlds, bool) or not isinstance(num_worlds, int) or num_worlds < 1:
            raise ValueError('num_worlds must be a positive integer')
        if not math.isfinite(timestep) or timestep <= 0:
            raise ValueError('timestep must be finite and positive')
        if device != 'cpu' and not device.startswith('cuda:'):
            raise ValueError('Select cpu for debugging or an explicit cuda:N device')
        mj, mjw, np, wp = dependencies()
        wp.init()
        try:
            selected = wp.get_device(device)
        except Exception as exc:
            raise RuntimeError(f'MJWarp device unavailable: {device}; check NVIDIA device access and driver') from exc
        if device.startswith('cuda:') and not selected.is_cuda:
            raise RuntimeError('CUDA requested but no CUDA device selected')
        self.mj, self.mjw, self.np, self.wp = mj, mjw, np, wp
        self.device = selected
        self.num_worlds = num_worlds
        self.cpu_model = mj.MjModel.from_xml_string(xml)
        self.cpu_model.opt.timestep = timestep
        self.xml_hash = hashlib.sha256(xml.encode()).hexdigest()
        self.timestep = timestep
        self.closed = False
        self.graph = None
        self.steps = 0
        self.step_seconds = 0.
        with wp.ScopedDevice(selected):
            self.model = mjw.put_model(self.cpu_model)
            self.data = mjw.make_data(self.cpu_model, nworld=num_worlds, nconmax=nconmax, njmax=njmax)
        self.reset(range(num_worlds))

    def _open(self):
        if self.closed:
            raise RuntimeError('BatchPhysics is closed')

    def _assign(self, name, values):
        target = getattr(self.data, name)
        target.assign(self.np.asarray(values, dtype=self.np.float32))

    def reset(self, indices):
        """Reset selected worlds only; no other world's integration state changes."""
        self._open()
        ids = list(indices)
        if len(set(ids)) != len(ids) or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < self.num_worlds for i in ids):
            raise ValueError('Reset indices must be unique valid world indices')
        if not ids:
            return
        mask = self.np.zeros(self.num_worlds, dtype=self.np.bool_)
        mask[ids] = True
        with self.wp.ScopedDevice(self.device):
            reset = self.wp.array(mask, dtype=self.wp.bool, device=self.device)
            self.mjw.reset_data(self.model, self.data, reset)
            self.mjw.forward(self.model, self.data)

    def step(self, controls):
        self._open()
        values = self.np.asarray(controls)
        expected = (self.num_worlds, self.cpu_model.nu)
        if values.shape != expected or not self.np.isfinite(values).all():
            raise ValueError(f'Controls must have finite values and shape {expected}')
        started = time.perf_counter()
        self._assign('ctrl', values)
        with self.wp.ScopedDevice(self.device):
            if self.device.is_cuda:
                if self.graph is None:
                    with self.wp.ScopedCapture(device=self.device) as capture:
                        self.mjw.step(self.model, self.data)
                    self.graph = capture.graph
                self.wp.capture_launch(self.graph)
            else:
                self.mjw.step(self.model, self.data)
        self.wp.synchronize_device(self.device)
        if int(self.data.nacon.numpy()[0]) > self.data.naconmax or bool((self.data.nefc.numpy() > self.data.njmax).any()):
            raise RuntimeError('MJWarp contact/constraint capacity exceeded; increase nconmax/njmax')
        self.steps += 1
        self.step_seconds += time.perf_counter()-started

    def state(self):
        self._open()
        return {name: getattr(self.data, name).numpy().copy() for name in ('qpos', 'qvel', 'time')}

    def metadata(self):
        return {'backend': 'mjwarp', 'device': str(self.device), 'worlds': self.num_worlds,
                'model_sha256': self.xml_hash, 'timestep': self.timestep,
                'steps': self.steps, 'step_seconds': self.step_seconds,
                'dependencies': {name: importlib.metadata.version(name) for name in ('mujoco', 'mujoco-warp', 'warp-lang')}}

    def close(self):
        if not self.closed:
            self.wp.synchronize_device(self.device)
            self.graph = None
            self.data = self.model = None
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
