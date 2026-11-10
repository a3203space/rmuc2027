"""Owned pageable host buffers for repeated synchronous-frame uploads."""
import numpy as np
import warp as wp


class HostUploads:
    """Reuse CPU array metadata; callers finish frame readback before reuse.

    Each destination has independent owned storage. Holding the destination
    prevents id reuse, and close synchronizes before releasing host storage.
    """
    def __init__(self):
        self.entries = {}
        self.closed = False

    def assign(self, target, values):
        if self.closed:
            raise RuntimeError('Host upload buffers are closed')
        if not isinstance(values, np.ndarray):
            raise TypeError('Host upload source must be a NumPy array')
        entry = self.entries.get(id(target))
        if entry is None or entry[2].shape != values.shape:
            host = wp.array(np.array(values, copy=True, order='C'), dtype=target.dtype,
                            device='cpu', copy=False)
            view = np.asarray(host)
            if view.shape != values.shape:
                # Preserve Warp's alternate vector layouts and shape errors.
                target.assign(values)
                return
            entry = (target, host, view)
            self.entries[id(target)] = entry
        else:
            np.copyto(entry[2], values, casting='unsafe')
        wp.copy(target, entry[1])

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            devices = {str(target.device): target.device for target, _, _ in self.entries.values()
                       if target.device.is_cuda}
            for device in devices.values():
                wp.synchronize_device(device)
        finally:
            self.entries.clear()
