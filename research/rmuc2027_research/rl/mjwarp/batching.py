"""Microbatch CPU rule workers into one device integrator, with no barriers."""
from collections import defaultdict, OrderedDict
from concurrent.futures import Future
import queue
import threading
import time

active_service = None


class PhysicsService:
    def __init__(self, max_worlds):
        self.max_worlds = max_worlds
        self.queue = queue.Queue()
        self.engines = OrderedDict()
        self.closed = False
        self.batches = self.world_steps = self.max_batch = 0
        self.integration_seconds = 0.
        self.phase_seconds = defaultdict(float)
        self.batch_histogram = defaultdict(int)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, device, request):
        if self.closed:
            raise RuntimeError('Physics batch service closed')
        future = Future()
        self.queue.put((device, request, future))
        return future.result(timeout=300)

    def run(self):
        from .arena import ArenaPhysics
        while True:
            first = self.queue.get()
            if first is None:
                break
            batch = [first]
            until = time.monotonic()+.002
            while len(batch) < self.max_worlds:
                try:
                    item = self.queue.get(timeout=max(0, until-time.monotonic()))
                except queue.Empty:
                    break
                if item is None:
                    self.queue.put(None)
                    break
                batch.append(item)
            groups = defaultdict(list)
            for item in batch:
                groups[(item[0], ArenaPhysics.signature(item[1]))].append(item)
            for (device, signature), items in groups.items():
                key = device, signature, len(items)
                try:
                    if key not in self.engines:
                        # Bound GPU memory if covers or model topologies change.
                        if len(self.engines) >= 4:
                            _, old = self.engines.popitem(last=False)
                            old.close()
                        self.engines[key] = ArenaPhysics(device)
                    self.engines.move_to_end(key)
                    started = time.perf_counter()
                    results = self.engines[key].step_many([i[1] for i in items])
                    self.integration_seconds += time.perf_counter()-started
                    for phase, seconds in self.engines[key].last_profile.items():
                        self.phase_seconds[phase] += seconds
                    self.batch_histogram[len(items)] += 1
                    self.batches += 1
                    self.world_steps += len(items)
                    self.max_batch = max(self.max_batch, len(items))
                    for item, result in zip(items, results):
                        item[2].set_result(result)
                except BaseException as exc:
                    for item in items:
                        item[2].set_exception(exc)
        for engine in self.engines.values():
            engine.close()

    def reset_metrics(self):
        """Call only at a completed vector-step boundary, with workers idle."""
        self.batches = self.world_steps = self.max_batch = 0
        self.integration_seconds = 0.
        self.phase_seconds.clear()
        self.batch_histogram.clear()

    def close(self):
        self.closed = True
        self.queue.put(None)
        self.thread.join(timeout=300)
        if self.thread.is_alive():
            raise RuntimeError('Physics service shutdown timed out')
