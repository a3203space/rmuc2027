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
        self.lock = threading.Lock()
        self.cleanup_errors = []; self.cleanup_error_count = 0; self.terminal_error = None
        self.batches = self.world_steps = self.max_batch = 0
        self.integration_seconds = 0.
        self.phase_seconds = defaultdict(float)
        self.batch_histogram = defaultdict(int)
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def submit(self, device, request):
        future = Future()
        with self.lock:
            if self.closed:
                raise RuntimeError('Physics batch service closed')
            self.queue.put((device, request, future))
        return future.result(timeout=300)

    @staticmethod
    def _fail(items, error):
        for item in items:
            if item is not None and not item[2].done():
                item[2].set_exception(error)

    def _dispose(self, engine):
        try: engine.close()
        except BaseException as error:
            self.cleanup_error_count += 1
            self.cleanup_errors.append(f'{type(error).__name__}: {error}')
            del self.cleanup_errors[:-32]

    def run(self):
        batch=[]
        try:
            from .arena import ArenaPhysics
            while True:
                first = self.queue.get()
                if first is None: break
                batch = [first]
                until = time.monotonic()+.002
                while len(batch) < self.max_worlds:
                    try: item = self.queue.get(timeout=max(0, until-time.monotonic()))
                    except queue.Empty: break
                    if item is None:
                        self.queue.put(None); break
                    batch.append(item)
                groups = defaultdict(list)
                for item in batch:
                    try: groups[(item[0], ArenaPhysics.signature(item[1]))].append(item)
                    except BaseException as error: self._fail([item], error)
                for (device, signature), items in groups.items():
                    key = device, signature, len(items)
                    try:
                        if key not in self.engines:
                            if len(self.engines) >= 4:
                                _, old = self.engines.popitem(last=False)
                                self._dispose(old)
                            self.engines[key] = ArenaPhysics(device)
                        self.engines.move_to_end(key)
                        started = time.perf_counter()
                        results = self.engines[key].step_many([i[1] for i in items])
                        if len(results)!=len(items): raise RuntimeError('Physics batch returned an incomplete result set')
                        self.integration_seconds += time.perf_counter()-started
                        for phase, seconds in self.engines[key].last_profile.items():
                            self.phase_seconds[phase] += seconds
                        self.batch_histogram[len(items)] += 1
                        self.batches += 1
                        self.world_steps += len(items)
                        self.max_batch = max(self.max_batch, len(items))
                        for item, result in zip(items, results):
                            if not item[2].done(): item[2].set_result(result)
                    except BaseException as error:
                        failed=self.engines.pop(key,None)
                        if failed is not None: self._dispose(failed)
                        self._fail(items,error)
                batch=[]
        except BaseException as error:
            self.terminal_error=error
        finally:
            with self.lock: self.closed=True
            error=self.terminal_error or RuntimeError('Physics batch service closed')
            self._fail(batch,error)
            while True:
                try: item=self.queue.get_nowait()
                except queue.Empty: break
                self._fail([item],error)
            for engine in self.engines.values(): self._dispose(engine)
            self.engines.clear()

    def reset_metrics(self):
        """Call only at a completed vector-step boundary, with workers idle."""
        self.batches = self.world_steps = self.max_batch = 0
        self.integration_seconds = 0.
        self.phase_seconds.clear()
        self.batch_histogram.clear()

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.queue.put(None)
        self.thread.join(timeout=300)
        if self.thread.is_alive():
            raise RuntimeError('Physics service shutdown timed out')
