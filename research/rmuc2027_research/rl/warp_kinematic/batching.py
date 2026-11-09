"""Asynchronous world batches with per-client traversal state and one CUDA context.

Only submitted worlds advance. State follows the client across batch sizes,
worker skew and episode resets; it is never owned by a cache slot.
"""
from collections import OrderedDict, defaultdict
from concurrent.futures import Future
import json
import queue
import threading
import time
import numpy as np
import warp as wp
from .arena import ArenaMotion
from .protocol import DT, MotionFrame

active_service = None


@wp.kernel
def restore_state(state: wp.array2d(dtype=float),starts: wp.array(dtype=wp.vec3),ends: wp.array(dtype=wp.vec3),
                  elapsed: wp.array(dtype=float),durations: wp.array(dtype=float),cooldown: wp.array(dtype=float),
                  status: wp.array(dtype=wp.int32),fall: wp.array(dtype=float),altitude: wp.array(dtype=float),
                  offsets: wp.array(dtype=wp.vec2),base: wp.array(dtype=float),flight: wp.array(dtype=wp.int32),
                  leg_command: wp.array(dtype=float)):
    i=wp.tid()
    starts[i]=wp.vec3(state[i,0],state[i,1],state[i,2]);ends[i]=wp.vec3(state[i,3],state[i,4],state[i,5])
    elapsed[i]=state[i,6];durations[i]=state[i,7];cooldown[i]=state[i,8];status[i]=int(state[i,9])
    fall[i]=state[i,10];altitude[i]=state[i,11];offsets[i]=wp.vec2(state[i,12],state[i,13])
    base[i]=state[i,14];flight[i]=int(state[i,15])
    leg_command[i]=state[i,16]


class BatchClient:
    protocol='kinematic_frame'

    def __init__(self,service,device):
        self.service=service;self.device=device;self.closed=False;self.next_sequence=0
        self.ids=None;self.state=None;self.expected_positions=None
        self.arena=None;self.designs=None
        self.scene_signature=None;self.design_signature=None

    def step(self,request):
        if self.closed:raise RuntimeError('Adapter closed')
        if (request.get('op')!='kinematic_frame' or request.get('version')!=1 or
            request.get('sequence')!=self.next_sequence or request.get('substeps')!=6 or abs(request.get('dt',0)-DT)>1e-12):
            raise ValueError('Invalid motion request identity')
        ids=tuple(r['id'] for r in request['robots'])
        if self.ids is None:
            self.ids=ids
            self.arena=request['arena']
            self.designs=[r['design'] for r in request['robots']]
            self.scene_signature=json.dumps(request['arena'],sort_keys=True,separators=(',',':'))
            self.design_signature=json.dumps([(r['design'],r['radius'],r['role']) for r in request['robots']],sort_keys=True,separators=(',',':'))
        if ids!=self.ids:raise ValueError('Robot order changed')
        response=self.service.submit(self,request)
        self.next_sequence+=1
        return response

    def close(self):
        self.closed=True;self.state=None;self.expected_positions=None;self.arena=None;self.designs=None


def make_adapter(device):
    return ArenaMotion(device) if active_service is None else BatchClient(active_service,device)


class WorldBatch:
    def __init__(self,device):
        self.engine=ArenaMotion(device,carry_state=True)
        self.upload=None;self.restore_command=None

    def step_many(self,items):
        engine=self.engine;robots=[];slices=[]
        for world,(client,request) in enumerate(items):
            start=len(robots)
            robots.extend(dict(r,id=f'{world}:{r["id"]}',_world=world,
                design=client.designs[index] if client.designs is not None else r['design'])
                for index,r in enumerate(request['robots']))
            slices.append(slice(start,len(robots)))
        combined=dict(items[0][1],robots=robots,sequence=engine.next_sequence,
                      arena=items[0][0].arena or items[0][1]['arena'])
        if engine.motion is None:
            engine.initialize(combined)
            self.upload=wp.empty((len(robots),17),dtype=float,device=engine.device)
            self.restore_command=wp.launch(restore_state,len(robots),inputs=[self.upload,engine.jump_start,
                engine.jump_end,engine.jump_elapsed,engine.jump_duration,engine.jump_cooldown,engine.jump_status,
                engine.fall_speed,engine.motion.flight_z,engine.motion.height_offsets,engine.base,engine.motion.flight,engine.leg_command],
                device=engine.device,record_cmd=True)
        states=[]
        for (client,request),span in zip(items,slices):
            bodies=request['robots']
            designs=client.designs if client.designs is not None else [r['design'] for r in bodies]
            if client.state is None:
                floor=engine.terrain.heights([r['pos'][:2] for r in bodies])
                base=np.asarray([r['pos'][2] for r in bodies])-floor
                state=np.zeros((len(bodies),17),dtype=np.float32)
                state[:,12]=base
                state[:,13]=base+np.asarray([d.get('muzzle_z_m',.4)+.06 for d in designs])
                state[:,14]=base
                state[:,16]=[d.get('leg_nominal_m',0.) for d in designs]
            else:
                state=client.state.copy()
                authoritative=np.asarray([r['pos'] for r in request['robots']])
                changed=np.linalg.norm(authoritative-client.expected_positions,axis=1)>.05
                state[np.ix_(changed,[6,8,9,10,15])]=0.
                reset_pose=changed & np.asarray([r.get('grounded',True) for r in bodies],dtype=bool)
                if reset_pose.any():
                    bodies=request['robots']
                    floor=engine.terrain.heights([r['pos'][:2] for r in bodies])
                    base=authoritative[:,2]-floor
                    heights=state[:,13]-state[:,12]
                    state[reset_pose,12]=base[reset_pose];state[reset_pose,13]=(base+heights)[reset_pose]
                    state[reset_pose,14]=base[reset_pose]
                    state[reset_pose,16]=np.asarray([d.get('leg_nominal_m',0.) for d in designs])[reset_pose]
            states.append(state)
        self.upload.assign(np.concatenate(states))
        self.restore_command.launch()
        # Previous use of this cache slot may have contained different clients.
        # Per-client authority correction was applied above instead.
        engine.expected_positions=None
        response=engine.step(combined)
        positions=np.asarray(response['positions']);velocities=np.asarray(response['velocities'])
        headings=np.asarray(response['headings']);folded=np.asarray(response['folded']);grounded=np.asarray(response['grounded'])
        results=[]
        for (client,request),span in zip(items,slices):
            frame=MotionFrame(client.ids,positions[:,span],velocities[:,span],headings[:,span],folded[:,span],grounded[:,span])
            client.state=engine.last_state[span].copy()
            client.expected_positions=frame.positions[-1].copy()
            results.append(frame.response(request['sequence']))
        return results

    def close(self):
        self.restore_command=None;self.upload=None;self.engine.close()


class PhysicsService:
    def __init__(self,max_worlds,wait_seconds=.0005):
        self.max_worlds=max_worlds;self.wait_seconds=wait_seconds
        self.queue=queue.Queue();self.closed=False;self.engines=OrderedDict()
        self.lock=threading.Lock()
        self.round_workers=None
        self.reset_metrics()
        self.thread=threading.Thread(target=self.run,name='kinematic-cuda-batches',daemon=True)
        self.thread.start()

    def submit(self,client,request):
        future=Future()
        with self.lock:
            if self.closed:raise RuntimeError('Physics batch service closed')
            self.queue.put((client,request,future))
        return future.result(timeout=300)

    def begin_round(self,count):
        self.queue.put(('round',count))

    def worker_done(self):
        self.queue.put(('done',0))

    def _control(self,item):
        if isinstance(item[0],str):
            if item[0]=='round':self.round_workers=item[1]
            else:
                self.round_workers-=1
                if self.round_workers==0:self.round_workers=None
            return True
        return False

    def run(self):
        while True:
            first=self.queue.get()
            if first is None:break
            if self._control(first):continue
            batch=[first];until=time.monotonic()+self.wait_seconds
            # During a vector step every unfinished worker either submits its
            # next frame or explicitly reports completion. Wait for that set,
            # rather than timing out before a slower policy has reached CUDA.
            # Resets/direct clients still use bounded opportunistic batching.
            while len(batch)<min(self.max_worlds,self.round_workers or self.max_worlds):
                try:item=self.queue.get(timeout=None if self.round_workers is not None else max(0.,until-time.monotonic()))
                except queue.Empty:break
                if item is None:
                    self.queue.put(None);break
                if self._control(item):continue
                batch.append(item)
            groups=defaultdict(list)
            for item in batch:
                client=item[0];groups[(client.device,client.scene_signature)].append(item)
            for (device,scene),items in groups.items():
                key=(device,scene,tuple(client.design_signature for client,_,_ in items))
                try:
                    if key not in self.engines:
                        if len(self.engines)>=4:
                            _,old=self.engines.popitem(last=False);old.close()
                        self.engines[key]=WorldBatch(device)
                    self.engines.move_to_end(key)
                    started=time.perf_counter()
                    results=self.engines[key].step_many([(c,r) for c,r,_ in items])
                    self.integration_seconds+=time.perf_counter()-started
                    self.batches+=1;self.world_steps+=len(items);self.max_batch=max(self.max_batch,len(items))
                    self.batch_histogram[len(items)]+=1
                    for item,result in zip(items,results):item[2].set_result(result)
                except BaseException as exc:
                    # A failed integration cannot leave partial state in a slot
                    # which another set of worlds may later reuse.
                    failed=self.engines.pop(key,None)
                    if failed is not None:failed.close()
                    for item in items:item[2].set_exception(exc)
        for engine in self.engines.values():engine.close()
        self.engines.clear()

    def reset_metrics(self):
        self.batches=self.world_steps=self.max_batch=0;self.integration_seconds=0.
        self.batch_histogram=defaultdict(int);self.phase_seconds={}

    def close(self):
        with self.lock:
            if self.closed:return
            self.closed=True;self.queue.put(None)
        self.thread.join(timeout=300)
        if self.thread.is_alive():raise RuntimeError('Physics service shutdown timed out')
