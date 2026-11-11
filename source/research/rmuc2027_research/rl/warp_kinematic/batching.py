"""Asynchronous world batches with per-client traversal state and one CUDA context.

Only submitted worlds advance. State follows the client across batch sizes,
worker skew and episode resets; it is never owned by a cache slot.
"""
from collections import OrderedDict, defaultdict
from concurrent.futures import Future
from copy import deepcopy
import json
import queue
import threading
import time
import numpy as np
import warp as wp
from .arena import ArenaMotion, validate_request_identity
from .protocol import MotionFrame
from .uploads import HostUploads
from .pose import resting_origin, collision_heights

active_service = None

DRIVE_UPLOAD_FIELDS = frozenset(('speed_reference_mps', 'motor_force_max_n', 'drive_efficiency'))


def topology_signature(robots):
    """Only uploaded drive scalars may differ when reusing a topology."""
    return json.dumps([({k:v for k,v in r['design'].items() if k not in DRIVE_UPLOAD_FIELDS},
                        r['radius'],r['role']) for r in robots],sort_keys=True,separators=(',',':'))


@wp.kernel
def restore_state(state: wp.array2d(dtype=float),starts: wp.array(dtype=wp.vec3),ends: wp.array(dtype=wp.vec3),
                  elapsed: wp.array(dtype=float),durations: wp.array(dtype=float),cooldown: wp.array(dtype=float),
                  status: wp.array(dtype=wp.int32),fall: wp.array(dtype=float),altitude: wp.array(dtype=float),
                  offsets: wp.array(dtype=wp.vec2),base: wp.array(dtype=float),flight: wp.array(dtype=wp.int32),
                  leg_command: wp.array(dtype=float), latched: wp.array(dtype=float), strength: wp.array(dtype=float)):
    i=wp.tid()
    starts[i]=wp.vec3(state[i,0],state[i,1],state[i,2]);ends[i]=wp.vec3(state[i,3],state[i,4],state[i,5])
    elapsed[i]=state[i,6];durations[i]=state[i,7];cooldown[i]=state[i,8];status[i]=int(state[i,9])
    fall[i]=state[i,10];altitude[i]=state[i,11];offsets[i]=wp.vec2(state[i,12],state[i,13])
    base[i]=state[i,14];flight[i]=int(state[i,15])
    leg_command[i]=state[i,16]
    latched[i]=0.0
    if state.shape[1]>17:latched[i]=state[i,17]
    strength[i]=0.0
    if state.shape[1]>18:strength[i]=state[i,18]


class BatchClient:
    protocol='kinematic_frame'

    def __init__(self,service,device):
        self.service=service;self.device=device;self.closed=False;self.next_sequence=0
        self.ids=None;self.state=None;self.expected_positions=None
        self.arena=None;self.designs=None
        self.scene_signature=None;self.design_signature=None
        self.version=None;self.body_signature=None

    def step(self,request):
        return self.step_frame(request).response(request['sequence'])

    def step_encoded(self,request):
        return self.step_frame(request).encoded_response(request['sequence'])

    def step_frame(self,request):
        if self.closed:raise RuntimeError('Adapter closed')
        validate_request_identity(request,self.next_sequence)
        robots=request['robots']
        ids=tuple(r['id'] for r in robots)
        body_signature=tuple((r['radius'],r['role']) for r in robots)
        if self.ids is None:
            if request.get('arena') is None or any(r.get('design') is None for r in robots):
                raise ValueError('Initial motion request requires arena and robot designs')
            scene_signature=json.dumps([request['version'],request['arena']],sort_keys=True,separators=(',',':'))
            design_signature=topology_signature(robots)
            self.ids=ids
            self.version=request['version'];self.body_signature=body_signature
            self.arena=deepcopy(request['arena'])
            self.designs=[deepcopy(r['design']) for r in robots]
            self.scene_signature=scene_signature;self.design_signature=design_signature
        else:
            if request['version']!=self.version:raise ValueError('Motion protocol changed within world')
            if ids!=self.ids:raise ValueError('Robot order changed')
            if body_signature!=self.body_signature:raise ValueError('Motion topology changed within world')
            if request.get('arena') is not None:
                scene_signature=json.dumps([request['version'],request['arena']],sort_keys=True,separators=(',',':'))
                if scene_signature!=self.scene_signature:raise ValueError('Motion scene changed within world')
            # Native compact frames omit designs; avoid rehashing them in that
            # hot path. Full frames may refresh only the uploaded drive fields.
            if any(r.get('design') is not None for r in robots):
                full=[dict(r,design=r['design'] if r.get('design') is not None else d)
                      for r,d in zip(robots,self.designs)]
                if topology_signature(full)!=self.design_signature:raise ValueError('Motion topology changed within world')
                self.designs=[deepcopy(r['design']) if r.get('design') is not None else d
                              for r,d in zip(robots,self.designs)]
        for design,robot in zip(self.designs,robots):
            design.update({k:v for k,v in robot.get('drive_parameters',{}).items() if k in DRIVE_UPLOAD_FIELDS})
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
        self.host_uploads=HostUploads()

    def step_many(self,items,frames=False):
        engine=self.engine;robots=[];slices=[]
        for world,(client,request) in enumerate(items):
            start=len(robots)
            # Cache keys describe ordered topology, not external robot names.
            # Stable slot ids permit reuse by a new client or reordered worlds.
            robots.extend(dict(r,id=f'{world}:{index}',_world=world,
                design=client.designs[index] if client.designs is not None else r['design'])
                for index,r in enumerate(request['robots']))
            slices.append(slice(start,len(robots)))
        combined=dict(items[0][1],robots=robots,sequence=engine.next_sequence,
                      arena=items[0][0].arena or items[0][1]['arena'])
        if engine.motion is None:
            engine.initialize(combined)
            self.upload=wp.empty((len(robots),19 if engine.extended else 17),dtype=float,device=engine.device)
            self.restore_command=wp.launch(restore_state,len(robots),inputs=[self.upload,engine.jump_start,
                engine.jump_end,engine.jump_elapsed,engine.jump_duration,engine.jump_cooldown,engine.jump_status,
                engine.fall_speed,engine.motion.flight_z,engine.motion.height_offsets,engine.base,engine.motion.flight,engine.leg_command,engine.jump_latched,engine.jump_strength],
                device=engine.device,record_cmd=True)
        states=[]
        for (client,request),span in zip(items,slices):
            bodies=request['robots']
            designs=client.designs if client.designs is not None else [r['design'] for r in bodies]
            if client.state is None:
                floor=engine.terrain.heights([r['pos'][:2] for r in bodies])
                base=np.asarray([r['pos'][2] for r in bodies])-floor
                airborne=np.asarray([engine.extended and not r.get('grounded',True) and r['role']!='aerial' for r in bodies])
                base[airborne]=np.asarray([resting_origin(d) for d in designs])[airborne]
                state=np.zeros((len(bodies),19 if engine.extended else 17),dtype=np.float32)
                state[airborne,9]=4.;state[airborne,15]=1.
                if engine.extended:
                    state[:,10]=[r['velocity'][2] for r in bodies]
                    state[:,11]=[r['pos'][2] for r in bodies]
                state[:,12]=base
                state[:,13]=base+np.asarray([collision_heights(d)[0] for d in designs])
                state[:,14]=base
                state[:,16]=[d.get('leg_nominal_m',0.) for d in designs]
            else:
                state=client.state.copy()
                authoritative=np.asarray([r['pos'] for r in request['robots']])
                changed=np.linalg.norm(authoritative-client.expected_positions,axis=1)>.05
                state[np.ix_(changed,[6,8,9,10,15])]=0.
                if engine.extended:
                    state[np.ix_(changed,[17,18])]=0.
                    airborne=changed & np.asarray([r.get('enabled',True) and not r.get('grounded',True) and r['role']!='aerial'
                                                   for r in bodies],dtype=bool)
                    # Match direct-adapter authority restoration after reset.
                    state[airborne,9]=4.;state[airborne,15]=1.
                    state[airborne,10]=np.asarray([r['velocity'][2] for r in bodies])[airborne]
                    state[airborne,11]=authoritative[airborne,2]
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
        self.host_uploads.assign(self.upload,np.concatenate(states))
        self.restore_command.launch()
        # Previous use of this cache slot may have contained different clients.
        # Per-client authority correction was applied above instead.
        engine.expected_positions=None
        response=engine.step_frame(combined)
        results=[]
        for (client,request),span in zip(items,slices):
            frame=MotionFrame(client.ids,response.positions[:,span],response.velocities[:,span],response.headings[:,span],response.folded[:,span],response.grounded[:,span])
            if response.rotations is not None:
                frame=MotionFrame(frame.ids,frame.positions,frame.velocities,frame.headings,frame.folded,frame.grounded,
                                  response.rotations[:,span],response.angular_velocities[:,span],response.legs[:,span])
            client.state=engine.last_state[span].copy()
            client.expected_positions=frame.positions[-1].copy()
            results.append(frame if frames else frame.response(request['sequence']))
        return results

    def close(self):
        self.restore_command=None
        try:self.host_uploads.close()
        finally:self.upload=None;self.engine.close()


class PhysicsService:
    def __init__(self,max_worlds,wait_seconds=.0005):
        self.max_worlds=max_worlds;self.wait_seconds=wait_seconds
        self.queue=queue.Queue();self.closed=False;self.engines=OrderedDict()
        self.lock=threading.Lock()
        self.cleanup_errors=[];self.cleanup_error_count=0;self.terminal_error=None
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
        with self.lock:
            if self.closed:raise RuntimeError('Physics batch service closed')
            self.queue.put(('round',count))

    def worker_done(self):
        with self.lock:
            if self.closed:raise RuntimeError('Physics batch service closed')
            self.queue.put(('done',0))

    def _control(self,item):
        if isinstance(item[0],str):
            if item[0]=='round':self.round_workers=item[1]
            else:
                self.round_workers-=1
                if self.round_workers==0:self.round_workers=None
            return True
        return False

    @staticmethod
    def _fail(items,error):
        for item in items:
            if item is not None and len(item)==3 and not item[2].done():
                item[2].set_exception(error)

    def _dispose(self,engine):
        try:engine.close()
        except BaseException as error:
            self.cleanup_error_count+=1
            self.cleanup_errors.append(f'{type(error).__name__}: {error}')
            del self.cleanup_errors[:-32]

    def run(self):
        batch=[]
        try:
            while True:
                first=self.queue.get()
                if first is None:break
                if self._control(first):continue
                batch=[first];until=time.monotonic()+self.wait_seconds
                # Explicit round completion still controls batching; shutdown
                # must always wake this wait and settle every queued request.
                while len(batch)<min(self.max_worlds,self.round_workers or self.max_worlds):
                    try:item=self.queue.get(timeout=None if self.round_workers is not None else max(0.,until-time.monotonic()))
                    except queue.Empty:break
                    if item is None:
                        self.queue.put(None);break
                    if self._control(item):continue
                    batch.append(item)
                groups=defaultdict(list)
                for item in batch:
                    try:
                        client=item[0];groups[(client.device,client.scene_signature)].append(item)
                    except BaseException as error:self._fail([item],error)
                for (device,scene),items in groups.items():
                    key=None
                    try:
                        key=(device,scene,tuple(client.design_signature for client,_,_ in items))
                        if key not in self.engines:
                            if len(self.engines)>=4:
                                _,old=self.engines.popitem(last=False);self._dispose(old)
                            self.engines[key]=WorldBatch(device)
                        self.engines.move_to_end(key)
                        started=time.perf_counter()
                        results=self.engines[key].step_many([(c,r) for c,r,_ in items],frames=True)
                        if len(results)!=len(items):raise RuntimeError('Physics batch returned an incomplete result set')
                        self.integration_seconds+=time.perf_counter()-started
                        self.batches+=1;self.world_steps+=len(items);self.max_batch=max(self.max_batch,len(items))
                        self.batch_histogram[len(items)]+=1
                        for item,result in zip(items,results):
                            if not item[2].done():item[2].set_result(result)
                    except BaseException as error:
                        failed=self.engines.pop(key,None)
                        if failed is not None:self._dispose(failed)
                        self._fail(items,error)
                batch=[]
        except BaseException as error:
            self.terminal_error=error
        finally:
            with self.lock:self.closed=True
            error=self.terminal_error or RuntimeError('Physics batch service closed')
            self._fail(batch,error)
            while True:
                try:item=self.queue.get_nowait()
                except queue.Empty:break
                self._fail([item],error)
            for engine in self.engines.values():self._dispose(engine)
            self.engines.clear()

    def reset_metrics(self):
        self.batches=self.world_steps=self.max_batch=0;self.integration_seconds=0.
        self.batch_histogram=defaultdict(int);self.phase_seconds={}

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed=True;self.queue.put(None)
        self.thread.join(timeout=300)
        if self.thread.is_alive():raise RuntimeError('Physics service shutdown timed out')
