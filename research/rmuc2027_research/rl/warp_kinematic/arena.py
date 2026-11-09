"""Development Arena adapter: GPU planar motion, terrain and folding.

Experimental RL integration; full traversal, energy and transfer gates remain incomplete.
"""
import math
import numpy as np
import warp as wp
from .motion import MotionBatch
from .terrain import Terrain, TerrainData, surface_height
from .protocol import MotionFrame, DT


@wp.func
def wrapped(angle: float):
    return wp.atan2(wp.sin(angle),wp.cos(angle))


@wp.kernel
def controls(pos: wp.array(dtype=wp.vec2), ground: wp.array(dtype=float),
             heading: wp.array(dtype=float), fold: wp.array(dtype=float),
             requested: wp.array(dtype=wp.vec2), targets: wp.array(dtype=wp.vec2),
             flags: wp.array(dtype=wp.vec3i), params: wp.array(dtype=wp.vec4),
             legged: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.int32),
             limits: wp.array(dtype=wp.vec3), boxes: wp.array(dtype=wp.vec4),
             angles: wp.array(dtype=float), wall_z: wp.array(dtype=wp.vec2), masks: wp.array2d(dtype=wp.int32),
             offsets: wp.array(dtype=wp.vec2), commands: wp.array(dtype=wp.vec2)):
    i=wp.tid();a=heading[i];spec=params[i];command=requested[i]
    if enabled[i]!=0:
        was_folded=fold[i]>.999
        goal=a
        if wp.length(command)>.01:
            goal=wp.atan2(command[1],command[0])
            if legged[i]!=0 and wp.abs(wrapped(goal+3.14159265359-a))<wp.abs(wrapped(goal-a)):
                goal=wrapped(goal+3.14159265359)
        if flags[i][0]!=0:goal=targets[i][0]
        rate=wp.clamp(wrapped(goal-a)/DT,-spec[3],spec[3])
        if flags[i][1]!=0:rate=wp.clamp(targets[i][1],-spec[3],spec[3])
        a=wrapped(a+rate*DT)
        heading[i]=a
        progress=fold[i]
        if spec[2]>0.0:
            target=float(flags[i][2]);candidate=progress+wp.clamp(target-progress,-DT/spec[2],DT/spec[2])
            top=ground[i]+offsets[i][0]+spec[0]+(spec[1]-spec[0])*candidate
            clear=int(1)
            if candidate<progress:
                # Do not begin unfolding under a ceiling which cannot admit
                # the final standing envelope, even if the first increment fits.
                top=ground[i]+offsets[i][0]+spec[0]
                for j in range(boxes.shape[0]):
                    b=boxes[j];c=wp.cos(angles[j]);s=wp.sin(angles[j])
                    relative=pos[i]-wp.vec2((b[0]+b[2])*0.5,(b[1]+b[3])*0.5)
                    local=wp.vec2(c*relative[0]+s*relative[1],-s*relative[0]+c*relative[1])
                    if masks[i,j]!=0 and wp.abs(local[0])<(b[2]-b[0])*0.5+limits[i][2] and wp.abs(local[1])<(b[3]-b[1])*0.5+limits[i][2]:
                        if top>wall_z[j][0] and ground[i]+offsets[i][0]<wall_z[j][1]:clear=0
            if clear!=0:progress=candidate
        fold[i]=progress
        offsets[i]=wp.vec2(offsets[i][0],offsets[i][0]+spec[0]+(spec[1]-spec[0])*progress)
        # Match the existing physical controller: a fully folded chassis has
        # half its normal requested speed, using the pre-substep fold state.
        if was_folded:command=command*.5
        if legged[i]!=0:
            forward=wp.vec2(wp.cos(a),wp.sin(a))
            command=forward*wp.dot(command,forward)
    else:command=wp.vec2(0.0,0.0)
    commands[i]=command


@wp.kernel
def sync_passive(scene: TerrainData,positions: wp.array(dtype=wp.vec2),z: wp.array(dtype=float),
                 vz: wp.array(dtype=float),base: wp.array(dtype=float),enabled: wp.array(dtype=wp.int32),
                 ground_agents: wp.array(dtype=wp.int32),status: wp.array(dtype=wp.int32),
                 altitude: wp.array(dtype=float),fall_speed: wp.array(dtype=float)):
    i=wp.tid()
    if enabled[i]==0 and ground_agents[i]!=0:
        if z[i]>surface_height(scene,positions[i])+base[i]+.005:
            if status[i]!=4:fall_speed[i]=wp.min(0.0,vz[i])
            status[i]=4;altitude[i]=z[i]
        else:status[i]=0


@wp.kernel
def reset_traversal(mask: wp.array(dtype=wp.int32),status: wp.array(dtype=wp.int32),
                    elapsed: wp.array(dtype=float),cooldown: wp.array(dtype=float),fall_speed: wp.array(dtype=float)):
    i=wp.tid()
    if mask[i]!=0:
        status[i]=0;elapsed[i]=0.0;cooldown[i]=0.0;fall_speed[i]=0.0


@wp.kernel
def record(pos: wp.array(dtype=wp.vec2), vel: wp.array(dtype=wp.vec2),
           ground: wp.array(dtype=float), offsets: wp.array(dtype=wp.vec2),
           headings: wp.array(dtype=float), folds: wp.array(dtype=float),
           last_z: wp.array(dtype=float), positions: wp.array2d(dtype=wp.vec3),
           velocities: wp.array2d(dtype=wp.vec3), angles: wp.array2d(dtype=float),
           folded: wp.array2d(dtype=float), flight: wp.array(dtype=wp.int32), altitude: wp.array(dtype=float),
           grounded: wp.array2d(dtype=wp.int32), packed: wp.array3d(dtype=float), index: int,
           starts: wp.array(dtype=wp.vec3), ends: wp.array(dtype=wp.vec3),
           elapsed: wp.array(dtype=float), durations: wp.array(dtype=float), cooldown: wp.array(dtype=float),
           status: wp.array(dtype=wp.int32), fall_speed: wp.array(dtype=float), base: wp.array(dtype=float),
           carry_state: int):
    i=wp.tid();z=ground[i]+offsets[i][0]
    if flight[i]!=0:z=altitude[i]
    grounded[index,i]=1-flight[i]
    positions[index,i]=wp.vec3(pos[i][0],pos[i][1],z)
    velocities[index,i]=wp.vec3(vel[i][0],vel[i][1],(z-last_z[i])/DT)
    last_z[i]=z;angles[index,i]=headings[i];folded[index,i]=folds[i]
    for axis in range(3):
        packed[index,i,axis]=positions[index,i][axis]
        packed[index,i,axis+3]=velocities[index,i][axis]
    packed[index,i,6]=headings[i];packed[index,i,7]=folds[i];packed[index,i,8]=float(grounded[index,i])
    if carry_state!=0 and index==5:
        for axis in range(3):
            packed[index,i,9+axis]=starts[i][axis];packed[index,i,12+axis]=ends[i][axis]
        packed[index,i,15]=elapsed[i];packed[index,i,16]=durations[i];packed[index,i,17]=cooldown[i]
        packed[index,i,18]=float(status[i]);packed[index,i,19]=fall_speed[i];packed[index,i,20]=altitude[i]
        packed[index,i,21]=offsets[i][0];packed[index,i,22]=offsets[i][1]
        packed[index,i,23]=base[i];packed[index,i,24]=float(flight[i])


class ArenaMotion:
    protocol='kinematic_frame'
    def __init__(self,device='cuda:0',use_graph=False,reuse_commands=True,packed_readback=True,carry_state=False):
        self.device=device;self.motion=None;self.next_sequence=0;self.closed=False
        self.use_graph=use_graph;self.graph=None
        self.packed_readback=packed_readback
        self.carry_state=carry_state;self.last_state=None
        self.reuse_commands=reuse_commands;self.launches=[];self.launch_index=0
        self.expected_positions=None

    def initialize(self,request):
        robots=request['robots'];self.ids=tuple(r['id'] for r in robots)
        self.designs=[r['design'] for r in robots]
        self.terrain=Terrain(request['arena'],self.device)
        physical=request['arena']['obstacles']
        self.physical_count=len(physical)
        self.forbidden=request['arena'].get('forbidden',[])
        obstacles=physical+[dict(x=z['x'],y=z['y'],w=z['w'],h=z['h'],bottom=-1.e6,height=2.e6) for z in self.forbidden]
        boxes=[[w['x'],w['y'],w['x']+w['w'],w['y']+w['h']] for w in obstacles]
        limits=[];params=[]
        for r,d in zip(robots,self.designs):
            limits.append([max(.1,d.get('speed_reference_mps',1.)),
                           min(6.,max(.2,d.get('brake_force_max_n',100)/max(1.,d.get('mass_kg',25)))),r['radius']])
            params.append([d.get('muzzle_z_m',.4)+.06,d.get('folded_muzzle_z_m',d.get('muzzle_z_m',.4))+.06,
                           d.get('transform_time_s',0) if d.get('transform_allowed') else 0.,3.])
        m=self.motion=MotionBatch([r['pos'][:2] for r in robots],limits,boxes,self.device,
            worlds=[len(robots)+i if r['role']=='aerial' else r.get('_world',0) for i,r in enumerate(robots)])
        m.terrain=self.terrain
        m.angles.assign(np.asarray([w.get('rotation',0) for w in obstacles],dtype=np.float32))
        m.wall_z.assign(np.asarray([[w.get('bottom',0),w.get('bottom',0)+w['height']] for w in obstacles],dtype=np.float32))
        floor=self.terrain.heights([r['pos'][:2] for r in robots])
        self.base_offsets=np.maximum(0.,np.asarray([r['pos'][2] for r in robots])-floor)
        m.height_offsets.assign(np.asarray([[z,z+p[0]] for z,p in zip(self.base_offsets,params)],dtype=np.float32))
        n=m.count
        def array(values,dtype):return wp.array(values,dtype=dtype,device=m.device)
        self.params=array(params,wp.vec4)
        self.ground_agents=array([int(r['role']!='aerial') for r in robots],wp.int32)
        self.base=array(self.base_offsets,wp.float32)
        supported=request['arena'].get('layout_revision')=='chapter4-2027-v1'
        self.gates=array([[8.1,6.9,9.15,8.1],[18.85,6.9,19.9,8.1]] if supported else [],wp.vec4)
        self.destinations=array([[10.85,7.5],[17.15,7.5]] if supported else [],wp.vec2)
        self.capable=array([int(bool(d.get('capabilities',{}).get('jump')) and d.get('capabilities',{}).get('step_climb_m',0)>0) for d in self.designs],wp.int32)
        self.jump_command=wp.zeros(n,dtype=float,device=m.device)
        self.jump_start=wp.zeros(n,dtype=wp.vec3,device=m.device);self.jump_end=wp.zeros(n,dtype=wp.vec3,device=m.device)
        self.jump_elapsed=wp.zeros(n,dtype=float,device=m.device);self.jump_duration=wp.zeros(n,dtype=float,device=m.device)
        self.jump_cooldown=wp.zeros(n,dtype=float,device=m.device);self.jump_status=wp.zeros(n,dtype=wp.int32,device=m.device)
        self.fall_speed=wp.zeros(n,dtype=float,device=m.device)
        self.passive_vz=wp.zeros(n,dtype=float,device=m.device)
        self.grounded=wp.empty((6,n),dtype=wp.int32,device=m.device)
        self.legged=array([int(d.get('chassis_type')=='wheel_legged') for d in self.designs],wp.int32)
        self.heading=wp.zeros(n,dtype=float,device=m.device);self.fold=wp.zeros(n,dtype=float,device=m.device)
        self.requested=wp.zeros(n,dtype=wp.vec2,device=m.device);self.targets=wp.zeros(n,dtype=wp.vec2,device=m.device)
        self.flags=wp.zeros(n,dtype=wp.vec3i,device=m.device);self.last_z=wp.zeros(n,dtype=float,device=m.device)
        self.packed=wp.zeros((6,n,25 if self.carry_state else 9),dtype=float,device=m.device)
        m.resistance.assign(np.asarray([[d.get('rolling_resistance',0),d.get('drag_area_m2',0)] for d in self.designs],dtype=np.float32))
        self.positions=wp.empty((6,n),dtype=wp.vec3,device=m.device);self.velocities=wp.empty((6,n),dtype=wp.vec3,device=m.device)
        self.angles=wp.empty((6,n),dtype=float,device=m.device);self.folds=wp.empty((6,n),dtype=float,device=m.device)

    def step(self,request):
        if self.closed:raise RuntimeError('Adapter closed')
        if request.get('op')!='kinematic_frame' or request.get('version')!=1 or request.get('sequence')!=self.next_sequence or request.get('substeps')!=6 or abs(request.get('dt',0)-DT)>1e-12:
            raise ValueError('Invalid motion request identity')
        robots=request['robots']
        if self.motion is None:self.initialize(request)
        if tuple(r['id'] for r in robots)!=self.ids:raise ValueError('Robot order changed')
        m=self.motion
        authoritative=np.asarray([r['pos'] for r in robots],dtype=np.float64)
        if self.expected_positions is not None:
            changed=np.linalg.norm(authoritative-self.expected_positions,axis=1)>.05
            if changed.any():
                mask=wp.array(changed.astype(np.int32),dtype=wp.int32,device=m.device)
                wp.launch(reset_traversal,m.count,inputs=[mask,self.jump_status,self.jump_elapsed,
                          self.jump_cooldown,self.fall_speed],device=m.device)
        def assign(target,values):target.assign(np.asarray(values,dtype=np.float32))
        assign(m.pos,[r['pos'][:2] for r in robots]);assign(m.vel,[r['velocity'][:2] for r in robots])
        assign(self.heading,[r['heading'] for r in robots]);assign(self.fold,[r['folded'] for r in robots])
        assign(self.jump_command,[r.get('jump',0.) for r in robots])
        assign(self.last_z,[r['pos'][2] for r in robots]);assign(self.requested,[r['move'] for r in robots])
        assign(self.targets,[[r.get('heading_goal') or 0.,r.get('yaw_rate') or 0.] for r in robots])
        self.flags.assign(np.asarray([[r.get('heading_goal') is not None,r.get('yaw_rate') is not None,r['transform']] for r in robots],dtype=np.int32))
        m.enabled.assign(np.asarray([r['enabled'] and r['role']!='aerial' for r in robots],dtype=np.int32))
        m.coast.assign(np.asarray([bool(r.get('coast',False)) and math.hypot(*r['move'])<.01
                                  for r in robots],dtype=np.int32))
        if any(not r['enabled'] and not r.get('grounded',True) and r['role']!='aerial' for r in robots):
            assign(self.passive_vz,[r['velocity'][2] for r in robots])
            wp.launch(sync_passive,m.count,inputs=[self.terrain.data,m.pos,self.last_z,self.passive_vz,self.base,
                      m.enabled,self.ground_agents,self.jump_status,m.flight_z,self.fall_speed],device=m.device)
        assign(m.drive,[[d.get('mass_kg',25),max(.1,d.get('drive_efficiency',.75)),r.get('motor_budget',r['power_limit'])] for r,d in zip(robots,self.designs)])
        masks=[]
        for r in robots:
            masks.append([1]*self.physical_count+[int(z['kind'] in ('base','dart_station','pad') or
                z['kind']=='highway' and r.get('fly_pass',0)<2 or z.get('owner') not in (None,r['team'])) for z in self.forbidden])
        m.wall_masks.assign(np.asarray(masks,dtype=np.int32))
        if self.use_graph and self.next_sequence>0:
            if self.graph is None:
                # The first frame loads all kernels before capture. All arrays
                # retain their addresses; subsequent requests update contents.
                with wp.ScopedCapture(device=m.device) as capture:
                    self._simulate()
                self.graph=capture.graph
            wp.capture_launch(self.graph)
        else:
            self._simulate()
        if self.packed_readback:
            samples=self.packed.numpy()
            if self.carry_state:self.last_state=samples[-1,:,9:].copy()
            frame=MotionFrame(self.ids,samples[:,:,:3],samples[:,:,3:6],samples[:,:,6],samples[:,:,7],samples[:,:,8].astype(bool))
        else:
            frame=MotionFrame(self.ids,self.positions.numpy(),self.velocities.numpy(),self.angles.numpy(),self.folds.numpy(),
                              self.grounded.numpy().astype(bool))
        reply=frame.response(request['sequence'])
        self.expected_positions=frame.positions[-1].copy()
        self.next_sequence+=1
        return reply

    def _launch(self,*args,**kwargs):
        if not self.reuse_commands:
            return wp.launch(*args,**kwargs)
        if self.launch_index==len(self.launches):
            self.launches.append(wp.launch(*args,**kwargs,record_cmd=True))
        command=self.launches[self.launch_index]
        self.launch_index+=1
        command.launch()

    def _simulate(self):
        m=self.motion
        self.launch_index=0
        from .terrain import sample_heights
        for index in range(6):
            self._launch(sample_heights,m.count,inputs=[self.terrain.data,m.pos,m.heights],device=m.device)
            self._launch(controls,m.count,inputs=[m.pos,m.heights,self.heading,self.fold,self.requested,self.targets,self.flags,
                      self.params,self.legged,m.enabled,m.limits,m.boxes,m.angles,m.wall_z,m.wall_masks,m.height_offsets,m.commands],device=m.device)
            from .traversal import prepare_jump, finish_jump
            self._launch(prepare_jump,m.count,inputs=[self.terrain.data,m.pos,m.vel,self.heading,self.base,self.params,m.limits,m.enabled,
                self.capable,self.jump_command,self.gates,self.destinations,m.boxes,m.angles,m.wall_z,m.wall_masks,m.worlds,
                self.jump_start,self.jump_end,self.jump_elapsed,self.jump_duration,self.jump_cooldown,self.jump_status,
                m.flight,m.flight_goal,m.flight_z,self.fall_speed,DT],device=m.device)
            m._advance(DT,launch=self._launch)
            self._launch(finish_jump,m.count,inputs=[m.pos,m.vel,m.flight,m.flight_z,m.heights,self.base,self.jump_end,
                self.jump_elapsed,self.jump_duration,self.jump_cooldown,self.jump_status,m.flight_goal,self.fall_speed],device=m.device)
            self._launch(record,m.count,inputs=[m.pos,m.vel,m.heights,m.height_offsets,self.heading,self.fold,self.last_z,
                      self.positions,self.velocities,self.angles,self.folds,m.flight,m.flight_z,self.grounded,self.packed,index,
                      self.jump_start,self.jump_end,self.jump_elapsed,self.jump_duration,self.jump_cooldown,
                      self.jump_status,self.fall_speed,self.base,int(self.carry_state)],device=m.device)

    def close(self):
        self.closed=True
        self.graph=None
        self.launches.clear()
        self.motion=None
