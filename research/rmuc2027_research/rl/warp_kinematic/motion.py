"""Resident CUDA motion buffers. Coordinates and limits are SI units.

This primitive implements acceleration-limited planar movement, swept
circle/AABB collision and symmetric agent blocking. Terrain transitions belong
to later stages; it must not be advertised as a complete Arena physics adapter.
"""
import numpy as np
import warp as wp
from .terrain import TerrainData, surface_height


@wp.func
def segment_box(p: wp.vec2, d: wp.vec2, lo: wp.vec2, hi: wp.vec2):
    # Float32 roundoff at a previous contact can put the start a fraction
    # inside the slab. Snap only this tolerance to the boundary so tangential
    # motion retains a contact normal and can slide on subsequent sweeps.
    for axis in range(2):
        if wp.abs(p[axis]-lo[axis])<1.e-5:
            p[axis]=lo[axis]
        elif wp.abs(p[axis]-hi[axis])<1.e-5:
            p[axis]=hi[axis]
    enter = float(0.0)
    leave = float(1.0)
    normal = wp.vec2(0.0, 0.0)
    valid = int(1)
    for axis in range(2):
        if wp.abs(d[axis]) < 1.0e-8:
            if p[axis] <= lo[axis] or p[axis] >= hi[axis]:
                valid = 0
        else:
            a = (lo[axis] - p[axis]) / d[axis]
            b = (hi[axis] - p[axis]) / d[axis]
            near = wp.min(a, b)
            far = wp.max(a, b)
            if near >= enter:
                enter = near
                normal = wp.vec2(0.0, 0.0)
                normal[axis] = -wp.sign(d[axis])
            leave = wp.min(leave, far)
    result = wp.vec3(1.0, 0.0, 0.0)
    if valid == 1 and enter < leave and leave > 0.0 and enter < 1.0:
        result = wp.vec3(enter, normal[0], normal[1])
    return result


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec2), vel: wp.array(dtype=wp.vec2),
              commands: wp.array(dtype=wp.vec2), limits: wp.array(dtype=wp.vec3),
              enabled: wp.array(dtype=wp.int32), boxes: wp.array(dtype=wp.vec4),
              angles: wp.array(dtype=float), wall_z: wp.array(dtype=wp.vec2),
              body_z: wp.array(dtype=wp.vec2), masks: wp.array2d(dtype=wp.int32),
              drive: wp.array(dtype=wp.vec3), resistance: wp.array(dtype=wp.vec2),
              flight: wp.array(dtype=wp.int32), goal: wp.array(dtype=wp.vec2), dt: float, output: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    p = pos[i]
    v = wp.vec2(0.0, 0.0)
    if enabled[i] != 0:
        command = commands[i]
        desired = command / wp.max(1.0, wp.length(command)) * limits[i][0]
        change = desired - vel[i]
        v = vel[i] + change * wp.min(1.0, limits[i][1] * dt / wp.max(wp.length(change), 1.e-8))
        # Positive kinetic work is bounded by electrical budget and drivetrain
        # efficiency; passive resistance still acts when commanded power is zero.
        old_speed=wp.length(vel[i]);spec=drive[i]
        loss=(resistance[i][0]*spec[0]*9.81+0.6125*resistance[i][1]*old_speed*old_speed)*old_speed
        maximum=wp.sqrt(wp.max(0.0,old_speed*old_speed+2.0*dt*(wp.max(0.0,spec[2]-3.0)*spec[1]-loss)/spec[0]))
        if wp.length(v)>maximum:
            v=v*(maximum/wp.max(wp.length(v),1.e-8))
    if flight[i]!=0 and enabled[i]!=0:
        v=(goal[i]-p)/dt
    delta = v * dt
    radius = limits[i][2]
    # Bounded successive sweeps allow wall sliding without endpoint tunnelling.
    for iteration in range(3):
        hit = wp.vec3(1.0, 0.0, 0.0)
        for j in range(boxes.shape[0]):
            b = boxes[j]
            if masks[i,j]!=0 and body_z[i][1] > wall_z[j][0] and body_z[i][0] < wall_z[j][1]:
                center=wp.vec2((b[0]+b[2])*0.5,(b[1]+b[3])*0.5)
                half=wp.vec2((b[2]-b[0])*0.5+radius,(b[3]-b[1])*0.5+radius)
                c=wp.cos(angles[j]);s=wp.sin(angles[j]);relative=p-center
                local=wp.vec2(c*relative[0]+s*relative[1],-s*relative[0]+c*relative[1])
                direction=wp.vec2(c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1])
                candidate=segment_box(local,direction,-half,half)
                if candidate[0] < hit[0]:
                    hit=wp.vec3(candidate[0],c*candidate[1]-s*candidate[2],s*candidate[1]+c*candidate[2])
        fraction = wp.max(0.0, hit[0] - 1.e-5)
        if hit[0] >= 1.0:
            fraction = 1.0
        p = p + delta * fraction
        normal = wp.vec2(hit[1],hit[2])
        delta = delta * (1.0-fraction)
        delta = delta - normal * wp.min(0.0,wp.dot(delta,normal))
        v = v - normal * wp.min(0.0,wp.dot(v,normal))
    pos[i] = p
    vel[i] = v
    output[i] = p


class MotionBatch:
    """World-isolated agents sharing a static map; no CPU fallback."""
    def __init__(self, positions, limits, boxes, device='cuda:0', worlds=None):
        wp.init()
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError('MotionBatch requires an explicit CUDA device')
        p = np.asarray(positions, dtype=np.float32)
        values = np.asarray(limits, dtype=np.float32)
        walls = np.asarray(boxes, dtype=np.float32).reshape(-1,4)
        if p.ndim != 2 or p.shape[1] != 2 or values.shape != (len(p),3) or len(p)==0:
            raise ValueError('Expected nonempty positions[N,2] and limits[N,3]')
        if not all(np.isfinite(a).all() for a in (p,values,walls)) or (values<=0).any():
            raise ValueError('Finite inputs and positive speed/acceleration/radius required')
        if len(walls) and (walls[:,2:] <= walls[:,:2]).any():
            raise ValueError('Box maximum must exceed minimum')
        world_ids = np.zeros(len(p), dtype=np.int32) if worlds is None else np.asarray(worlds, dtype=np.int32)
        if world_ids.shape != (len(p),) or (world_ids < 0).any():
            raise ValueError('Expected nonnegative world ids[N]')
        self.count = len(p)
        self.worlds = wp.array(world_ids,dtype=wp.int32,device=self.device)
        self.flight = wp.zeros(self.count,dtype=wp.int32,device=self.device)
        self.flight_goal = wp.empty(self.count,dtype=wp.vec2,device=self.device)
        self.flight_z = wp.zeros(self.count,dtype=float,device=self.device)
        self.previous = wp.empty(self.count,dtype=wp.vec2,device=self.device)
        self.resolved = wp.empty(self.count,dtype=wp.vec2,device=self.device)
        self.resolved_velocity = wp.empty(self.count,dtype=wp.vec2,device=self.device)
        self.samples = wp.empty((6,self.count),dtype=wp.vec2,device=self.device)
        self.sample_velocities = wp.empty((6,self.count),dtype=wp.vec2,device=self.device)
        self.pos = wp.array(p,dtype=wp.vec2,device=self.device)
        self.vel = wp.zeros(self.count,dtype=wp.vec2,device=self.device)
        self.commands = wp.zeros(self.count,dtype=wp.vec2,device=self.device)
        self.enabled = wp.ones(self.count,dtype=wp.int32,device=self.device)
        self.limits = wp.array(values,dtype=wp.vec3,device=self.device)
        self.boxes = wp.array(walls,dtype=wp.vec4,device=self.device)
        self.wall_masks = wp.ones((self.count,len(walls)),dtype=wp.int32,device=self.device)
        self.drive = wp.array([[1.,1.,1.e9]]*self.count,dtype=wp.vec3,device=self.device)
        self.resistance = wp.zeros(self.count,dtype=wp.vec2,device=self.device)
        self.angles = wp.zeros(len(walls),dtype=float,device=self.device)
        self.wall_z = wp.array([[-1.e6,1.e6]]*len(walls),dtype=wp.vec2,device=self.device)
        self.body_z = wp.array([[0.,1.]]*self.count,dtype=wp.vec2,device=self.device)
        self.height_offsets = wp.array([[0.,1.]]*self.count,dtype=wp.vec2,device=self.device)
        self.terrain = None
        self.flat_scene = TerrainData()
        self.heights = wp.zeros(self.count,dtype=float,device=self.device)
        self.step_limits = wp.array([.04]*self.count,dtype=float,device=self.device)
        self.output = wp.empty(self.count,dtype=wp.vec2,device=self.device)

    def step(self, commands, dt=.05):
        values = np.asarray(commands,dtype=np.float32)
        if values.shape != (self.count,2) or not np.isfinite(values).all() or not 0 < dt <= .05:
            raise ValueError('Invalid commands or timestep')
        self.commands.assign(values)
        self._advance(dt)

    def _advance(self, dt, launch=wp.launch):
        wp.copy(self.previous,self.pos)
        if self.terrain is not None:
            from .terrain import sample_heights, update_body_height
            launch(sample_heights,self.count,inputs=[self.terrain.data,self.pos,self.heights],device=self.device)
            launch(update_body_height,self.count,inputs=[self.heights,self.height_offsets,self.body_z],device=self.device)
        from .traversal import jump_envelope
        launch(jump_envelope,self.count,inputs=[self.flight,self.flight_z,self.body_z],device=self.device)
        launch(integrate,self.count,inputs=[self.pos,self.vel,self.commands,self.limits,
                  self.enabled,self.boxes,self.angles,self.wall_z,self.body_z,self.wall_masks,self.drive,self.resistance,self.flight,self.flight_goal,float(dt),self.output],device=self.device)

        if self.terrain is not None:
            from .terrain import limit_terrain
            launch(limit_terrain,self.count,inputs=[self.terrain.data,self.previous,self.pos,
                      self.vel,self.limits,self.step_limits,self.heights,self.flight],device=self.device)
        launch(resolve_agents,self.count,inputs=[self.previous,self.pos,self.vel,self.limits,
                  self.worlds,self.body_z,self.enabled,self.flight,self.boxes,self.angles,self.wall_z,self.wall_masks,
                  self.terrain.data if self.terrain is not None else self.flat_scene,int(self.terrain is not None),
                  self.step_limits,self.resolved,self.resolved_velocity],device=self.device)
        wp.copy(self.pos,self.resolved)
        wp.copy(self.vel,self.resolved_velocity)
        if self.terrain is not None:
            from .terrain import sample_heights
            launch(sample_heights,self.count,inputs=[self.terrain.data,self.pos,self.heights],device=self.device)

    def frame(self, commands):
        """Queue six 120 Hz samples; one readback per 50 ms rule frame."""
        values = np.asarray(commands,dtype=np.float32)
        if values.shape != (self.count,2) or not np.isfinite(values).all():
            raise ValueError('Invalid commands')
        self.commands.assign(values)
        for index in range(6):
            self._advance(1./120.)
            wp.launch(record_sample,self.count,inputs=[self.pos,self.vel,self.samples,
                      self.sample_velocities,index],device=self.device)
        return self.samples.numpy(), self.sample_velocities.numpy()

    def state(self):
        return self.pos.numpy(), self.vel.numpy()


@wp.kernel
def resolve_agents(previous: wp.array(dtype=wp.vec2), proposed: wp.array(dtype=wp.vec2),
                   velocity: wp.array(dtype=wp.vec2), limits: wp.array(dtype=wp.vec3),
                   worlds: wp.array(dtype=wp.int32), body_z: wp.array(dtype=wp.vec2),
                   enabled: wp.array(dtype=wp.int32), flight: wp.array(dtype=wp.int32),
                   boxes: wp.array(dtype=wp.vec4), angles: wp.array(dtype=float), wall_z: wp.array(dtype=wp.vec2),
                   masks: wp.array2d(dtype=wp.int32), scene: TerrainData, terrain_enabled: int,
                   steps: wp.array(dtype=float), result: wp.array(dtype=wp.vec2),
                   result_velocity: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    displacement = proposed[i] - previous[i]
    fraction = float(1.0)
    correction=wp.vec2(0.0,0.0)
    for j in range(previous.shape[0]):
        if i != j and worlds[i] == worlds[j] and body_z[i][1]>body_z[j][0] and body_z[j][1]>body_z[i][0]:
            separation = previous[i]-previous[j]
            relative = displacement-(proposed[j]-previous[j])
            radius = limits[i][2]+limits[j][2]
            a = wp.dot(relative, relative)
            b = wp.dot(separation, relative)
            c = wp.dot(separation,separation)-radius*radius
            if c < -1.e-6 and enabled[i]!=0 and flight[i]==0 and flight[j]==0 and wp.length(proposed[i]-proposed[j])<radius:
                distance=wp.length(separation)
                normal=wp.vec2(1.0,0.0)
                if i<j:normal=wp.vec2(-1.0,0.0)
                if distance>1.e-6:normal=separation/distance
                share=float(1.0)
                if enabled[j]!=0:share=.5
                correction=correction+normal*(radius-distance+.0001)*share
            if b < 0.0:
                if c <= 0.0:
                    fraction = 0.0
                elif a > 1.e-12:
                    discriminant = b*b-a*c
                    if discriminant >= 0.0:
                        contact = (-b-wp.sqrt(discriminant))/a
                        if contact >= 0.0 and contact <= 1.0:
                            fraction = wp.min(fraction,wp.max(0.0,contact-1.e-5))
    result[i] = previous[i] + fraction*displacement
    result_velocity[i] = velocity[i]
    if fraction < 1.0:
        result_velocity[i] = wp.vec2(0.0,0.0)
    if wp.length(correction)>1.e-6:
        correction=correction*wp.min(1.0,.02/wp.length(correction))
        start=result[i];safe=float(1.0)
        for j in range(boxes.shape[0]):
            if masks[i,j]!=0 and body_z[i][1]>wall_z[j][0] and body_z[i][0]<wall_z[j][1]:
                wall=boxes[j];center=wp.vec2((wall[0]+wall[2])*.5,(wall[1]+wall[3])*.5)
                half=wp.vec2((wall[2]-wall[0])*.5+limits[i][2],(wall[3]-wall[1])*.5+limits[i][2])
                c=wp.cos(angles[j]);s=wp.sin(angles[j]);p=start-center
                hit=segment_box(wp.vec2(c*p[0]+s*p[1],-s*p[0]+c*p[1]),
                                wp.vec2(c*correction[0]+s*correction[1],-s*correction[0]+c*correction[1]),-half,half)
                safe=wp.min(safe,wp.max(0.0,hit[0]-1.e-5))
        end=start+correction*safe
        if terrain_enabled!=0:
            accepted=start;old_height=surface_height(scene,start);blocked=int(0)
            for sample in range(1,5):
                p=start+(end-start)*(float(sample)/4.0);height=surface_height(scene,p)
                if p[0]<limits[i][2] or p[0]>scene.width-limits[i][2] or p[1]<limits[i][2] or p[1]>scene.height-limits[i][2] or wp.abs(height-old_height)>steps[i]:blocked=1
                if blocked==0:accepted=p;old_height=height
            end=accepted
        result[i]=end;result_velocity[i]=wp.vec2(0.0,0.0)


@wp.kernel
def record_sample(positions: wp.array(dtype=wp.vec2), velocities: wp.array(dtype=wp.vec2),
                  samples: wp.array2d(dtype=wp.vec2), speeds: wp.array2d(dtype=wp.vec2), index: int):
    i = wp.tid()
    samples[index,i] = positions[i]
    speeds[index,i] = velocities[i]
