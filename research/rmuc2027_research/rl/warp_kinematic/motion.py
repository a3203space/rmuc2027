"""Resident CUDA motion buffers. Coordinates and limits are SI units.

This primitive implements acceleration-limited planar movement, swept
circle/AABB collision and symmetric agent blocking. Terrain transitions belong
to later stages; it must not be advertised as a complete Arena physics adapter.
"""
import numpy as np
import warp as wp


@wp.func
def segment_box(p: wp.vec2, d: wp.vec2, lo: wp.vec2, hi: wp.vec2):
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
              dt: float, output: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    p = pos[i]
    v = wp.vec2(0.0, 0.0)
    if enabled[i] != 0:
        command = commands[i]
        desired = command / wp.max(1.0, wp.length(command)) * limits[i][0]
        change = desired - vel[i]
        v = vel[i] + change * wp.min(1.0, limits[i][1] * dt / wp.max(wp.length(change), 1.e-8))
    delta = v * dt
    radius = limits[i][2]
    # Bounded successive sweeps allow wall sliding without endpoint tunnelling.
    for iteration in range(3):
        hit = wp.vec3(1.0, 0.0, 0.0)
        for j in range(boxes.shape[0]):
            b = boxes[j]
            candidate = segment_box(p, delta, wp.vec2(b[0]-radius,b[1]-radius),
                                    wp.vec2(b[2]+radius,b[3]+radius))
            if candidate[0] < hit[0]:
                hit = candidate
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
        self.output = wp.empty(self.count,dtype=wp.vec2,device=self.device)

    def step(self, commands, dt=.05):
        values = np.asarray(commands,dtype=np.float32)
        if values.shape != (self.count,2) or not np.isfinite(values).all() or not 0 < dt <= .05:
            raise ValueError('Invalid commands or timestep')
        self.commands.assign(values)
        self._advance(dt)

    def _advance(self, dt):
        wp.copy(self.previous,self.pos)
        wp.launch(integrate,self.count,inputs=[self.pos,self.vel,self.commands,self.limits,
                  self.enabled,self.boxes,float(dt),self.output],device=self.device)

        wp.launch(resolve_agents,self.count,inputs=[self.previous,self.pos,self.vel,self.limits,
                  self.worlds,self.resolved,self.resolved_velocity],device=self.device)
        wp.copy(self.pos,self.resolved)
        wp.copy(self.vel,self.resolved_velocity)

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
                   worlds: wp.array(dtype=wp.int32), result: wp.array(dtype=wp.vec2),
                   result_velocity: wp.array(dtype=wp.vec2)):
    i = wp.tid()
    displacement = proposed[i] - previous[i]
    fraction = float(1.0)
    for j in range(previous.shape[0]):
        if i != j and worlds[i] == worlds[j]:
            separation = previous[i]-previous[j]
            relative = displacement-(proposed[j]-previous[j])
            radius = limits[i][2]+limits[j][2]
            a = wp.dot(relative, relative)
            b = wp.dot(separation, relative)
            c = wp.dot(separation,separation)-radius*radius
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


@wp.kernel
def record_sample(positions: wp.array(dtype=wp.vec2), velocities: wp.array(dtype=wp.vec2),
                  samples: wp.array2d(dtype=wp.vec2), speeds: wp.array2d(dtype=wp.vec2), index: int):
    i = wp.tid()
    samples[index,i] = positions[i]
    speeds[index,i] = velocities[i]
