"""Reduced 120 Hz support/flight dynamics for version-two motion frames.

This is a calibrated approximation, not a replacement rigid-body solver.
Flight carries measured horizontal momentum, never a prescribed landing point.
"""
import warp as wp
from .terrain import TerrainData, surface_height


@wp.kernel
def launch(scene: TerrainData, pos: wp.array(dtype=wp.vec2), vel: wp.array(dtype=wp.vec2),
           base: wp.array(dtype=float), capable: wp.array(dtype=wp.int32),
           enabled: wp.array(dtype=wp.int32), command: wp.array(dtype=float),
           latched: wp.array(dtype=float), spec: wp.array(dtype=wp.vec4),
           stance: wp.array(dtype=wp.vec4), leg: wp.array(dtype=float),
           elapsed: wp.array(dtype=float), cooldown: wp.array(dtype=float),
           status: wp.array(dtype=wp.int32), flight: wp.array(dtype=wp.int32),
           z: wp.array(dtype=float), vz: wp.array(dtype=float), goal: wp.array(dtype=wp.vec2),
           tilt: wp.array(dtype=wp.vec4), limits: wp.array(dtype=wp.vec3),
           worlds: wp.array(dtype=wp.int32), dt: float):
    i=wp.tid()
    cooldown[i]=wp.max(0.0, cooldown[i]-dt)
    rising=command[i]>0.0 and latched[i]<.5
    latched[i]=float(command[i]>0.0)
    if rising and capable[i]!=0 and enabled[i]!=0 and flight[i]==0 and cooldown[i]<=0.0:
        clear=int(1)
        push=spec[i][0];speed=9.81*spec[i][3]*push
        flight_time=push+(speed+wp.sqrt(speed*speed+9.81*9.81*spec[i][3]*push*push))/9.81
        landing=pos[i]+vel[i]*flight_time
        for other in range(pos.shape[0]):
            if other!=i and worlds[other]==worlds[i] and wp.length(pos[other]-landing)<limits[i][2]+limits[other][2]+.08:
                clear=0
        if clear!=0 and wp.cos(tilt[i][0])*wp.cos(tilt[i][1])>.95:
            status[i]=1;flight[i]=1;elapsed[i]=0.0
            z[i]=surface_height(scene,pos[i])+base[i];vz[i]=0.0
            cooldown[i]=spec[i][2]
        else:
            status[i]=3;cooldown[i]=.25
    if flight[i]!=0:
        acceleration=float(-9.81)
        if status[i]==1 and elapsed[i]<spec[i][0] and enabled[i]!=0:
            # A pulse owns the whole actuator push, including later frames.
            acceleration=9.81*spec[i][3]
            leg[i]=stance[i][1]
        elif elapsed[i]<spec[i][0]+spec[i][1]:
            leg[i]=stance[i][0]+.01
        elapsed[i]=elapsed[i]+dt
        vz[i]=vz[i]+acceleration*dt
        z[i]=z[i]+vz[i]*dt
        goal[i]=pos[i]+vel[i]*dt


@wp.kernel
def land(pos: wp.array(dtype=wp.vec2), vel: wp.array(dtype=wp.vec2),
         goal: wp.array(dtype=wp.vec2), ground: wp.array(dtype=float), base: wp.array(dtype=float),
         flight: wp.array(dtype=wp.int32), z: wp.array(dtype=float), vz: wp.array(dtype=float),
         status: wp.array(dtype=wp.int32), elapsed: wp.array(dtype=float), spec: wp.array(dtype=wp.vec4),
         boxes: wp.array(dtype=wp.vec4), angles: wp.array(dtype=float), wall_z: wp.array(dtype=wp.vec2),
         masks: wp.array2d(dtype=wp.int32), limits: wp.array(dtype=wp.vec3), envelope: wp.array(dtype=wp.vec4), dt: float):
    i=wp.tid()
    if flight[i]!=0:
        old_z=z[i]-vz[i]*dt
        for j in range(boxes.shape[0]):
            b=boxes[j];c=wp.cos(angles[j]);s=wp.sin(angles[j])
            relative=pos[i]-wp.vec2((b[0]+b[2])*.5,(b[1]+b[3])*.5)
            local=wp.vec2(c*relative[0]+s*relative[1],-s*relative[0]+c*relative[1])
            if masks[i,j]!=0 and wp.abs(local[0])<(b[2]-b[0])*.5+limits[i][2] and wp.abs(local[1])<(b[3]-b[1])*.5+limits[i][2]:
                if vz[i]>0.0 and old_z+envelope[i][0]<=wall_z[j][0] and z[i]+envelope[i][0]>=wall_z[j][0]:
                    z[i]=wall_z[j][0]-envelope[i][0]-.001;vz[i]=0.0;status[i]=4
        if wp.length(pos[i]-goal[i])>.002:
            status[i]=4
        if elapsed[i]>=spec[i][0] and vz[i]<=0.0 and z[i]<=ground[i]+base[i]:
            z[i]=ground[i]+base[i];vz[i]=0.0;flight[i]=0
            if status[i]==1:status[i]=2
            else:status[i]=3


@wp.kernel
def attitude(scene: TerrainData, pos: wp.array(dtype=wp.vec2), vel: wp.array(dtype=wp.vec2),
             previous_velocity: wp.array(dtype=wp.vec2), heading: wp.array(dtype=float),
             flight: wp.array(dtype=wp.int32), enabled: wp.array(dtype=wp.int32),
             support: wp.array(dtype=wp.vec4), geometry: wp.array(dtype=wp.vec4),
             balance: wp.array(dtype=wp.vec4), inertia: wp.array(dtype=wp.vec2),
             tilt: wp.array(dtype=wp.vec4), leg: wp.array(dtype=float),
             stance: wp.array(dtype=wp.vec4), packed: wp.array3d(dtype=float), index: int, dt: float):
    i=wp.tid();a=heading[i];g=geometry[i];b=balance[i];value=tilt[i]
    forward=wp.vec2(wp.cos(a),wp.sin(a));side=wp.vec2(-wp.sin(a),wp.cos(a))
    acceleration=(vel[i]-previous_velocity[i])/dt
    previous_velocity[i]=vel[i]
    target_roll=float(0.0);target_pitch=float(0.0)
    if flight[i]==0:
        left=surface_height(scene,pos[i]+side*g[1]*.5)
        right=surface_height(scene,pos[i]-side*g[1]*.5)
        front=surface_height(scene,pos[i]+forward*g[0]*.5)
        back=surface_height(scene,pos[i]-forward*g[0]*.5)
        target_roll=wp.atan2(left-right,g[1])
        target_pitch=-wp.atan2(front-back,g[0])
        # Two-wheel balance controls pitch about its axle; it does not place
        # fictional front/back tyres on opposite sides of a stair rim.
        if stance[i][1]>0.0:target_pitch=0.0
    for axis in range(2):
        angle=value[axis];omega=value[axis+2]
        target=target_roll
        induced=-wp.dot(acceleration,side)*support[i][1]*g[2]
        if axis==1:
            target=target_pitch
            induced=wp.dot(acceleration,forward)*support[i][1]*g[2]
        induced=wp.clamp(induced,-b[2]*.8,b[2]*.8)
        torque=float(0.0)
        if flight[i]==0:
            torque=wp.clamp(b[0]*(target-angle)-b[1]*omega,-b[2],b[2])+induced
            if enabled[i]==0:
                torque=wp.clamp(b[0]*(target-angle)*.2-b[1]*omega,-b[2],b[2])+induced
            if wp.abs(angle)>1.1 and g[3]<.5:
                torque=support[i][1]*9.81*g[2]*wp.sin(angle)-b[1]*omega*.25
        else:
            torque=-omega*.05
            if enabled[i]!=0 and stance[i][1]>0.0:
                torque=wp.clamp(-40.0*inertia[i][axis]*angle-6.0*omega,-b[3],b[3])
        omega=wp.clamp(omega+torque/wp.max(.05,inertia[i][axis])*dt,-12.0,12.0)
        angle=wp.clamp(angle+omega*dt,-3.14159,3.14159)
        value[axis]=angle;value[axis+2]=omega
    tilt[i]=value
    cr=wp.cos(value[0]*.5);sr=wp.sin(value[0]*.5)
    cp=wp.cos(value[1]*.5);sp=wp.sin(value[1]*.5)
    cy=wp.cos(a*.5);sy=wp.sin(a*.5)
    q=wp.vec4(sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy)
    for k in range(4):packed[index,i,26+k]=q[k]
    # Euler rates are local state; transport angular velocity in world axes.
    packed[index,i,30]=value[2]*wp.cos(value[1])*wp.cos(a)-value[3]*wp.sin(a)
    packed[index,i,31]=value[2]*wp.cos(value[1])*wp.sin(a)+value[3]*wp.cos(a)
    # Native authority adds the measured yaw rate for this substep.
    packed[index,i,32]=-value[2]*wp.sin(value[1])
    left_leg=leg[i];right_leg=leg[i]
    if stance[i][1]>0.0 and flight[i]==0:
        compression=support[i][1]*9.81*.5/wp.max(1.0,support[i][0])
        left_leg=wp.clamp(leg[i]-compression+wp.sin(value[0])*g[1]*.5,stance[i][0]-.01,stance[i][1])
        right_leg=wp.clamp(leg[i]-compression-wp.sin(value[0])*g[1]*.5,stance[i][0]-.01,stance[i][1])
    packed[index,i,33]=left_leg;packed[index,i,34]=right_leg


@wp.kernel
def record_latch(latched: wp.array(dtype=float), packed: wp.array3d(dtype=float)):
    i=wp.tid();packed[5,i,35]=latched[i]
