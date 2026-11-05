"""CUDA timed traversal of explicit map links; no general step-height exemption."""
import warp as wp
from .terrain import TerrainData, surface_height
from .motion import segment_box


@wp.func
def arc_position(start: wp.vec3, end: wp.vec3, progress: float, rise: float):
    p=start+(end-start)*progress
    p[2]=p[2]+4.0*rise*progress*(1.0-progress)
    return p


@wp.kernel
def prepare_jump(scene: TerrainData, positions: wp.array(dtype=wp.vec2),
                 velocities: wp.array(dtype=wp.vec2), heading: wp.array(dtype=float),
                 base: wp.array(dtype=float), envelope: wp.array(dtype=wp.vec4),
                 limits: wp.array(dtype=wp.vec3), enabled: wp.array(dtype=wp.int32),
                 capable: wp.array(dtype=wp.int32), commands: wp.array(dtype=float),
                 gates: wp.array(dtype=wp.vec4), destinations: wp.array(dtype=wp.vec2),
                 boxes: wp.array(dtype=wp.vec4), angles: wp.array(dtype=float),
                 wall_z: wp.array(dtype=wp.vec2), masks: wp.array2d(dtype=wp.int32),
                 worlds: wp.array(dtype=wp.int32),
                 starts: wp.array(dtype=wp.vec3), ends: wp.array(dtype=wp.vec3),
                 elapsed: wp.array(dtype=float), durations: wp.array(dtype=float), cooldown: wp.array(dtype=float),
                 status: wp.array(dtype=wp.int32), flight: wp.array(dtype=wp.int32),
                 goal: wp.array(dtype=wp.vec2), altitude: wp.array(dtype=float), dt: float):
    i=wp.tid();p=positions[i];flight[i]=0
    cooldown[i]=wp.max(0.0,cooldown[i]-dt)
    if enabled[i]==0:
        status[i]=0
    if status[i]!=1 and capable[i]!=0 and enabled[i]!=0 and commands[i]>0.0 and cooldown[i]<=0.0:
        selected=int(-1)
        for j in range(gates.shape[0]):
            g=gates[j];target=destinations[j]
            direction=wp.normalize(target-p)
            if p[0]>=g[0] and p[0]<=g[2] and p[1]>=g[1] and p[1]<=g[3] and wp.dot(direction,wp.vec2(wp.cos(heading[i]),wp.sin(heading[i])))>.95:
                selected=j
        if selected>=0:
            target=destinations[selected]
            start=wp.vec3(p[0],p[1],surface_height(scene,p)+base[i])
            end=wp.vec3(target[0],target[1],surface_height(scene,target)+base[i])
            clear=int(1)
            # Entry and exit occupy real space; an occupied landing rejects launch.
            for other in range(positions.shape[0]):
                if i!=other and worlds[i]==worlds[other] and wp.length(positions[other]-target)<limits[i][2]+limits[other][2]+.1:
                    clear=0
            previous=start
            for sample in range(1,33):
                current=arc_position(start,end,float(sample)/32.0,.8)
                point=wp.vec2(current[0],current[1])
                if current[2]+.005<surface_height(scene,point)+base[i]:clear=0
                for j in range(boxes.shape[0]):
                    if masks[i,j]!=0 and wp.max(previous[2],current[2])+envelope[i][0]>wall_z[j][0] and wp.min(previous[2],current[2])<wall_z[j][1]:
                        b=boxes[j];center=wp.vec2((b[0]+b[2])*.5,(b[1]+b[3])*.5)
                        half=wp.vec2((b[2]-b[0])*.5+limits[i][2],(b[3]-b[1])*.5+limits[i][2])
                        c=wp.cos(angles[j]);s=wp.sin(angles[j])
                        rel=wp.vec2(previous[0],previous[1])-center
                        delta=point-wp.vec2(previous[0],previous[1])
                        hit=segment_box(wp.vec2(c*rel[0]+s*rel[1],-s*rel[0]+c*rel[1]),
                                        wp.vec2(c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]),-half,half)
                        if hit[0]<1.0:clear=0
                previous=current
            if clear!=0:
                starts[i]=start;ends[i]=end;elapsed[i]=0.0
                durations[i]=wp.clamp(wp.length(target-p)/wp.max(1.5,wp.length(velocities[i])),.7,1.4)
                status[i]=1
            else:
                status[i]=3;cooldown[i]=.25
    if status[i]==1:
        elapsed[i]=wp.min(durations[i],elapsed[i]+dt)
        value=arc_position(starts[i],ends[i],elapsed[i]/durations[i],.8)
        flight[i]=1;goal[i]=wp.vec2(value[0],value[1]);altitude[i]=value[2]


@wp.kernel
def jump_envelope(flight: wp.array(dtype=wp.int32), altitude: wp.array(dtype=float),
                  bounds: wp.array(dtype=wp.vec2)):
    i=wp.tid()
    if flight[i]!=0:
        height=bounds[i][1]-bounds[i][0]
        bounds[i]=wp.vec2(altitude[i],altitude[i]+height)


@wp.kernel
def finish_jump(positions: wp.array(dtype=wp.vec2), velocities: wp.array(dtype=wp.vec2),
                flight: wp.array(dtype=wp.int32), altitude: wp.array(dtype=float), ground: wp.array(dtype=float),
                base: wp.array(dtype=float), ends: wp.array(dtype=wp.vec3), elapsed: wp.array(dtype=float),
                durations: wp.array(dtype=float), cooldown: wp.array(dtype=float), status: wp.array(dtype=wp.int32)):
    i=wp.tid()
    if flight[i]!=0 and elapsed[i]>=durations[i]:
        target=wp.vec2(ends[i][0],ends[i][1])
        status[i]=2
        if wp.length(positions[i]-target)>.1:status[i]=3
        flight[i]=0;altitude[i]=ground[i]+base[i];cooldown[i]=1.0
        velocities[i]=wp.vec2(0.0,0.0)
