"""Planar swept circle contacts, with inelastic impulses and world isolation.

This remains a coarse chassis footprint, not the native compound hull. One CUDA
thread owns each world's small contact island, so pair updates cannot race.
"""
import warp as wp
from .motion import segment_box
from .terrain import TerrainData, surface_height


@wp.func
def inverse_mass(enabled: int, mass: float):
    result=float(0.0)
    if enabled!=0:result=1.0/wp.max(mass,1.e-6)
    return result


@wp.func
def safe_recovery(p: wp.vec2, shift: wp.vec2, radius: float, body: wp.vec2,
                  boxes: wp.array(dtype=wp.vec4), angles: wp.array(dtype=float),
                  wall_z: wp.array(dtype=wp.vec2), masks: wp.array2d(dtype=wp.int32),
                  agent: int, scene: TerrainData, terrain_enabled: int, step: float):
    fraction=float(1.0)
    for j in range(boxes.shape[0]):
        if masks[agent,j]!=0 and body[1]>wall_z[j][0] and body[0]<wall_z[j][1]:
            wall=boxes[j];center=wp.vec2((wall[0]+wall[2])*.5,(wall[1]+wall[3])*.5)
            half=wp.vec2((wall[2]-wall[0])*.5+radius,(wall[3]-wall[1])*.5+radius)
            c=wp.cos(angles[j]);s=wp.sin(angles[j]);relative=p-center
            hit=segment_box(wp.vec2(c*relative[0]+s*relative[1],-s*relative[0]+c*relative[1]),
                wp.vec2(c*shift[0]+s*shift[1],-s*shift[0]+c*shift[1]),-half,half)
            fraction=wp.min(fraction,wp.max(0.0,hit[0]-1.e-5))
    end=p+shift*fraction
    if terrain_enabled!=0:
        accepted=p;old_height=surface_height(scene,p);blocked=int(0)
        for k in range(1,5):
            sample=p+(end-p)*(float(k)/4.0);height=surface_height(scene,sample)
            if sample[0]<radius or sample[0]>scene.width-radius or sample[1]<radius or sample[1]>scene.height-radius or wp.abs(height-old_height)>step:blocked=1
            if blocked==0:accepted=sample;old_height=height
        end=accepted
    return end


@wp.kernel
def resolve(previous: wp.array(dtype=wp.vec2), proposed: wp.array(dtype=wp.vec2),
            velocity: wp.array(dtype=wp.vec2), limits: wp.array(dtype=wp.vec3),
            worlds: wp.array(dtype=wp.int32), body_z: wp.array(dtype=wp.vec2),
            enabled: wp.array(dtype=wp.int32), flight: wp.array(dtype=wp.int32), drive: wp.array(dtype=wp.vec3),
            boxes: wp.array(dtype=wp.vec4), angles: wp.array(dtype=float), wall_z: wp.array(dtype=wp.vec2),
            masks: wp.array2d(dtype=wp.int32), scene: TerrainData, terrain_enabled: int,
            steps: wp.array(dtype=float), result: wp.array(dtype=wp.vec2),
            result_velocity: wp.array(dtype=wp.vec2), moves: wp.array(dtype=wp.vec2),
            touched: wp.array(dtype=wp.int32), exhausted: wp.array(dtype=wp.int32)):
    # Exhaustion flags are sticky across samples/frames until explicitly cleared.
    owner=wp.tid();world=worlds[owner];first=int(1);count=previous.shape[0]
    for i in range(owner):
        if worlds[i]==world:first=0
    if first!=0:
        # Each owner writes only its world. Other worlds may occupy identical
        # coordinates and use unrelated physical masses or contact histories.
        for i in range(count):
            if worlds[i]==world:
                result[i]=previous[i];result_velocity[i]=velocity[i]
                moves[i]=proposed[i]-previous[i];touched[i]=0
        # Bounded recovery for externally supplied/spawned overlaps. Recovery
        # changes positions, never injects kinetic velocity, and cannot cross a wall.
        for i in range(count):
            if worlds[i]==world and enabled[i]!=0 and flight[i]==0:
                correction=wp.vec2(0.0,0.0)
                for j in range(count):
                    if i!=j and worlds[j]==world and flight[j]==0 and body_z[i][1]>body_z[j][0] and body_z[j][1]>body_z[i][0]:
                        separation=previous[i]-previous[j];distance=wp.length(separation)
                        radius=limits[i][2]+limits[j][2]
                        if distance<radius-.00001:
                            normal=wp.vec2(1.0,0.0)
                            if i<j:normal=wp.vec2(-1.0,0.0)
                            if distance>1.e-6:normal=separation/distance
                            own=inverse_mass(enabled[i],drive[i][0]);other=inverse_mass(enabled[j],drive[j][0])
                            correction=correction+normal*(radius-distance+.00001)*own/(own+other)
                if wp.length(correction)>1.e-6:
                    correction=correction*wp.min(1.0,.02/wp.length(correction))
                    result[i]=safe_recovery(previous[i],correction,limits[i][2],body_z[i],boxes,angles,wall_z,masks,i,scene,terrain_enabled,steps[i])
                    touched[i]=1
        remaining=float(1.0);events=int(0)
        # At most 4096 contact events per 120 Hz sample. Exhaustion is visible
        # and stops the unconsumed movement rather than allowing tunnelling.
        for iteration in range(4096):
            if remaining<=1.e-7:break
            earliest=remaining;left=int(-1);right=int(-1);normal=wp.vec2(0.0,0.0);severity=float(0.0)
            for i in range(count):
                if worlds[i]==world:
                    for j in range(i+1,count):
                        if worlds[j]==world and body_z[i][1]>body_z[j][0] and body_z[j][1]>body_z[i][0]:
                            separation=result[i]-result[j];relative=moves[i]-moves[j]
                            a=wp.dot(relative,relative);b=wp.dot(separation,relative)
                            radius=limits[i][2]+limits[j][2];c=wp.dot(separation,separation)-radius*radius
                            if b < -1.e-9 and a>1.e-16:
                                time=float(0.0);valid=int(1)
                                # A 0.1 mm contact skin keeps an already touching
                                # island active despite float32 gap roundoff.
                                if c>2.0*radius*.0001+.00000001:
                                    discriminant=b*b-a*c
                                    if discriminant<0.0:valid=0
                                    else:time=(-b-wp.sqrt(discriminant))/a
                                if valid!=0 and time>=0.0 and time<=remaining and time<=earliest+1.e-7:
                                    delta=separation+relative*time
                                    direction=delta/wp.max(wp.length(delta),1.e-9)
                                    mass=inverse_mass(enabled[i],drive[i][0])+inverse_mass(enabled[j],drive[j][0])
                                    closing=-wp.dot(relative,direction)
                                    if mass>0.0 and closing>1.e-6 and (time<earliest-1.e-7 or closing>severity):
                                        earliest=time;left=i;right=j;normal=direction;severity=closing
                    # Original free proposals already respect static geometry.
                    # Only collision-modified trajectories need a second sweep.
                    if touched[i]!=0 and wp.length(moves[i])>1.e-10:
                        for j in range(boxes.shape[0]):
                            if masks[i,j]!=0 and body_z[i][1]>wall_z[j][0] and body_z[i][0]<wall_z[j][1]:
                                wall=boxes[j];center=wp.vec2((wall[0]+wall[2])*.5,(wall[1]+wall[3])*.5)
                                half=wp.vec2((wall[2]-wall[0])*.5+limits[i][2],(wall[3]-wall[1])*.5+limits[i][2])
                                c=wp.cos(angles[j]);s=wp.sin(angles[j]);p=result[i]-center;delta=moves[i]*remaining
                                hit=segment_box(wp.vec2(c*p[0]+s*p[1],-s*p[0]+c*p[1]),wp.vec2(c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]),-half,half)
                                direction=wp.vec2(c*hit[1]-s*hit[2],s*hit[1]+c*hit[2])
                                if hit[0]<1.0 and hit[0]*remaining<=earliest and wp.dot(moves[i],direction)<-1.e-9:
                                    earliest=wp.max(0.0,hit[0]*remaining-1.e-6);left=i;right=-1;normal=direction;severity=-wp.dot(moves[i],direction)
                        if terrain_enabled!=0 and flight[i]==0:
                            old_height=surface_height(scene,result[i]);blocked=int(0)
                            for k in range(1,13):
                                sample=result[i]+moves[i]*(remaining*float(k)/12.0);height=surface_height(scene,sample)
                                if blocked==0 and (sample[0]<limits[i][2] or sample[0]>scene.width-limits[i][2] or sample[1]<limits[i][2] or sample[1]>scene.height-limits[i][2] or wp.abs(height-old_height)>steps[i]):
                                    time=remaining*float(k-1)/12.0
                                    if time<=earliest:earliest=time;left=i;right=-1;normal=wp.vec2(0.0,0.0)
                                    blocked=1
                                old_height=height
            if left<0:
                for i in range(count):
                    if worlds[i]==world:
                        if events==0 and touched[i]==0:result[i]=proposed[i]
                        else:result[i]=result[i]+moves[i]*remaining
                remaining=0.0
            else:
                for i in range(count):
                    if worlds[i]==world:result[i]=result[i]+moves[i]*earliest
                remaining=remaining-earliest;events=events+1;touched[left]=1
                if right>=0:
                    touched[right]=1
                    own=inverse_mass(enabled[left],drive[left][0]);other=inverse_mass(enabled[right],drive[right][0]);total=own+other
                    impulse=-wp.min(0.0,wp.dot(moves[left]-moves[right],normal))/total
                    moves[left]=moves[left]+normal*(impulse*own);moves[right]=moves[right]-normal*(impulse*other)
                    impulse=-wp.min(0.0,wp.dot(result_velocity[left]-result_velocity[right],normal))/total
                    result_velocity[left]=result_velocity[left]+normal*(impulse*own);result_velocity[right]=result_velocity[right]-normal*(impulse*other)
                elif wp.length(normal)>0.0:
                    moves[left]=moves[left]-normal*wp.min(0.0,wp.dot(moves[left],normal))
                    result_velocity[left]=result_velocity[left]-normal*wp.min(0.0,wp.dot(result_velocity[left],normal))
                else:
                    moves[left]=wp.vec2(0.0,0.0);result_velocity[left]=wp.vec2(0.0,0.0)
        if remaining>1.e-7:
            for i in range(count):
                if worlds[i]==world:exhausted[i]=1;result_velocity[i]=wp.vec2(0.0,0.0)
