"""Planar swept circle contacts, with inelastic impulses and world isolation.

This remains a coarse chassis footprint, not the native compound hull. One CUDA
thread owns each world's small contact island, so pair updates cannot race.
"""
import warp as wp
from .motion import segment_box
from .terrain import TerrainData, surface_height, smooth_sweep


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
            members: wp.array2d(dtype=wp.int32), sizes: wp.array(dtype=wp.int32), body_z: wp.array(dtype=wp.vec2),
            enabled: wp.array(dtype=wp.int32), flight: wp.array(dtype=wp.int32), drive: wp.array(dtype=wp.vec3),
            boxes: wp.array(dtype=wp.vec4), angles: wp.array(dtype=float), wall_z: wp.array(dtype=wp.vec2),
            masks: wp.array2d(dtype=wp.int32), scene: TerrainData, terrain_enabled: int,
            steps: wp.array(dtype=float), result: wp.array(dtype=wp.vec2),
            result_velocity: wp.array(dtype=wp.vec2), moves: wp.array(dtype=wp.vec2),
            touched: wp.array(dtype=wp.int32), exhausted: wp.array(dtype=wp.int32)):
    # Static member lists preserve global index order, including interleaved worlds.
    # Exhaustion flags remain sticky until explicitly cleared.
    owner = wp.tid()
    size = sizes[owner]
    for member_i in range(size):
        i = members[owner, member_i]
        result[i] = previous[i]
        result_velocity[i] = velocity[i]
        moves[i] = proposed[i] - previous[i]
        touched[i] = 0
    # Recovery is bounded and injects no kinetic velocity.
    for member_i in range(size):
        i = members[owner, member_i]
        if enabled[i] != 0 and flight[i] == 0:
            correction = wp.vec2(0.0, 0.0)
            for member_j in range(size):
                j = members[owner, member_j]
                if i != j and flight[j] == 0 and (body_z[i][1] > body_z[j][0]) and (body_z[j][1] > body_z[i][0]):
                    separation = previous[i] - previous[j]
                    distance = wp.length(separation)
                    radius = limits[i][2] + limits[j][2]
                    if distance < radius - 1e-05:
                        normal = wp.vec2(1.0, 0.0)
                        if i < j:
                            normal = wp.vec2(-1.0, 0.0)
                        if distance > 1e-06:
                            normal = separation / distance
                        own = inverse_mass(enabled[i], drive[i][0])
                        other = inverse_mass(enabled[j], drive[j][0])
                        correction = correction + normal * (radius - distance + 1e-05) * own / (own + other)
            if wp.length(correction) > 1e-06:
                correction = correction * wp.min(1.0, 0.02 / wp.length(correction))
                result[i] = safe_recovery(previous[i], correction, limits[i][2], body_z[i], boxes, angles, wall_z, masks, i, scene, terrain_enabled, steps[i])
                touched[i] = 1
    remaining = float(1.0)
    events = int(0)
    # Preserve the event budget, contact skin, tie ordering and arithmetic.
    for iteration in range(4096):
        if remaining <= 1e-07:
            break
        earliest = remaining
        left = int(-1)
        right = int(-1)
        normal = wp.vec2(0.0, 0.0)
        severity = float(0.0)
        for member_i in range(size):
            i = members[owner, member_i]
            for member_j in range(member_i + 1, size):
                j = members[owner, member_j]
                if body_z[i][1] > body_z[j][0] and body_z[j][1] > body_z[i][0]:
                    separation = result[i] - result[j]
                    relative = moves[i] - moves[j]
                    a = wp.dot(relative, relative)
                    b = wp.dot(separation, relative)
                    radius = limits[i][2] + limits[j][2]
                    c = wp.dot(separation, separation) - radius * radius
                    if b < -1e-09 and a > 1e-16:
                        time = float(0.0)
                        valid = int(1)
                        # The original 0.1 mm skin avoids roundoff contact collapse.
                        if c > 2.0 * radius * 0.0001 + 1e-08:
                            discriminant = b * b - a * c
                            if discriminant < 0.0:
                                valid = 0
                            else:
                                time = (-b - wp.sqrt(discriminant)) / a
                        if valid != 0 and time >= 0.0 and (time <= remaining) and (time <= earliest + 1e-07):
                            delta = separation + relative * time
                            direction = delta / wp.max(wp.length(delta), 1e-09)
                            mass = inverse_mass(enabled[i], drive[i][0]) + inverse_mass(enabled[j], drive[j][0])
                            closing = -wp.dot(relative, direction)
                            if mass > 0.0 and closing > 1e-06 and (time < earliest - 1e-07 or closing > severity):
                                earliest = time
                                left = i
                                right = j
                                normal = direction
                                severity = closing
            if touched[i] != 0 and wp.length(moves[i]) > 1e-10:
                for j in range(boxes.shape[0]):
                    if masks[i, j] != 0 and body_z[i][1] > wall_z[j][0] and (body_z[i][0] < wall_z[j][1]):
                        wall = boxes[j]
                        center = wp.vec2((wall[0] + wall[2]) * 0.5, (wall[1] + wall[3]) * 0.5)
                        half = wp.vec2((wall[2] - wall[0]) * 0.5 + limits[i][2], (wall[3] - wall[1]) * 0.5 + limits[i][2])
                        c = wp.cos(angles[j])
                        s = wp.sin(angles[j])
                        p = result[i] - center
                        delta = moves[i] * remaining
                        hit = segment_box(wp.vec2(c * p[0] + s * p[1], -s * p[0] + c * p[1]), wp.vec2(c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]), -half, half)
                        direction = wp.vec2(c * hit[1] - s * hit[2], s * hit[1] + c * hit[2])
                        if hit[0] < 1.0 and hit[0] * remaining <= earliest and (wp.dot(moves[i], direction) < -1e-09):
                            earliest = wp.max(0.0, hit[0] * remaining - 1e-06)
                            left = i
                            right = -1
                            normal = direction
                            severity = -wp.dot(moves[i], direction)
                if terrain_enabled != 0 and flight[i] == 0 and smooth_sweep(scene,result[i],moves[i]*remaining,limits[i][2],steps[i])==0:
                    old_height = surface_height(scene, result[i])
                    blocked = int(0)
                    for k in range(1, 13):
                        sample = result[i] + moves[i] * (remaining * float(k) / 12.0)
                        height = surface_height(scene, sample)
                        if blocked == 0 and (sample[0] < limits[i][2] or sample[0] > scene.width - limits[i][2] or sample[1] < limits[i][2] or (sample[1] > scene.height - limits[i][2]) or (wp.abs(height - old_height) > steps[i])):
                            time = remaining * float(k - 1) / 12.0
                            if time <= earliest:
                                earliest = time
                                left = i
                                right = -1
                                normal = wp.vec2(0.0, 0.0)
                            blocked = 1
                        old_height = height
        if left < 0:
            for member_i in range(size):
                i = members[owner, member_i]
                if events == 0 and touched[i] == 0:
                    result[i] = proposed[i]
                else:
                    result[i] = result[i] + moves[i] * remaining
            remaining = 0.0
        else:
            for member_i in range(size):
                i = members[owner, member_i]
                result[i] = result[i] + moves[i] * earliest
            remaining = remaining - earliest
            events = events + 1
            touched[left] = 1
            if right >= 0:
                touched[right] = 1
                own = inverse_mass(enabled[left], drive[left][0])
                other = inverse_mass(enabled[right], drive[right][0])
                total = own + other
                impulse = -wp.min(0.0, wp.dot(moves[left] - moves[right], normal)) / total
                moves[left] = moves[left] + normal * (impulse * own)
                moves[right] = moves[right] - normal * (impulse * other)
                impulse = -wp.min(0.0, wp.dot(result_velocity[left] - result_velocity[right], normal)) / total
                result_velocity[left] = result_velocity[left] + normal * (impulse * own)
                result_velocity[right] = result_velocity[right] - normal * (impulse * other)
            elif wp.length(normal) > 0.0:
                moves[left] = moves[left] - normal * wp.min(0.0, wp.dot(moves[left], normal))
                result_velocity[left] = result_velocity[left] - normal * wp.min(0.0, wp.dot(result_velocity[left], normal))
            else:
                moves[left] = wp.vec2(0.0, 0.0)
                result_velocity[left] = wp.vec2(0.0, 0.0)
    if remaining > 1e-07:
        for member_i in range(size):
            i = members[owner, member_i]
            exhausted[i] = 1
            result_velocity[i] = wp.vec2(0.0, 0.0)
