"""Hold-to-track assistance using only the controlled robot's optical observation.

Design reference: chenjunnn/rm_auto_aim armor_tracker/src/tracker.cpp
(center-priority acquisition, identity retention, continuous yaw). This is a
simulation-specific implementation using the existing SDK, with no copied code.
"""
import math
from rmuc2027_core import sdk


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class AimAssist:
    def __init__(self):
        self.target_id = None
        self.state = 'idle'
        self.raw = None
        self.manual = None
        self.tracking = False
        self.reanchor = False
        self.outpost_rotations = {}

    def reset(self):
        self.target_id = None
        self.state = 'idle'
        self.raw = self.manual = None
        self.tracking = False
        self.reanchor = True
        self.outpost_rotations.clear()

    def outpost_targets(self, observation, robot, optical):
        """Aim at visible armor, with rotation measured from consecutive optical samples."""
        targets = []
        samples = {}
        now = observation.get('time', 0.)
        for site, faces in optical.get('outposts', {}).items():
            post = observation.get('outposts', {}).get(site, {})
            if post.get('hp', 0) <= 0 or post.get('owner', site) == robot['team']:
                continue
            rotation = post.get('rotation', 0.)
            last = self.outpost_rotations.get(site)
            spin = 0.
            if last and 0 < now-last[0] <= .25 and not post.get('rotation_stopped'):
                spin = wrap(rotation-last[1]) / (now-last[0])
            samples[site] = (now, rotation)
            for plate in post.get('armor_plates', []):
                if plate['face'] not in faces or (plate.get('heavy_only') and not sdk.is_42(robot)):
                    continue
                x, y, z = plate['center']
                target = dict(id=f'{site}-outpost', x=x, y=y, aim_z=z,
                              kind='outpost', face=plate['face'])
                # Estimate travel time, then rotate the actual plate about the post.
                # Preserve its individual height (the middle plates are staggered).
                lead = dict(target)
                for _ in range(2):
                    pitch = sdk.aim_pitch(robot, lead)
                    flight = math.hypot(lead['x']-robot['x'], lead['y']-robot['y']) / max(.1, sdk.muzzle_speed(robot)*math.cos(pitch))
                    angle = spin*flight if plate['face'] < 3 else 0.
                    c, s = math.cos(angle), math.sin(angle)
                    dx, dy = x-post['x'], y-post['y']
                    lead['x'], lead['y'] = post['x']+c*dx-s*dy, post['y']+s*dx+c*dy
                nx, ny, nz = plate['normal']
                normal = (c*nx-s*ny, s*nx+c*ny, nz)
                toward = (robot['x']-lead['x'], robot['y']-lead['y'], robot.get('z', 0)+robot.get('muzzle_height_m', .45)-z)
                if sum(a*b for a,b in zip(normal, toward)) > 0:
                    target['prediction'] = lead
                    targets.append(target)
        self.outpost_rotations = samples
        return targets

    def angles(self, observation, robot, intent):
        if not intent.active:
            # Menu/RPC ticks use synthetic angles, not mouse samples. Never
            # let those establish the baseline for the next captured input.
            self.reset()
            return robot['aim'], robot['pitch']
        raw = (intent.yaw, intent.pitch)
        delta = (0., 0.) if self.raw is None else (wrap(raw[0] - self.raw[0]), raw[1] - self.raw[1])
        # A host-owned anchor lets release hand back the actual gimbal angle;
        # subsequent absolute client inputs contribute only their mouse delta.
        if self.manual is None:
            self.manual = (robot['aim'], robot['pitch']) if self.reanchor else raw
            self.reanchor = False
        elif self.tracking:
            self.manual = (robot['aim'] + delta[0], robot['pitch'] + delta[1])
        else:
            self.manual = (self.manual[0] + delta[0], self.manual[1] + delta[1])
        self.manual = (wrap(self.manual[0]), max(-1.5, min(1.5, self.manual[1])))
        self.raw = raw
        was_tracking = self.tracking
        self.tracking = False
        if not intent.aim_assist:
            self.target_id, self.state = None, 'idle'
            self.outpost_rotations.clear()
            return self.manual

        optical = observation.get('vision', {}).get(str(robot['slot']), {})
        camera = optical.get('camera')
        visible = set(optical.get('enemies', []))
        candidates = []
        if camera:
            direction = camera['direction']
            direction_length = math.sqrt(sum(v*v for v in direction))
            robots = [target for target in observation.get('enemies', [])
                      if target['id'] in visible and target.get('team') != robot['team']
                      and target.get('role') != 'aerial' and target.get('alive')
                      and target.get('targetable', True) and not target.get('invulnerable')]
            for target in robots + self.outpost_targets(observation, robot, optical):
                offset = (target['x'] - camera['origin'][0], target['y'] - camera['origin'][1],
                          sdk.aim_height(target) - camera['origin'][2])
                length = math.sqrt(sum(v*v for v in offset)) * direction_length
                if length < 1e-9:
                    continue
                angle = math.acos(max(-1., min(1., sum(a*b for a, b in zip(offset, direction))/length)))
                retained = target['id'] == self.target_id
                if angle <= math.radians(20 if retained else 15):
                    candidates.append((not retained, angle, target['id'], target.get('face', -1), target))
        else:
            self.outpost_rotations.clear()
        for target in sorted(candidates, key=lambda item: item[:4]):
            enemy = target[4].get('prediction') or sdk.intercept(robot, target[4])
            if sdk.ballistic_reachable(robot, enemy):
                self.target_id, self.state, self.tracking = enemy['id'], 'tracking', True
                # SDK and the physics engine apply the existing muzzle geometry
                # and gimbal limits; assistance never sets a fire command.
                return sdk.aim_at(robot, enemy), sdk.aim_pitch(robot, enemy)
        self.target_id = None
        self.state = 'lost' if was_tracking else 'searching'
        return self.manual

    def snapshot(self):
        return {'state': self.state, 'target_id': self.target_id}
