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

    def reset(self):
        self.target_id = None
        self.state = 'idle'
        self.raw = self.manual = None
        self.tracking = False
        self.reanchor = True

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
            return self.manual

        optical = observation.get('vision', {}).get(str(robot['slot']), {})
        camera = optical.get('camera')
        visible = set(optical.get('enemies', []))
        candidates = []
        if camera:
            direction = camera['direction']
            direction_length = math.sqrt(sum(v*v for v in direction))
            for target in observation.get('enemies', []):
                if (target['id'] not in visible or target.get('team') == robot['team']
                        or target.get('role') == 'aerial' or not target.get('alive')
                        or not target.get('targetable', True) or target.get('invulnerable')):
                    continue
                offset = (target['x'] - camera['origin'][0], target['y'] - camera['origin'][1],
                          sdk.aim_height(target) - camera['origin'][2])
                length = math.sqrt(sum(v*v for v in offset)) * direction_length
                if length < 1e-9:
                    continue
                angle = math.acos(max(-1., min(1., sum(a*b for a, b in zip(offset, direction))/length)))
                retained = target['id'] == self.target_id
                if angle <= math.radians(20 if retained else 15):
                    candidates.append((not retained, angle, target['id'], target))
        for target in sorted(candidates):
            enemy = sdk.intercept(robot, target[3])
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
