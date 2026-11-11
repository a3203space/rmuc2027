"""Training-only revival accounting. Never imported by the actor runtime."""
import copy
import math


class RecoveryAccounting:
    def __init__(self, robot=None):
        self.seen = set()
        self.armed = bool(robot and robot.get('alive') and robot.get('weak'))
        self.hold = 0.
        self.recoveries = 0
        self.paid_coins = 0.
        self.paid_revives = self.free_revives = 0
        self.alive_weak_seconds = self.dead_seconds = 0.
        self.death_episodes = []
        self.current = None
        if robot and not robot.get('alive', True):
            self.current = dict(prefix_death=True, choices={}, first_intent={}, paid_exposure_seconds=0., censored=True)
            self.death_episodes.append(self.current)

    def decision(self, obs, action, seconds=.1):
        from .sentry_v6 import revive_mask
        if self.current is None:
            return
        mask = revive_mask(obs)
        if not mask[1]:
            return
        choice = ('wait', 'free', 'paid')[int(action[6])]
        self.current['choices'][choice] = self.current['choices'].get(choice, 0) + 1
        self.current['first_intent'].setdefault(choice, obs['time'])
        self.current['paid_exposure_seconds'] += float(mask[2]) * seconds

    def advance(self, before, robot, frame, dt):
        result = dict(paid_coins=0., recovery=0.)
        if before['alive'] and not robot['alive']:
            self.current = dict(death_time=frame['time'], choices={}, first_intent={},
                                paid_exposure_seconds=0., censored=True)
            self.death_episodes.append(self.current)
            self.armed = False
        for event in frame.get('events', []):
            if event.get('type') != 'respawn' or event.get('robot') != robot['id']:
                continue
            key = (event.get('tick', frame.get('tick')), event.get('time', frame['time']), robot['id'])
            if key in self.seen:
                continue
            self.seen.add(key)
            paid = bool(event.get('instant'))
            cost = float(event.get('cost', 0.)) if paid else 0.
            if not math.isfinite(cost) or cost < 0:
                raise ValueError('Invalid successful revival cost')
            self.paid_coins += cost
            self.paid_revives += int(paid); self.free_revives += int(not paid)
            result['paid_coins'] -= .001 * cost
            self.armed = True; self.hold = 0.
            if self.current is not None:
                self.current.update(censored=False, revived_at=frame['time'],
                                    observed_mode='paid' if paid else 'free', cost=cost,
                                    initial_hp=event.get('hp'))
                self.current = None
        self.dead_seconds += dt * (not robot['alive'])
        self.alive_weak_seconds += dt * bool(robot['alive'] and robot.get('weak'))
        active = robot['alive'] and not robot.get('weak', False)
        self.hold = self.hold + dt if active and self.armed else 0.
        if self.armed and self.hold + 1e-9 >= 1.:
            result['recovery'] = .2
            self.recoveries += 1; self.armed = False
        return result

    def report(self):
        return dict(paid_coins=self.paid_coins, paid_revives=self.paid_revives,
                    free_revives=self.free_revives, recoveries=self.recoveries,
                    alive_weak_seconds=self.alive_weak_seconds, dead_seconds=self.dead_seconds,
                    death_episodes=copy.deepcopy(self.death_episodes))
