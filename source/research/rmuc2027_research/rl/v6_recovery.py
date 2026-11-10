"""Bounded recovery curricula produced by ordinary actions from the real spawn.

This module never assigns a robot/match field. Prefix controllers and their
navigation states exist only during preparation; the original frozen policies
are restored before the first learner action. Private frame fields are used
only to verify preparation and report its cost, never to construct the actor's
legal recovery memory. A failed prefix is not a fabricated training sample.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import time

from rmuc2027_core import sdk
from rmuc2027_core.policy import EncodedObservation
from . import recovery_belief, sentry_v6


@dataclass(frozen=True)
class RecoveryPrefixSpec:
    kind: str
    max_simulation_seconds: float
    goal_red: tuple
    confirmation_during_prefix: bool = False
    spend_before_departure: bool = False


SPECS = {
    'free_ready': RecoveryPrefixSpec('free_ready', 60., (11.7, 8.5)),
    'paid_beneficial': RecoveryPrefixSpec('paid_beneficial', 45., (11.7, 8.5)),
    'unavailable': RecoveryPrefixSpec('unavailable', 45., (11.7, 8.5), spend_before_departure=True),
    'alive_weak': RecoveryPrefixSpec('alive_weak', 60., (11.7, 8.5), confirmation_during_prefix=True),
}
ALIASES = {'free': 'free_ready', 'paid': 'paid_beneficial', 'paid_early': 'paid_beneficial', 'neither_ready': 'unavailable',
           'no_paid': 'unavailable', 'recover': 'alive_weak'}


class RecoveryPreparationError(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__('Recovery prefix not prepared: '+str(report.get('reason')))


def _robot(obs, team, slot):
    return next(r for r in obs['robots'] if r['team'] == team and r['slot'] == slot)


def _mirror(point, team):
    return tuple(point) if team == 'red' else (28-point[0], 15-point[1])


def _noop():
    return dict(move=[0., 0.], brake=True, fire=False)


def _legal_summary(obs):
    """Hash only what the learner actually received, never private validation."""
    me = _robot(obs, obs['team'], 7)
    return dict(time=obs['time'], robot={k: me.get(k) for k in
        ('hp', 'max_hp', 'power_outputs', 'rfid', 'x', 'y', 'z', 'alive')},
        referee=obs.get('referee', {}), command_feedback=obs.get('command_feedback', {}))


class _Driver:
    def __init__(self, session, spec):
        self.session, self.spec = session, spec
        self.team = session.config['team']
        self.other = 'blue' if self.team == 'red' else 'red'
        self.goal = _mirror(spec.goal_red, self.team)
        self.state = session.controller.state
        self.dead_seen = None
        self.phase = 'spend' if spec.spend_before_departure else 'travel'
        self.engage = False
        self.spent = False
        self.purchase_requested = None
        self.purchase_cost = 0.
        self.history_hash = hashlib.sha256()
        self.history_count = 0
        self.last_history_tick = None
        self.events = []
        self.event_counts = {}
        self.initial_position = None

    def receive(self, obs):
        # Called by the main thread between steps and by only the learner-seat
        # controller during a step. No privileged auxiliary observation enters.
        sentry_v6.observe(obs, self.state)
        if obs['tick'] != self.last_history_tick:
            self.history_hash.update(json.dumps(_legal_summary(obs), sort_keys=True,
                separators=(',', ':'), allow_nan=False).encode()+b'\n')
            self.history_count += 1
            self.last_history_tick = obs['tick']

    def word(self, obs, moving):
        self.receive(obs)
        me = _robot(obs, self.team, 7)
        ack = obs['referee'].get('ack', {})
        fresh = recovery_belief.fresh_packet(obs, '0x020D')
        counters = self.state.setdefault('referee_commands',
            dict(local_ammo=0, remote_ammo=0, remote_heal=0))
        if fresh:
            for key in ('local_ammo', 'remote_ammo', 'remote_heal'):
                counters[key] = max(counters[key], int(ack.get(key, 0)))
        # Spend only via an ordinary local 17 mm order, in legal batches. We
        # wait for the actual counter ACK/coin deduction before departing.
        if self.phase == 'spend' and self.purchase_requested is None and fresh:
            coins = obs['referee'].get('coins')
            cards = me.get('rfid', [0]*41)
            local = recovery_belief.fresh_packet(obs, '0x0209') and any(
                len(cards) > i and cards[i] for i in (0, 2 if self.team == 'red' else 3, 10))
            if local and me.get('alive') and isinstance(coins, (float, int)) and coins >= 100:
                # Public 2027 contract: 10 rounds cost 10 coins; at most 1000
                # team rounds. No account is edited and no purchase is assumed.
                amount = min(int(coins)//10*10, int(ack.get('ammo_quota', 0))//10*10,
                             (2047-counters['local_ammo'])//10*10)
                if amount:
                    counters['local_ammo'] += amount
                    self.purchase_requested = counters['local_ammo']
        free = bool(self.spec.confirmation_during_prefix and sentry_v6.revive_mask(obs)[1])
        word = (int(free) | counters['local_ammo'] << 2 | counters['remote_ammo'] << 13
                | counters['remote_heal'] << 17 | (3 if moving else 1) << 21)
        sentry_v6.record_tx(self.state, obs['time'], word)
        return word

    def update(self, obs, frame):
        self.receive(obs)
        me = _robot(obs, self.team, 7)
        if self.initial_position is None: self.initial_position = (me['x'], me['y'])
        hp_valid = recovery_belief.fresh_packet(obs, '0x0201')
        dead = hp_valid and me['hp'] == 0
        ack = obs['referee'].get('ack', {})
        if self.phase == 'spend' and self.purchase_requested is not None:
            if (recovery_belief.fresh_packet(obs, '0x020D')
                    and ack.get('local_ammo', -1) >= self.purchase_requested):
                coins = obs['referee'].get('coins')
                cost = ack.get('paid_cost', 100)
                if isinstance(coins, (int, float)) and coins < max(100, cost):
                    self.spent = True; self.phase = 'travel'
        if dead and self.dead_seen is None:
            self.dead_seen = obs['referee']['sources']['0x0201']['source_time']
        if dead: self.phase = 'dead'; self.engage = False
        elif self.dead_seen is not None: self.phase = 'revived'; self.engage = False
        elif self.phase != 'spend':
            self.engage = math.dist((me['x'], me['y']), self.goal) <= .7
            self.phase = 'exposed' if self.engage else 'travel'
        for event in frame.get('events', []):
            kind = event.get('type', event.get('kind', 'unknown'))
            self.event_counts[kind] = self.event_counts.get(kind, 0)+1
            if kind in ('destroy', 'respawn', 'ammo_purchase', 'invalid_action', 'sentry_command_rejected'):
                self.events.append(dict(event))
            if kind == 'ammo_purchase' and event.get('robot') == self.team+'-7':
                self.purchase_cost += float(event.get('cost', 0))

    def prepared(self, obs, frame):
        me = _robot(obs, self.team, 7)
        truth = _robot(frame, self.team, 7)
        hp_fresh = recovery_belief.fresh_packet(obs, '0x0201')
        ack = obs['referee'].get('ack', {})
        packet = obs['referee'].get('sources', {}).get('0x020D', {})
        # A still-valid pre-death ACK says nothing about this life's options.
        ack_current = (self.dead_seen is not None and recovery_belief.fresh_packet(obs, '0x020D')
                       and packet.get('source_time', -1) >= self.dead_seen)
        dead = hp_fresh and me['hp'] == 0 and not truth['alive']
        if self.spec.kind == 'free_ready':
            return bool(dead and ack_current and ack.get('free_revive') and ack.get('paid_revive'))
        if self.spec.kind == 'paid_beneficial':
            remaining = float(truth.get('respawn_required', 0))-float(truth.get('respawn_progress', 0))
            return bool(dead and ack_current and ack.get('paid_revive') and not ack.get('free_revive') and remaining >= 8.)
        if self.spec.kind == 'unavailable':
            return bool(self.spent and dead and ack_current and not ack.get('paid_revive') and not ack.get('free_revive'))
        return bool(self.dead_seen is not None and hp_fresh and me['hp'] > 0
                    and truth['alive'] and truth.get('weak') and self.state['recovery_belief']['revive_types'] == ['normal'])


class _PrefixController:
    control_hz = 20
    disabled = False
    is_learning_controller = False
    consecutive_timeouts = 0

    def __init__(self, driver, team, slot, profile):
        self.driver, self.team, self.slot = driver, team, slot
        self.observation_profile = profile
        self.navigation, self.logs = {}, []
        self.fire_commands = 0
        self.ammo_order_time = None
        self.guard_counts = {}
        self.navigation_phase_counts = {}
        self.last_diagnostic = {}
        self.diagnostic_samples = []
        self.last_diagnostic_second = None

    def _count(self, key):
        self.guard_counts[key] = self.guard_counts.get(key, 0)+1

    def _diagnose(self, obs, robot, action, target=None, goal=None):
        phase = self.navigation.get('phase', 'idle')
        self.navigation_phase_counts[phase] = self.navigation_phase_counts.get(phase, 0)+1
        fields = ('id', 'team', 'slot', 'model_slot', 'x', 'y', 'z', 'vx', 'vy',
                  'hp', 'alive', 'tipped', 'grounded', 'transform_progress',
                  'ammo', 'heat', 'max_heat', 'weak', 'overheated',
                  'heat_permanent_lock', 'overspeed_locked', 'overspeed_permanent_lock',
                  'launcher_fault', 'red_card')
        self.last_diagnostic = dict(time=obs['time'], driver_phase=self.driver.phase,
            driver_engage=self.driver.engage, robot={k: robot.get(k) for k in fields},
            target=None if target is None else {k: target.get(k) for k in fields},
            goal=None if goal is None else list(goal),
            range=None if target is None else sdk.distance(robot, target),
            navigation_phase=phase, navigation_rejected=list(self.navigation.get('rejected', [])),
            trajectory=list(self.navigation.get('trajectory', [])),
            waiting_peer=self.navigation.get('wait_for_peer'),
            move=list(action.get('move', (0., 0.))), fire=bool(action.get('fire', False)))
        second = int(obs['time'])
        if second != self.last_diagnostic_second:
            self.diagnostic_samples.append(dict(self.last_diagnostic))
            self.last_diagnostic_second = second

    def decide(self, observation):
        obs = json.loads(observation.payload) if isinstance(observation, EncodedObservation) else observation
        robot = _robot(obs, self.team, self.slot)
        action = _noop()
        driver = self.driver
        target, goal = None, None
        if self.team == driver.team and self.slot == 7:
            if robot.get('alive') and driver.phase not in ('spend', 'dead', 'revived'):
                goal = driver.goal
                action = sdk.navigate_action(robot, goal, obs, self.navigation)
                action['fire'] = False
            action['sentry_cmd'] = driver.word(obs, math.hypot(*action.get('move', (0., 0.))[:2]) > .05)
        elif self.team == driver.other and self.slot in (3, 4):
            # Auxiliary actors are permitted full vision. The learner gets none
            # of the shooter's observations or navigation/controller state.
            target = next((r for r in obs.get('enemies', []) if r['slot'] == 7 and r['team'] == driver.team), None)
            # Both staging goals are on the public central deck. In the
            # shooter's team-local frame they trigger SDK's existing legal
            # fold/jump approach, avoiding the v1 40+ metre outer-map detour.
            goal_red = (16.3, 6.5 if self.slot == 3 else 8.5)
            if driver.phase in ('dead', 'revived'):
                goal_red = (16.8, 6.5 if self.slot == 3 else 8.5)
            goal = _mirror(goal_red, driver.team)
            self._count('decisions')
            if target is not None: self._count('target_seen')
            if robot.get('alive'):
                self._count('shooter_alive')
                # Infantry starts with zero ammunition. Make one ordinary
                # remote purchase, then wait for observed delivery before
                # leaving spawn. Neither ammunition nor team coins are edited.
                # Repeated 20 Hz commands must not purchase another batch while
                # the first order is pending (normal delivery takes 6 seconds).
                if robot.get('ammo', 0) <= 0:
                    if self.ammo_order_time is None:
                        action['buy_ammo'] = dict(remote=True, amount=100)
                        self.ammo_order_time = float(obs['time'])
                        self._count('ammo_order_sent')
                    self._count('awaiting_actual_ammo')
                    self._diagnose(obs, robot, action, target, goal)
                    return {str(self.slot): action}, None
                action = sdk.navigate_action(robot, goal, obs, self.navigation)
                action['fire'] = False
                if target is not None and target.get('alive'):
                    self._count('target_alive')
                    # Preparation creates a real firefight; it does not wait
                    # for the learner to park at an exact waypoint. Actual
                    # death/readiness is still verified before handoff.
                    if sdk.distance(robot, target) <= 7.:
                        self._count('in_range')
                        can_fire = bool(sdk.can_fire(robot))
                        reachable = bool(sdk.ballistic_reachable(robot, target))
                        self._count('can_fire' if can_fire else 'cannot_fire')
                        self._count('ballistic_reachable' if reachable else 'ballistic_blocked')
                        action.update(aim=sdk.aim_at(robot, target), pitch=sdk.aim_pitch(robot, target),
                                      fire=bool(can_fire and reachable))
                        self.fire_commands += int(action['fire'])
                    else: self._count('out_of_range')
        self._diagnose(obs, robot, action, target, goal)
        return {str(self.slot): action}, None

    def close(self):
        self.disabled = True


def prepare_recovery_prefix(session, kind, deadline=None):
    """Return a JSON-safe report; normal inability/timeout returns prepared=False.

    Deadline is absolute ``time.monotonic()``. Every Battle.step is real work
    inside this deadline; nothing here calls a policy optimizer or session.step.
    A caller must discard/retry a not_prepared episode, never count it as PPO.
    """
    kind = ALIASES.get(kind, kind)
    if kind not in SPECS: raise ValueError('Unknown recovery prefix kind: '+str(kind))
    spec, battle = SPECS[kind], session.battle
    if session.config.get('slot') != 7 or session.config.get('policy_version') != 6:
        raise ValueError('Recovery prefix requires the v6 main sentry')
    start, wall_start = float(battle.match.time), time.monotonic()
    if start > .1: raise ValueError('Recovery prefixes must begin at the real spawn')
    original_policies, original_seats = battle.policies, battle._seat_policies
    if any(slots is None or len(slots) != 1 for _, slots, _ in original_policies):
        raise ValueError('Recovery prefixes require isolated single-seat controllers')
    driver = _Driver(session, spec)
    replacements = []
    for team, slots, policy in original_policies:
        slot = slots[0]
        profile = ('privileged_aux_2027_v1' if team == driver.other and slot in (3, 4)
                   else getattr(policy, 'observation_profile', None))
        replacements.append((team, slots, _PrefixController(driver, team, slot, profile)))
    error_count = len(battle.diagnostics)
    report = dict(version=3, kind=kind, prepared=False, reason='simulation_limit',
        max_simulation_seconds=spec.max_simulation_seconds, learning_samples=0,
        bootstrap_excluded_from_policy_score=True, start_time=start,
        preparation_controller='legal_script_prefix_v3')
    try:
        battle.policies = replacements
        battle._seat_policies = {(team, slots[0]): p for team, slots, p in replacements}
        battle.held_commands = {'red': {}, 'blue': {}}
        obs = battle.observation(driver.team, [7])
        frame = battle.match._frame
        driver.update(obs, frame)
        while float(battle.match.time)-start < spec.max_simulation_seconds-1e-8:
            if deadline is not None and time.monotonic() >= deadline or battle.cancelled():
                report['reason'] = 'wall_deadline'; break
            if battle.match.status == 'finished':
                report['reason'] = 'match_finished'; break
            if driver.prepared(obs, frame):
                report.update(prepared=True, reason='prepared'); break
            frame = battle.step()
            obs = battle.observation(driver.team, [7])
            driver.update(obs, frame)
            if len(battle.diagnostics) > error_count:
                report['reason'] = 'prefix_policy_error'; break
        else:
            if driver.prepared(obs, frame): report.update(prepared=True, reason='prepared')
        session.obs = obs
        truth = _robot(frame, driver.team, 7)
        report['handoff_legal'] = dict(hp=_robot(obs, driver.team, 7)['hp'],
            revive_mask=sentry_v6.revive_mask(obs), ack=obs.get('referee', {}).get('ack', {}))
        # Explicitly private audit evidence; never passed to observe/encode.
        report['validation_ground_truth'] = {key: truth.get(key) for key in
            ('alive', 'weak', 'respawn_progress', 'respawn_required', 'respawned_at')}
    finally:
        battle.policies, battle._seat_policies = original_policies, original_seats
        battle.held_commands = {'red': {}, 'blue': {}}
        for _, _, policy in replacements:
            battle.navigation_observations.pop(id(policy), None)
        # Preserve received history and idempotent purchase counters, discard
        # every prefix navigation/goal/intent latch before RL gets control.
        for key in list(driver.state):
            if key not in ('recovery_belief', 'referee_commands', 'next_exchange'):
                driver.state.pop(key)
        report.update(simulation_seconds=float(battle.match.time)-start,
            end_time=float(battle.match.time), wall_seconds=time.monotonic()-wall_start,
            legal_history_count=driver.history_count, legal_history_sha256=driver.history_hash.hexdigest(),
            events=driver.events, event_counts=driver.event_counts, purchase_cost=driver.purchase_cost,
            purchase_requested=driver.purchase_requested, purchase_acknowledged=driver.spent,
            fire_commands=sum(p.fire_commands for _, _, p in replacements),
            diagnostics=dict(driver_phase=driver.phase, driver_engage=driver.engage,
                learner_goal=list(driver.goal), dead_seen=driver.dead_seen,
                seats={team+':'+str(slots[0]): dict(guard_counts=p.guard_counts,
                    ammo_order_time=p.ammo_order_time,
                    navigation_phase_counts=p.navigation_phase_counts,
                    last=p.last_diagnostic, samples=p.diagnostic_samples)
                    for team, slots, p in replacements}))
        battle.metadata['recovery_prefix'] = report
    return report
