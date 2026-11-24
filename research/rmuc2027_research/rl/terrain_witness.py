"""Terrain witnesses: infer an opponent's ground ability from what it was seen doing.

The radar only ever reports coordinates (protocol p.67 `RadarInfoToClient` carries
x, y and a highlight flag per robot, nothing else), so the encoder cannot be told
what chassis an opponent has. It has to watch instead. Two terrain features are
observable this way, and the rulebook makes them worth watching:

* Tunnels. Rulebook table 5-27 splits the terrain-crossing buff in two. Crossing
  over the top (one end, the other end) grants 25% defence for 30 s; driving
  through (one end, the inside, the other end) grants 1.5x barrel cooling for
  120 s. Only a robot that folds can reach the card under the roof, so passing
  the inside card is proof of the stronger opponent. The engine agrees: a model
  without `transform_allowed` gets a collision hull at `max(muzzle_z, 0.40)`
  (physics.rs), while the near tunnel roof hangs at 0.292 m (rules.json), so it
  physically cannot get under there.
* The 200 mm terrace (rulebook 4.3.1). A robot whose height jumps by roughly
  one terrace step, twice, with no tunnel nearby, climbed it. The second
  requirement is what keeps a fold on a ramp from counting.

Every witness is scoped to what this robot could see. The filter is the one the
observation itself already applies, so a slot only earns evidence while it is in
`obs['enemies']`; a slot that has never been seen cannot be described. This is
deliberately weaker than reading `DESIGNS`: the network gets the opponent's
observed behaviour, not a lookup of what that model happens to be, and it has to
generalise to opponents it has only partly watched.

Kept separate from `portable.py` so the encoder stays a pure function of one
observation. `session.py` owns an instance and hands the snapshot in.

Evidenced beats guessed, per bit
--------------------------------
Watching is strictly better than looking the answer up, but it is also strictly
less available: an exported submission gets no witness at all, and even in
training a slot that was never seen cannot be described. Reporting "cannot do
it" for such a slot would be a lie, and it would also make the exported policy
read three different numbers than the one that was trained.

So the witness is built as an *override over a prior*. `bits()` starts from the
public design table (`portable.capability_bits`) and lets each witnessed bit
replace its prior entry. Two consequences worth stating:

* A bit with no evidence costs exactly what the design table says. That is the
  same answer an exported submission gives, so training and serving agree on
  every slot the policy has not personally watched cross a tunnel.
* Where evidence exists, the two disagree by construction - that is the entire
  point of watching - and the disagreement is confined to the bits that were
  actually witnessed. `omni` has no behavioural witness at all (a position track
  cannot show a strafe), so it always stays on the prior, in training and in
  exporting alike.

`bits(slot)` without a robot therefore yields the pure prior: that is the
exported path.
"""
import math

from .portable import capability_bits

# The three cards of every tunnel group sit 0.56-0.60 m apart along its axis and
# the middle one is the smaller disc (radius 0.17 against 0.30 for the ends), so
# the inside card is where a low profile is forced. `configs/rules.json` ids:
# {red,blue}_{near,far,road}_tunnel_{0,1,2}.
TUNNEL_INSIDE_CARD = 1

# Rulebook 4.3.1 puts the terrace 200 mm above the field. `2731`/`2714`/`2772`
# carry step_climb_m == 0.2, and 0.12 m of margin tolerates the suspension
# compressing on the ramp without letting a road bump (70 mm) through.
TERRACE_M = 0.20
CLIMB_RISE_M = 0.12

# Positional report rate is 1 Hz, so a two-sample window is 1 s. A climb's two
# rises have to land within this window to be one event rather than two bumps.
CLIMB_WINDOW_S = 3.0

# Tunnels occupy a small cluster of the map. A height change measured in that
# cluster is the tunnel's doing, not a terrace climb.
TUNNEL_RADIUS_M = 2.5

TUNNEL_CENTRES = (
    (14.93, 13.475), (14.00, 14.43), (9.32, 12.82),      # red half
    (13.08, 1.525), (14.00, 0.57), (18.68, 2.18),        # blue half
)


def near_tunnel(x, y):
    return any(math.hypot(x-cx, y-cy) <= TUNNEL_RADIUS_M for cx, cy in TUNNEL_CENTRES)


class TerrainWitness:
    """Remembers, per enemy slot, which terrain feats that opponent was seen doing.

    Both flags latch on and stay on for the rest of the episode. That is the
    right shape for the question: "can this opponent take the tunnel shortcut"
    does not change once answered, and a latched bit cannot flicker between
    steps, which would put noise into a network that sees one frame at a time.
    """

    __slots__ = ('_tunnel', '_climb', '_last', '_rises')

    def __init__(self):
        self._tunnel = {}    # slot -> True once the inside card was contacted
        self._climb = {}     # slot -> True once a terrace climb was measured
        self._last = {}      # slot -> previous height, for the rise measurement
        self._rises = {}     # slot -> [time, height delta] of pending rises

    def observe(self, obs, slot):
        """Fold one observation in, scoped to what the controlled robot can see.

        `obs['enemies']` already holds only what the observer's vision reaches
        (`lib.rs` filters on `visible`), so iterating it *is* the visibility
        filter. The tunnel half additionally needs the card occupancy, which the
        observation carries at match scope, so it is joined on robot id and then
        restricted back to slots that are currently seen.
        """
        seen = {r['slot']: r for r in obs.get('enemies', []) if r['slot'] in (1, 3, 4, 7)}
        self._read_cards(obs, seen)
        for s, robot in seen.items():
            self._read_heights(obs.get('time', 0.0), s, robot)

    def _read_cards(self, obs, seen):
        """An occupied inside card is proof; the ends alone are not.

        The engine reports occupancy at match scope rather than per observer, so
        the id has to be mapped back to a slot and then re-checked against what
        this robot can see. A robot under the roof is often outside line of
        sight, which makes this the rarer of the two witnesses - that is the
        honest cost of not being omniscient.
        """
        by_id = {r.get('id'): s for s, r in seen.items()}
        for name, zone in (obs.get('zones') or {}).items():
            if 'tunnel' not in name or not name.endswith(str(TUNNEL_INSIDE_CARD)):
                continue
            for robot_id in zone.get('occupants') or ():
                s = by_id.get(robot_id)
                if s is not None:
                    self._tunnel[s] = True

    def _read_heights(self, now, slot, robot):
        """Two consecutive ~terrace-sized rises away from a tunnel means it climbed.

        A single rise is ambiguous: the road bumps are 70 mm and the central
        highland has ramps, so one step up proves nothing. Requiring the pair
        within a short window is what separates "drove up onto the terrace" from
        "went over a bump", and the distance check removes tunnels, whose roof is
        the other way a height can jump by this much.
        """
        if not robot.get('alive', True):
            return
        x, y, z = robot['x'], robot['y'], robot.get('z', 0.0)
        previous = self._last.get(slot)
        self._last[slot] = z
        if previous is None or near_tunnel(x, y):
            self._rises.pop(slot, None)
            return
        rise = z - previous
        if rise < CLIMB_RISE_M:
            return
        pending = [t for t, _ in self._rises.get(slot, []) if now - t <= CLIMB_WINDOW_S]
        pending.append(now)
        if len(pending) >= 2:
            self._climb[slot] = True
        self._rises[slot] = [(t, 0.0) for t in pending]

    def bits(self, slot, robot=None):
        """(tunnel, climb, omni) for one slot: witnessed where seen, else prior.

        `robot` is the enemy's own public observation dict, which is what the
        design-table prior needs. Omitting it means "no observation to lean on",
        which yields the pure prior - the same three numbers an exported
        submission computes, since a self-contained policy has no witness and no
        choice but to read the table.

        Omni never has evidence to override: nothing in a position track says
        whether a robot can strafe. The prior is the only honest answer there.
        """
        prior = capability_bits(robot) if robot is not None else [0.0, 0.0, 0.0]
        if self._tunnel.get(slot):
            prior[0] = 1.0
        if self._climb.get(slot):
            prior[1] = 1.0
        return prior

    def snapshot(self):
        """Plain data, so a caller that pickles per-episode state can carry it."""
        return {'tunnel': sorted(s for s, v in self._tunnel.items() if v),
                'climb': sorted(s for s, v in self._climb.items() if v)}
