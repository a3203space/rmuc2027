"""Real-engine projections for the operator HUD, including visibility boundaries."""
import copy
import unittest
from rmuc2027_core.engine import Match
from rmuc2027_server.projection import project
from rmuc2027_server.hud import actions


class OperatorHudTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.match = Match()
        cls.frame = copy.deepcopy(cls.match._frame)
        cls.observations = {slot: cls.match.compact_observation('red', [slot]) for slot in (1, 3, 4, 6, 7)}

    @classmethod
    def tearDownClass(cls):
        cls.match.close()

    def view(self, f=None, slot=3):
        return project(f or self.frame, self.observations[slot], 'red', slot, 'hud-test', include_static=True)

    def test_all_control_positions_and_polygons_survive_projection(self):
        f = copy.deepcopy(self.frame)
        f['positions']['red_outpost'].update(progress=2.5, active=['red', 'blue'])
        f['zones']['red_outpost'].update(progress=.5, contested=True, occupants=['red-3', 'blue-7'])
        p = self.view(f)
        self.assertEqual(set(p['tactical']['positions']), {'red_outpost', 'blue_outpost', 'red_base', 'blue_base'})
        self.assertEqual(p['tactical']['control_rules']['capture_s'], 5)
        self.assertEqual(p['tactical']['positions']['red_outpost']['progress'], 2.5)
        self.assertEqual(p['scene']['zones']['red_outpost']['progress'], .5)
        self.assertEqual(p['tactical']['zones']['red_outpost']['active_teams'], ['blue', 'red'])
        self.assertTrue(p['tactical']['zones']['red_outpost']['contested'])
        self.assertEqual(p['static']['arena']['zones'], f['arena']['zones'])
        self.assertTrue(any(z.get('polygon') for z in p['static']['arena']['zones']))

    def test_real_structure_maximum_and_rebuild_are_not_overwritten(self):
        f = copy.deepcopy(self.frame)
        f['bases']['red'].update(hp=875, max_hp=1500, shield_hp=80)
        f['outposts']['blue'].update(hp=0, rebuild={'elapsed':2, 'required':5, 'progress':.4, 'robot':'red-3'})
        p = self.view(f)
        self.assertEqual(p['tactical']['bases']['red']['max_hp'], 1500)
        self.assertEqual(p['tactical']['bases']['red']['hp'], 875)
        self.assertEqual(p['tactical']['outposts']['blue']['rebuild']['progress'], .4)

    def test_roles_only_offer_relevant_actions_and_host_prices(self):
        for slot in (1, 3, 4, 6, 7):
            p = self.view(slot=slot)
            names = {a['name'] for a in p['actions']}
            if slot == 1: self.assertIn('assemble', names)
            else: self.assertNotIn('assemble', names)
            self.assertEqual('air_launch' in names, slot == 6)
            self.assertEqual('posture' in names, slot == 7)
            if slot != 6:
                remote = next(a for a in p['actions'] if a['name'] == 'ammo' and a['value'] == 'remote')
                self.assertEqual(remote['cost'], 150)
                self.assertFalse(remote['enabled'])

    def test_purchase_conditions_pending_heal_and_death(self):
        f = copy.deepcopy(self.frame); f['time'] = 121; f['teams']['red']['resources'] = 1000
        r = next(r for r in f['robots'] if r['id'] == 'red-3')
        r.update(last_shot_time=0, last_damage_time=0, pending_heal=125)
        a = actions(f, r)
        self.assertFalse(next(x for x in a if x['name'] == 'heal')['enabled'])
        self.assertEqual(next(x for x in a if x['name'] == 'heal')['cost'], 91)
        r['alive'] = False
        a = actions(f, r)
        self.assertTrue(next(x for x in a if x['name'] == 'revive')['enabled'])
        self.assertTrue(all(not x['enabled'] for x in a if x['name'] == 'ammo'))

    def test_penalty_removes_tactical_actions_and_enemy_private_fields(self):
        f = copy.deepcopy(self.frame)
        r = next(r for r in f['robots'] if r['id'] == 'red-3'); r['fpv_blocked_until'] = 50
        p = self.view(f)
        for key in ('tactical', 'actions', 'economy', 'air', 'assembly', 'scene'):
            self.assertNotIn(key, p)
        p = self.view()
        for enemy in p['tactical']['robots']:
            self.assertNotIn('pending_heal', enemy)
            self.assertNotIn('heat', enemy)
            self.assertNotIn('drones', enemy)

    def test_camera_cut_keeps_tactical_hud(self):
        f = copy.deepcopy(self.frame)
        r = next(r for r in f['robots'] if r['id'] == 'red-3'); r.update(overheated=True, heat=100)
        p = self.view(f)
        self.assertTrue(p['display']['camera_off'])
        self.assertIn('bases', p['tactical'])
        self.assertEqual(p['scene']['robots'], [])
        self.assertEqual(self.view(slot=6)['display']['reason'], 'hangar')

class OperatorRoomTests(unittest.TestCase):
    def test_two_seats_receive_hud_and_controller_receipts(self):
        import tempfile
        import time
        from rmuc2027_server.room import Room
        from rmuc2027_server.protocol import Join, Seat, Operation
        with tempfile.TemporaryDirectory() as directory:
            room = Room(owner_key='hud-test', directory=directory, duration=2, ai=False, ceremony=False)
            try:
                host = room.join(Join(nickname='步兵', host_key='hud-test'))['token']
                sentry = room.join(Join(nickname='哨兵'))['token']
                for token, slot in ((host, 3), (sentry, 7)):
                    room.attach(token); room.select(token, Seat(team='red', slot=slot)); room.ready(token, True)
                room.start(host)
                deadline = time.monotonic() + 4
                while not room.snapshot(sentry)['player'] and time.monotonic() < deadline: time.sleep(.02)
                first = room.snapshot(sentry)['player']
                self.assertIsNotNone(first)
                mode = 'defense' if first['self'].get('posture') != 'defense' else 'move'
                room.submit(sentry, Operation(match_id=room.match_id, seq=10, name='posture', value=mode))
                room.submit(host, Operation(match_id=room.match_id, seq=10, name='ammo', value='remote'))
                while time.monotonic() < deadline:
                    a, b = room.snapshot(host), room.snapshot(sentry)
                    if a['operations'] and b['operations'] and b['player']['self'].get('posture') == mode: break
                    time.sleep(.02)
                self.assertIsNone(room.error)
                self.assertTrue(a['operations'])
                self.assertTrue(b['operations'])
                self.assertEqual(a['player']['slot'], 3)
                self.assertEqual(b['player']['slot'], 7)
                self.assertEqual(len(a['player']['tactical']['positions']), 4)
                self.assertEqual(b['operations'][-1]['name'], 'posture')
                self.assertEqual(b['operations'][-1]['status'], 'applied')
                self.assertEqual(b['player']['self']['posture'], mode)
                self.assertEqual(a['operations'][-1]['name'], 'ammo')
                self.assertEqual(a['operations'][-1]['status'], 'rejected')
                self.assertIn('经济', a['operations'][-1]['message'])
            finally: room.close()

class OperatorControllerTests(unittest.TestCase):
    def test_rpc_survives_odd_tick_and_is_consumed_once(self):
        from rmuc2027_server.control import HumanController, baseline_helpers
        from rmuc2027_server.protocol import Input
        for source_tick in (0, 1):
            with self.subTest(source_tick=source_tick):
                match = Match()
                try:
                    for _ in range(source_tick): match.step({})
                    controller = HumanController(baseline_helpers())
                    robot = next(r for r in match._frame['robots'] if r['id'] == 'red-7')
                    controller.operation('posture', 'defense', robot)
                    sent = []
                    for _ in range(4):
                        observation = match.compact_observation('red', [7])
                        action = controller.action(observation, robot, Input(match_id='test', seq=1, yaw=0, pitch=0))
                        if 'posture' in action: sent.append(observation['tick'])
                        frame = match.step({'red': {'7': action}})
                        robot = next(r for r in frame['robots'] if r['id'] == 'red-7')
                    self.assertEqual(sent, [source_tick + source_tick % 2])
                    self.assertEqual(robot['posture'], 'defense')
                    self.assertEqual(controller.pending, [])
                finally: match.close()

if __name__ == '__main__': unittest.main()
