"""Preparation-stage performance selection and authoritative first-frame stats."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from rmuc2027_core.paths import ROOT
from rmuc2027_server.protocol import Join,Profile,Seat,SeatConfig
from rmuc2027_server.room import Room

BOOK=json.loads((ROOT/'configs/rules.json').read_text())['rulebook']['profiles']

class PerformanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        env=patch.dict(os.environ,RMUC_USER_DATA=self.temp.name);env.start();self.addCleanup(env.stop)

    def room(self,slot=3,ceremony=False):
        room=Room(owner_key='profile',directory=Path(self.temp.name)/f'{slot}-{time.monotonic_ns()}',duration=1,realtime=False,ai=False,ceremony=ceremony)
        self.addCleanup(room.close)
        token=room.join(Join(nickname='步兵',host_key='profile'))['token']
        room.attach(token);room.select(token,Seat(team='red',slot=slot))
        return room,token

    def wait(self,predicate):
        deadline=time.monotonic()+5
        while not predicate() and time.monotonic()<deadline:time.sleep(.01)
        self.assertTrue(predicate())

    def assert_stats(self,robot,chassis,weapon):
        c=BOOK['infantry_chassis'][chassis][0];w=BOOK['infantry_weapon'][weapon][0]
        self.assertEqual(robot['chassis_profile'],chassis)
        self.assertEqual(robot['weapon_profile'],weapon)
        self.assertEqual(robot['max_hp'],c['hp'])
        self.assertEqual(robot['power_limit'],c['power_w'])
        self.assertEqual(robot['max_heat'],w['max_heat'])
        self.assertEqual(robot['cooling_rate'],w['cooling'])

    def test_four_combinations_on_both_slots_reach_real_engine(self):
        for slot in (3,4):
            for chassis in ('hp','power'):
                for weapon in ('cooling','burst'):
                    with self.subTest(slot=slot,chassis=chassis,weapon=weapon):
                        room,token=self.room(slot)
                        saved=room.performance(token,Profile(chassis=chassis,weapon=weapon))
                        me=next(p for p in saved['players'] if p['id']==room.player(token).id)
                        self.assertEqual(me['profile'],{'chassis':chassis,'weapon':weapon})
                        room.ready(token,True);room.start(token)
                        self.wait(lambda:room.status=='finished' or room.error is not None)
                        self.assertIsNone(room.error)
                        self.assert_stats(room.snapshot(token)['player']['self'],chassis,weapon)
                        replay=json.loads((room.directory/room.result['replay']).read_text())
                        robot=next(r for r in replay['frames'][1]['robots'] if r['id']==f'red-{slot}')
                        self.assert_stats(robot,chassis,weapon)
                        room.close()

    def test_presentation_uses_selected_profiles_without_advancing_time(self):
        for chassis in ('hp','power'):
            for weapon in ('cooling','burst'):
                with self.subTest(chassis=chassis,weapon=weapon):
                    room,token=self.room(ceremony=True)
                    room.performance(token,Profile(chassis=chassis,weapon=weapon))
                    room.ready(token,True);room.start(token)
                    self.wait(lambda:room.status=='loading')
                    intro=copy.deepcopy(room.snapshot(token)['presentation'])
                    self.assertEqual(intro['tick'],0);self.assertEqual(intro['time'],0)
                    self.assert_stats(next(r for r in intro['robots'] if r['id']=='red-3'),chassis,weapon)
                    room.close()

    def test_save_cancels_readiness_and_hardware_change_preserves_profile(self):
        room,token=self.room()
        room.ready(token,True)
        room.performance(token,Profile(chassis='power',weapon='burst'))
        self.assertFalse(room.player(token).ready)
        room.configure_seat(token,SeatConfig(team='red',slot=3,model=301))
        self.assertEqual(room.player(token).profile,{'chassis':'power','weapon':'burst'})
        room.performance(token,Profile(chassis='hp',weapon='cooling'))
        self.assertEqual(room.player(token).profile,{'chassis':'hp','weapon':'cooling'})

    def test_reconnect_preserves_saved_performance(self):
        room,token=self.room(4)
        room.performance(token,Profile(chassis='power',weapon='cooling'))
        room.detach(token,room.player(token).generation);room.attach(token)
        me=next(p for p in room.snapshot(token)['room']['players'] if p['id']==room.player(token).id)
        self.assertEqual(me['profile'],{'chassis':'power','weapon':'cooling'})

    def test_other_player_profile_is_not_modified_and_match_locks_changes(self):
        room,token=self.room()
        other=room.join(Join(nickname='另一台'))['token'];room.attach(other);room.select(other,Seat(team='blue',slot=4))
        room.performance(token,Profile(chassis='power',weapon='burst'))
        self.assertEqual(room.player(other).profile,{'chassis':'hp','weapon':'cooling'})
        for phase in ('starting','loading','intro','countdown','running'):
            room.status=phase
            with self.assertRaises(ValueError):room.performance(token,Profile(chassis='hp',weapon='cooling'))
        self.assertEqual(room.player(token).profile,{'chassis':'power','weapon':'burst'})

    def test_non_infantry_and_invalid_values_are_rejected(self):
        room,token=self.room(1)
        with self.assertRaises(ValueError):room.performance(token,Profile(chassis='hp',weapon='cooling'))
        for bad in ({'chassis':'bad','weapon':'cooling'},{'chassis':'hp','weapon':'bad'},{'chassis':'hp','weapon':'cooling','request_id':'local-only'}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):Profile(**bad)

if __name__=='__main__':unittest.main()
