import copy
import math
import struct
import unittest
from rmuc2027_core.engine import Match
from rmuc2027_core.sentry_protocol import (PROFILE, SourceClock, InteractionLink, frame, unframe,
    sentry_word, SentryGateway, autoaim, payloads, LENGTHS, Commander)
from rmuc2027_core.radar2027 import RadarReferee


class WireTests(unittest.TestCase):
    def test_crc_framing_length_and_tampering(self):
        wire=frame(0x20D,bytes(14),255)
        self.assertEqual(unframe(wire),(0x20D,bytes(14),255))
        for index in (0,1,4,9,len(wire)-1):
            bad=bytearray(wire);bad[index]^=1
            with self.assertRaises(ValueError):unframe(bad)
        with self.assertRaises(ValueError):frame(0x201,bytes(16))

    def test_three_hz_rational_clock_never_becomes_2_86_or_3_33(self):
        clock=SourceClock();events=[]
        for tick in range(200):events+=clock.due('health',3,tick*.05)
        self.assertEqual(events,[i/3 for i in range(30)])

    def test_shared_uplink_budget_across_content_ids(self):
        link=InteractionLink()
        for i in range(30):self.assertIsNotNone(link.send(i/100,9,7,0x200+i,bytes(3)))
        self.assertIsNone(link.send(.3,9,7,0x2FF,b''))
        self.assertIsNotNone(link.send(1.,9,7,0x200,b''))

    def test_recipient_permissions_payload_and_base_bandwidth(self):
        link=InteractionLink()
        for sender,recipient in ((3,7),(1,9),(9,107),(7,109)):
            self.assertIsNone(link.send(0,sender,recipient,0x200,b'1'))
        self.assertIsNotNone(link.send(0,7,9,0x200,b'1'))
        self.assertIsNone(link.send(0,9,7,0x200,bytes(113)))
        self.assertIsNone(link.send(0,9,7,0x200,bytes(112),19900))
        self.assertIsNotNone(link.send(0,9,7,0x200,bytes(112),0))
        self.assertIsNone(link.send(0,3,0x8080,0x120,bytes(4)))

    def test_command_layout_and_no_counter_rollover(self):
        word=sentry_word(free=True,paid=True,local_ammo=100,remote_ammo=3,remote_heal=2,posture=6,rune=True)
        self.assertEqual(word&3,3);self.assertEqual(word>>2&2047,100)
        self.assertEqual(word>>13&15,3);self.assertEqual(word>>17&15,2)
        self.assertEqual(word>>21&7,6);self.assertEqual(word>>24,1)
        with self.assertRaises(ValueError):sentry_word(remote_ammo=16)

    def test_commander_repeat_dedup_and_sender_rate(self):
        c=Commander();self.assertTrue(c.submit(0,3,5,source=0x106));self.assertFalse(c.submit(.49,4,6,source=0x106))
        copies=[c.poll(t/10) for t in range(5)]
        self.assertEqual({p['id'] for p in copies},{1})
        self.assertIsNone(c.poll(.9));self.assertEqual(c.poll(1)['id'],1)
        self.assertTrue(c.submit(1,4,6,source=0x106));self.assertEqual(c.poll(1)['id'],2)
        self.assertTrue(c.submit(2,4,6,source=0x101,semi=True))
        self.assertFalse(c.submit(4.9,4,6,source=0x101,semi=True))
        with self.assertRaises(ValueError):c.submit(8,4,6)
        self.assertFalse(Commander('red',7).submit(0,0,0,source=0x101,semi=True))

    def test_commander_periodic_clock_and_delayed_copy_identity(self):
        c=Commander();c.submit(0,3,5,source=0x106)
        c.poll(.4)  # Four old burst copies remain queued.
        c.submit(.6,7,8,source=0x106)
        old=c.poll(.6)
        self.assertEqual((old['id'],old['x'],old['scheduled_time']),(1,3,.1))
        messages=[c.poll(t/20) for t in range(13,24)]
        self.assertTrue(any(m and m['id']==2 and m['scheduled_time']==1 for m in messages))

    def test_radar_map_cap_and_zero_coordinate_omission(self):
        with Match(observation_profile=PROFILE) as match:snapshot=match.snapshot()
        r=RadarReferee();points=[0]*24
        target=next(b for b in snapshot['robots'] if b['id']=='blue-1')
        points[:2]=[round(target[k]*100) for k in ('x','y')]
        packet=struct.pack('<24H',*points)
        for i in range(5):self.assertTrue(r.submit_positions('red',i/10,packet,snapshot))
        self.assertFalse(r.submit_positions('red',.5,packet,snapshot))
        self.assertEqual(r.marks[('red','blue-1')]['p'],15)
        self.assertNotIn(('red','blue-7'),r.marks)
        self.assertTrue(r.submit_positions('red',1.,packet,snapshot))


class RadarTests(unittest.TestCase):
    def test_rulebook_marking_numeric_example(self):
        r=RadarReferee()
        values=[r.mark('red','blue-1',i*.2,0) for i in range(1,16)]
        self.assertEqual(values[:5],[1,3,6,10,15]);self.assertEqual(values[-1],120)
        for i,expected in enumerate((119.2,117.6,115.2,112,108),1):
            self.assertAlmostEqual(r.mark('red','blue-1',3+i*.2,2),expected)
        self.assertAlmostEqual(r.mark('red','blue-1',4.5,math.inf),103.2)
        self.assertAlmostEqual(r.mark('red','blue-1',5.,math.inf),97.6)
        self.assertAlmostEqual(r.mark('red','blue-1',5.2,0),98.6)
        self.assertAlmostEqual(r.mark('red','blue-1',5.4,0),100.6)

    def test_death_preserves_progress_revive_resets_increment(self):
        r=RadarReferee();r.mark('red','blue-7',0,0,True);r.mark('red','blue-7',.2,0,False)
        self.assertEqual(r.mark('red','blue-7',.4,0,True),4)

    def test_double_command_idempotent_queued_and_capped(self):
        r=RadarReferee();r.time=120;r.teams['red']['vulnerability_seconds']=120
        packet=lambda count:struct.pack('<BB6sBHH',count,0,b'000000',0,0,0)
        self.assertTrue(r.command('red',packet(1))['double'])
        r.command('red',packet(1));self.assertEqual(r.teams['red']['double_until'],150)
        r.command('red',packet(2));self.assertEqual(r.teams['red']['double_until'],180)
        self.assertFalse(r.command('red',packet(3))['double'])

    def test_key_verification_level_protection_and_cooldown(self):
        r=RadarReferee();key=r.teams['blue']['key'].encode()
        cmd=struct.pack('<BB6sBHH',0,2,key,0,0,0)
        self.assertTrue(r.command('red',cmd)['key']);self.assertEqual(r.teams['blue']['level'],2)
        self.assertFalse(r.command('red',cmd)['key'])
        r.marks[('red','blue-7')]=dict(p=150)
        self.assertEqual(r.vulnerability('red','blue-7'),0)

    def test_undefined_fortification_rejected_and_finite_formula(self):
        r=RadarReferee();r.time=60
        r.teams['red'].update(shots=20,hits=[(10,10)]*10)
        cmd=struct.pack('<BB6sBHH',0,0,b'000000',1,60,60)
        self.assertFalse(r.command('red',cmd)['fortify']);self.assertIsNone(r.teams['red']['center'])
        for t in r.teams.values():t.update(shots=20,hits=[(i,i) for i in range(10)])
        self.assertAlmostEqual(r.area_fraction('red'),1/3)
        self.assertTrue(r.command('red',cmd)['fortify'])
        r.command('red',cmd);self.assertEqual(r.teams['red']['fortify_used'],1)


class ObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Match(observation_profile=PROFILE) as match:cls.initial=match.snapshot()

    def test_all_payload_lengths_and_unknown_slots(self):
        for team in ('red','blue'):
            for cmd,body in payloads(self.initial,team).items():self.assertEqual(len(body),LENGTHS[cmd])
        g=SentryGateway(copy.deepcopy(self.initial));obs=g.observation('red',[])
        self.assertEqual([r['slot'] for r in obs['allies']],[1,3,4,6])
        self.assertFalse(obs['allies'][-1]['position_valid'])
        self.assertTrue(all(r['position_source']=='radar' for r in obs['allies'][:3]))
        self.assertNotIn('max_hp',obs['enemies'][0]);self.assertNotIn('positions',obs)

    def test_speed_module_bearing_is_clockwise_from_north_and_tracks_muzzle(self):
        snapshot=copy.deepcopy(self.initial)
        robot=next(r for r in snapshot['robots'] if r['id']=='red-7')
        for aim,bearing in ((math.pi/2,0.),(0.,90.),(-math.pi/2,180.),(math.pi,270.)):
            for chassis_heading in (0.,1.2,-2.4):
                robot.update(aim=aim,heading=chassis_heading)
                x,y,angle=struct.unpack('<3f',payloads(snapshot,'red')[0x203])
                self.assertAlmostEqual(angle,bearing)
                obs=SentryGateway(snapshot).observation('red',[])
                self.assertEqual(obs['robots'][0]['heading'],chassis_heading)
                self.assertEqual(obs['robots'][0]['referee_pose']['muzzle_angle_deg'],bearing)

    def test_no_truth_fill_and_sources_refresh_at_protocol_rate(self):
        initial=copy.deepcopy(self.initial);g=SentryGateway(initial)
        before=g.observation('red',[])
        later=copy.deepcopy(initial);later.update(time=.05,tick=1)
        own=next(r for r in later['robots'] if r['id']=='red-7');own.update(hp=123,buffer_energy_j=5,energy=10000)
        ally=next(r for r in later['robots'] if r['id']=='red-1');ally.update(hp=1,max_hp=99999,x=20)
        later['positions']['red_outpost']['progress']=4.99
        g.update(later);obs=g.observation('red',[])
        self.assertEqual(obs['robots'][0]['hp'],before['robots'][0]['hp'])
        self.assertNotIn('energy',obs['robots'][0]);self.assertNotIn('respawn_progress',obs['robots'][0])
        self.assertEqual(obs['allies'][0]['x'],before['allies'][0]['x'])
        later.update(time=.1,tick=2);g.update(later)
        self.assertEqual(g.observation('red',[])['robots'][0]['hp'],123)
        self.assertEqual(g.observation('red',[])['allies'][0]['hp'],before['allies'][0]['hp'])

    def test_loss_marks_stale_instead_of_observing_referee_truth(self):
        g=SentryGateway(copy.deepcopy(self.initial));g.faults[('red','wave')]={'drop':True}
        later=copy.deepcopy(self.initial);later.update(time=1,tick=20)
        g.update(later);obs=g.observation('red',[])
        self.assertFalse(obs['referee']['sources']['0x0A01']['valid'])
        self.assertFalse(obs['enemies'][0]['position_valid']);self.assertNotIn('x',obs['enemies'][0])

    def test_delay_and_out_of_order_packet_cannot_replace_new_data(self):
        g=SentryGateway(copy.deepcopy(self.initial));g.faults[('red',0x201)]={'delay':.4}
        s=copy.deepcopy(self.initial);s.update(time=.1,tick=2);g.update(s)
        g.faults.clear();s.update(time=.2,tick=4);g.update(s)
        s.update(time=.5,tick=10);g.update(s)
        self.assertEqual(g.cache['red'][0x201]['source_time'],.5)

    def test_same_frame_hits_preserve_each_sequence(self):
        g=SentryGateway(copy.deepcopy(self.initial))
        for sequence in (1,2):g._receive('red',0x206,struct.pack('<BHhh',0,sequence,30,40),.1,.1,.1,'referee')
        self.assertEqual(sum(e['command']==0x206 for e in g.events['red']),2)

    def test_autoaim_nearest_target_independent_of_health_and_id_tiebreak(self):
        me=next(r for r in self.initial['robots'] if r['id']=='red-7')
        targets=[dict(id='blue-4',x=me['x']+3,y=me['y'],z=me['z'],hp=1),
                 dict(id='blue-3',x=me['x']+2,y=me['y'],z=me['z'],hp=1000)]
        self.assertEqual(autoaim(me,targets)['target_id'],'blue-3')
        targets[0]['x']=targets[1]['x']
        self.assertEqual(autoaim(me,targets)['target_id'],'blue-3')

    def test_direct_compact_batch_same_boundary_and_no_team_bypass(self):
        with Match(observation_profile=PROFILE) as m:
            import json
            direct=m.observation('red',[7]);compact=m.compact_observation('red',[7])
            ids,lines=m.observations([('red',[7]),('blue',[7])])
            self.assertEqual(direct,compact);self.assertEqual(direct,json.loads(lines[0]))
            with self.assertRaises(ValueError):m.observation('red',None)


if __name__=='__main__':unittest.main()
