import unittest
from rmuc2027_core.engine import Match
from rmuc2027_core.sdk import terrain_height,aim_at,aim_pitch
from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion
from test_kinematic_engine import StationaryFixture


class PlacedMotion(ArenaMotion):
    def step_frame(self,request):
        if request['sequence']==0:
            for robot in request['robots']:
                if robot['id'] in ('red-3','blue-4'):
                    x=5. if robot['id']=='red-3' else 7.
                    robot['pos']=[x,3.,terrain_height(request['arena'],x,3.)]
        return super().step_frame(request)


class LegacyPlacedMotion(PlacedMotion):
    step_encoded=None


class EncodedTransportTests(unittest.TestCase):
    def test_encoded_frames_match_dictionary_transport_in_real_combat(self):
        results=[]
        for adapter in (LegacyPlacedMotion(),PlacedMotion()):
            trace=[];events=[]
            with Match(physics_adapter=adapter,battle_settings={'drones_enabled':False}) as match:
                def remember(frame):
                    trace.append([(r['id'],r['x'],r['y'],r['z'],r['vx'],r['vy'],r['hp'],r['ammo']) for r in frame['robots']])
                    events.extend(frame['events'])
                for _ in range(120):remember(match.step({}))
                remember(match.step({'red':{'3':{'buy_ammo':{'remote':True,'amount':100}}}}))
                for _ in range(130):frame=match.step({});remember(frame)
                self.assertEqual(next(r['ammo'] for r in frame['robots'] if r['id']=='red-3'),100)
                for tick in range(60):
                    shooter=next(r for r in frame['robots'] if r['id']=='red-3')
                    target=next(r for r in frame['robots'] if r['id']=='blue-4')
                    frame=match.step({'red':{'3':{'aim':aim_at(shooter,target),'pitch':aim_pitch(shooter,target),'fire':tick<20}}})
                    remember(frame)
                self.assertTrue(any(e['type']=='shot' for e in events))
                self.assertTrue(any(e['type']=='hit' for e in events))
                results.append((trace,events,match.observation('red',[3]),match.digest()))
        self.assertEqual(results[0],results[1])

    def test_multiline_encoded_response_breaks_transport_before_writing(self):
        class BadFrame(StationaryFixture):
            def step_encoded(self,request):return b'{}\n{}\n'
        with Match(physics_adapter=BadFrame(),battle_settings={'drones_enabled':False}) as match:
            before=match.snapshot()
            with self.assertRaisesRegex(ValueError,'one newline-terminated'):match.step({})
            self.assertTrue(match._broken)
            self.assertEqual(match.snapshot(),before)
            with self.assertRaisesRegex(RuntimeError,'interrupted'):match.step({})
