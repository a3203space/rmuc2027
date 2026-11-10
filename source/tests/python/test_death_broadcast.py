"""Operator death presentation and safe, repeated LAN combat announcements."""
import copy
import unittest
from rmuc2027_core.engine import Match
from rmuc2027_server.projection import project, PublicEventLog


class DeathBroadcastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.match = Match()
        cls.source = copy.deepcopy(cls.match._frame)
        cls.observation = cls.match.operator_observation('red', 3)

    @classmethod
    def tearDownClass(cls):
        cls.match.close()

    def test_operator_transport_retains_policy_shape_and_validates_slot(self):
        self.assertEqual(self.observation, self.match.compact_observation('red', [3]))
        with self.assertRaises(Exception):
            self.match.operator_observation('red', 2)

    def test_death_keeps_scene_and_only_camera_grayscale_then_revives(self):
        f = copy.deepcopy(self.source)
        own = next(r for r in f['robots'] if r['id'] == 'red-3')
        own.update(alive=False, hp=0)
        p = project(f, self.observation, 'red', 3, 'death')
        self.assertFalse(p['display']['camera_off'])
        self.assertTrue(p['display']['grayscale'])
        self.assertIsNotNone(p['display']['camera'])
        self.assertTrue(p['scene']['robots'])
        self.assertIn('tactical', p)
        self.assertFalse(p['weapon']['ready'])
        own.update(alive=True, hp=own['max_hp'])
        self.assertFalse(project(f, self.observation, 'red', 3, 'death')['display']['grayscale'])

    def test_cards_and_overheat_remain_blind_even_when_dead(self):
        for change in ({'fpv_blocked_until':10}, {'red_card':True}, {'overheated':True,'heat':300}):
            f = copy.deepcopy(self.source)
            own = next(r for r in f['robots'] if r['id']=='red-3')
            own.update(alive=False,hp=0,**change)
            p=project(f,self.observation,'red',3,'death')
            self.assertTrue(p['display']['camera_off'])
            self.assertFalse(p['display']['grayscale'])
            self.assertEqual(p.get('scene',{}).get('robots',[]),[])
            if p['display']['ui_blocked']:
                self.assertNotIn('public_events',p)

    def test_public_events_repeat_without_private_or_positional_fields(self):
        log=PublicEventLog()
        f=copy.deepcopy(self.source);f.update(tick=60,time=1)
        f['events']=[{'id':'60:0','type':'destroy','robot':'blue-3','killer':'red-7','team':'red','tick':60,'time':1,'x':23,'heat':200},
                     {'id':'60:1','type':'shot','robot':'blue-4','x':22},
                     {'id':'60:2','type':'ammo_purchase','robot':'blue-4','amount':100}]
        first=log.update(f)
        self.assertEqual(len(first),1)
        self.assertEqual(set(first[0]),{'id','type','robot','killer','team','tick','time'})
        self.assertEqual(log.update(f),first)
        f.update(tick=61,time=1.02,events=[])
        self.assertEqual(log.update(f),first)
        for team in ('red','blue'):
            p=project(f,self.match.operator_observation(team,3),team,3,'death',public_events=first)
            self.assertEqual(p['public_events'],first)
            self.assertNotIn('destroy',[e['type'] for e in p['events']])
            self.assertTrue(all('kills' in r for r in p['tactical']['robots']))
        f.update(tick=1020,time=17)
        self.assertEqual(log.update(f),[])

    def test_resurrection_order_and_prior_projection_stay_immutable(self):
        log=PublicEventLog();f=copy.deepcopy(self.source)
        f.update(tick=1,time=.02,events=[{'id':'1:0','tick':1,'time':.02,'type':'destroy','robot':'red-3','killer':'blue-7'}])
        old=log.update(f)
        f.update(tick=2,time=.04,events=[{'id':'2:0','tick':2,'time':.04,'type':'respawn','robot':'red-3'}])
        self.assertEqual([e['type'] for e in log.update(f)],['destroy','respawn'])
        self.assertEqual(len(old),1)

if __name__=='__main__':
    unittest.main()
