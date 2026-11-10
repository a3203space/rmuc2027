import copy
import math
import struct
import unittest

from rmuc2027_core.radar_inference import (RadarInference, RadarInferenceReceiver,
    ChassisClassifier, CONTENT_OVERVIEW, CONTENT_ALLIES, CONTENT_ENEMIES, CONTENT_PREDICTION)
from rmuc2027_core.sentry_protocol import SentryGateway, PROFILE
from rmuc2027_core.engine import Match


ARENA = {'width': 28, 'height': 15, 'obstacles': [], 'terrain': []}


def enemy_xy(x, y=5.):
    values = [0]*12
    values[4:6] = [round(x*100), round(y*100)]
    return struct.pack('<12H', *values)


def feed_combat(estimator, t, enemy_hp, second_ally=False):
    # A clean isolated duel, or two plausible allied shooters. Other slots dead.
    positions = [0]*12; positions[:2] = [800, 700]
    estimator.consume(0xA01, struct.pack('<12H', *positions), t)
    own_positions = [500, 700, 600, 700, 400, 400, 300, 300]
    estimator.consume(0xF001, struct.pack('<B8H', 15, *own_positions), t)
    estimator.consume(3, struct.pack('<4Hh7H', 400, 0, 400 if second_ally else 0, 0, 0, 0, 1000, 10000, 1000, 10000, 0, 0), t)
    estimator.consume(0xA02, struct.pack('<6H', enemy_hp, 0, 0, 0, 0, 0), t)
    estimator.consume(0xA05, bytes(47), t)
    estimator.update(t, ARENA)


class RadarInferenceTests(unittest.TestCase):
    def test_non_overlapping_motion_requires_six_seconds_and_static_does_not_lower_estimate(self):
        r = RadarInference('red')
        for i in range(201):
            t = i/20
            r.consume(0xA01, enemy_xy(2+t), t)
            r.update(t, ARENA)
            if t < 6:
                self.assertFalse(r.result['enemies'].get('3', {}).get('speed_valid', False))
        before = copy.deepcopy(r.result['enemies']['3'])
        self.assertTrue(before['speed_valid'])
        self.assertGreater(before['speed_lb'], .8)
        self.assertTrue(before['class_unknown'])
        for i in range(201, 601):
            t = i/20; r.consume(0xA01, enemy_xy(12), t); r.update(t, ARENA)
        after = r.result['enemies']['3']
        self.assertEqual(before['speed_lb'], after['speed_lb'])
        self.assertAlmostEqual(before['robust_speed'], after['robust_speed'], places=5)
        self.assertFalse(after['speed_valid'])

    def test_duplicate_and_out_of_order_packets_do_not_add_samples(self):
        r = RadarInference('red')
        for t in (0., .1, .2): r.consume(0xA01, enemy_xy(5+t), t)
        n = len(r.positions[('blue', 3)])
        r.consume(0xA01, enemy_xy(20), .1); r.consume(0xA01, enemy_xy(20), .2)
        self.assertEqual(n, len(r.positions[('blue', 3)]))

    def test_drifting_motion_updates_proxy_gradually_without_erasing_historical_peak(self):
        r=RadarInference('red');previous=0.;estimates=[]
        # Alternate along a long corridor to stay on the field. Speed first
        # increases, then decreases over enough moving windows to be evidence.
        x=3.;direction=1.
        for i in range(1601):
            t=i/20;speed=.5 if t<15 else 1.2 if t<40 else .45
            if i:
                if x>24:direction=-1.
                if x<3:direction=1.
                x+=direction*speed*.05
            r.consume(0xA01,enemy_xy(x),t);r.update(t,ARENA)
            if abs(t*2-round(t*2))<1e-8:
                estimates.append((t,r.result['enemies']['3']['robust_speed'],r.result['enemies']['3']['speed_lb']))
        at=lambda t:next(row for row in estimates if row[0]==t)
        self.assertGreater(at(40)[1],at(15)[1]+.3)
        self.assertLess(at(80)[1],at(40)[1]-.2)
        self.assertGreaterEqual(at(80)[2],at(40)[2])
        self.assertGreater(at(80)[2],at(80)[1])
        self.assertLess(max(abs(b[1]-a[1]) for a,b in zip(estimates,estimates[1:]) if a[0]>6),.1)

    def test_observation_history_not_hidden_attributes_controls_prediction(self):
        a, b = RadarInference('red'), RadarInference('red')
        for r in (a, b):
            for i in range(121):
                t=i/10; r.consume(0xA01, enemy_xy(3+.5*t), t); r.update(t, ARENA)
        # No API accepts a hidden robot design, true aiming skill, or damage source.
        a.secret_model = 2773; b.secret_model = 2772
        a.true_accuracy = .9; b.true_accuracy = .1
        self.assertEqual(a.update(12.5, ARENA), b.update(12.5, ARENA))

    def test_uncalibrated_class_keeps_passage_hypotheses(self):
        r=RadarInference('red')
        arena=dict(ARENA, obstacles=[dict(id='near_tunnel_roof',x=9.,y=4.5,w=.5,h=1.,bottom=.25)])
        for i in range(141):
            t=i/10; r.consume(0xA01,enemy_xy(2+.4*t),t); r.update(t,arena)
        prediction=r.result['predictions']['3']
        self.assertTrue(r.result['enemies']['3']['class_unknown'])
        self.assertGreater(prediction['tunnel_probability'],0)
        self.assertEqual(prediction['tunnel_confidence'],0)
        self.assertGreater(prediction['route']['tunnel'],0)
        self.assertGreater(prediction['route']['detour'],0)
        self.assertAlmostEqual(sum(prediction['route'][k] for k in ('straight','detour','tunnel','step')),1.)

    def test_classifier_has_explicit_validated_calibration_entry(self):
        rows=[]
        for index,label in enumerate(ChassisClassifier.labels):
            for j in range(6): rows.append(dict(label=label,features=[index*4+j*.01,index*4+.1,0.,.1]))
        self.assertFalse(ChassisClassifier.fit(rows).predict([0,.1,0,.1])[1])
        p,supported=ChassisClassifier.fit(rows,validated=True).predict([0,.1,0,.1])
        self.assertTrue(supported); self.assertGreater(p[0],.9)

    def test_public_clearance_and_step_height_change_class_hypotheses(self):
        r=RadarInference('red')
        for i in range(61):r.consume(0xA01,enemy_xy(4+i/100),i/10)
        def prediction(clearance,step_height,fold):
            arena=dict(ARENA,obstacles=[dict(id='tunnel_roof',x=6,y=4.5,w=.4,h=1.,bottom=clearance)],
                terrain=[dict(id='band',kind='platform',x=8,y=4,w=1,h=1,elevation=step_height)])
            estimate=dict(p_wheel=1-fold,p_fold=fold,p_omni=0.,class_unknown=False)
            return r._prediction(('blue',3),estimate,6.,arena)
        self.assertGreater(prediction(.25,.2,1.)['tunnel_probability'],prediction(.25,.2,0.)['tunnel_probability'])
        self.assertGreater(prediction(.65,.2,0.)['tunnel_probability'],prediction(.25,.2,0.)['tunnel_probability'])
        self.assertGreater(prediction(.25,.1,0.)['step_probability'],prediction(.25,.4,0.)['step_probability'])

    def test_personal_contribution_only_updates_in_isolated_windows(self):
        isolated, multiple=RadarInference('red'),RadarInference('red')
        for i in range(13):
            feed_combat(isolated,float(i),400-i*10)
            feed_combat(multiple,float(i),400-i*10,second_ally=True)
        own=isolated.result['robots']['ally']['1']
        self.assertTrue(own['contribution_valid']); self.assertGreater(own['contribution_rate'],0)
        self.assertEqual(multiple.result['robots']['ally']['1']['contribution_n'],0)
        self.assertEqual(multiple.result['robots']['ally']['3']['contribution_n'],0)
        self.assertTrue(multiple.result['robots']['enemy']['1']['pressure_valid'])
        self.assertTrue(multiple.result['robots']['enemy']['1']['unknown']['multiple_candidates'])

    def test_old_combat_evidence_does_not_regain_confidence_after_one_new_sample(self):
        r=RadarInference('red')
        for i in range(25):feed_combat(r,float(i),1000-i*10)
        initial=r.result['robots']['ally']['1']
        self.assertTrue(initial['contribution_valid'])
        # No legal observation for a long interval; one fresh clean interval
        # cannot restore confidence from an old match-wide cumulative count.
        feed_combat(r,125.,760);feed_combat(r,126.,750)
        result=r.result['robots']['ally']['1']
        self.assertFalse(result['contribution_valid'])
        self.assertLess(result['contribution_confidence'],initial['contribution_confidence']/2)
        self.assertGreaterEqual(result['contribution_n'],initial['contribution_n'])

    def test_healing_and_known_buff_are_not_credited_as_shooting_changes(self):
        r=RadarInference('red')
        for i in range(10): feed_combat(r,float(i),400-i*10)
        prior=r.long[('red',1)]['n']
        feed_combat(r,10.,400)
        self.assertEqual(r.long[('red',1)]['n'],prior)
        self.assertTrue(r.result['robots']['enemy']['1']['unknown']['health_reset'])
        buff=bytearray(47);buff[3]=50
        r.consume(0xA05,bytes(buff),10.5)
        r.consume(0xA02,struct.pack('<6H',300,0,0,0,0,0),11.)
        r.consume(0xA01,struct.pack('<12H',800,700,*([0]*10)),11.)
        r.update(11.,ARENA)
        self.assertEqual(r.long[('red',1)]['n'],prior)

    def test_packet_roundtrip_lengths_aging_and_page_order(self):
        r=RadarInference('red'); receiver=RadarInferenceReceiver()
        for i in range(13): feed_combat(r,float(i),400-i*10)
        for content,size in ((CONTENT_OVERVIEW,62),(CONTENT_ALLIES,88),(CONTENT_ENEMIES,88),(CONTENT_PREDICTION,83)):
            body=r.packet(content,12.)
            self.assertEqual(len(body),size);self.assertTrue(receiver.receive(content,body,12.2))
            self.assertFalse(receiver.receive(content,body,12.3))
        now=receiver.observation(12.2)
        self.assertTrue(now['robots']['ally']['1']['contribution_valid'])
        self.assertGreaterEqual(now['robots']['ally']['1']['contribution_age'],.19)
        old=receiver.observation(30.)
        self.assertFalse(old['robots']['ally']['1']['contribution_valid'])
        self.assertFalse(old['robots']['enemy']['1']['pressure_valid'])
        with self.assertRaises(ValueError):receiver.receive(CONTENT_ALLIES,b'bad',30)


class GatewayInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with Match(observation_profile=PROFILE) as match:cls.initial=match.snapshot()

    def test_real_wire_budget_and_dropped_inference_has_no_memory_bypass(self):
        initial=copy.deepcopy(self.initial)
        for team in ('red','blue'):initial['bases'][team]['hp']=0
        gateway=SentryGateway(initial)
        for tick in range(1,41):
            state=copy.deepcopy(initial);state.update(tick=tick,time=tick*.05);gateway.update(state)
        sent=[r for r in gateway.link.trace if r['sender']==9 and .999<r['time']<2]
        self.assertEqual(len(sent),26)
        self.assertEqual(sum(r['bytes'] for r in sent),2210)
        self.assertTrue(all(r['accepted'] for r in sent))
        self.assertTrue(gateway.observation('red',[])['radar_inference']['robots']['ally'])
        gateway.faults[('red','inference')]={'drop':True}
        last=dict(gateway.inference_received['red'].pages)
        for tick in range(41,65):
            state=copy.deepcopy(initial);state.update(tick=tick,time=tick*.05);gateway.update(state)
        self.assertEqual(last,gateway.inference_received['red'].pages)


if __name__=='__main__':unittest.main()
