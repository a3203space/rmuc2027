import copy
import math
import unittest

from rmuc2027_research.rl.v4_evaluation import detect_tunnel_passages


def trajectory(points, *, blue=False, reverse=False, angle=None):
    center=(18.68,2.265) if blue else (9.32,12.735)
    axis=(0.,-1.) if blue else (0.,1.)
    if angle is not None:axis=(math.cos(angle),math.sin(angle))
    passage=dict(id='blue_road_tunnel_roof' if blue else 'red_road_tunnel_roof',
        center=center,axis=axis,half=.175,width=1.)
    result=[]
    if reverse:points=list(reversed(points))
    for i,(along,across,fold) in enumerate(points):
        result.append(dict(time=i*.1,position=[center[0]+along*axis[0]-across*axis[1],
            center[1]+along*axis[1]+across*axis[0],.2],fold=fold,
            navigation={'passage':copy.deepcopy(passage)} if i<len(points)-2 else {}))
    return result


PASS=[(-.6,0.,1.),(-.3,0.,1.),(-.1,.02,1.),(.1,.02,1.),(.3,0.,1.),(.6,0.,0.)]


class TunnelGeometryTests(unittest.TestCase):
    def test_red_and_blue_observed_crossings_even_after_passage_is_cleared(self):
        for blue in (False,True):
            result=detect_tunnel_passages(trajectory(PASS,blue=blue))
            self.assertTrue(result['passed_tunnel']);self.assertEqual(len(result['crossings']),1)
            event=result['crossings'][0]
            self.assertEqual(event['observed_roof_samples'],2)
            self.assertGreaterEqual(event['min_fold_during_roof_intersection'],.8)
            self.assertLess(event['entry']['time'],event['roof_enter_time'])
            self.assertLess(event['roof_exit_time'],event['exit']['time'])

    def test_reverse_and_rotated_traversals(self):
        for reverse in (False,True):
            self.assertTrue(detect_tunnel_passages(trajectory([(*p[:2],1.) for p in PASS],reverse=reverse,angle=.73))['passed_tunnel'])

    def test_enter_then_back_out_and_start_inside_are_not_complete_crossings(self):
        partial=[(-.6,0.,1.),(-.3,0.,1.),(-.1,0.,1.),(0.,0.,1.),(-.1,0.,1.),(-.3,0.,1.)]
        self.assertFalse(detect_tunnel_passages(trajectory(partial))['passed_tunnel'])
        self.assertFalse(detect_tunnel_passages(trajectory(PASS[2:]))['passed_tunnel'])

    def test_folded_bypass_around_wall_endpoint_is_not_a_pass(self):
        around=[(-.6,0.,1.),(-.3,.8,1.),(-.1,.8,1.),(.1,.8,1.),(.3,.8,1.),(.6,0.,1.)]
        self.assertFalse(detect_tunnel_passages(trajectory(around))['passed_tunnel'])

    def test_fold_peak_away_from_roof_cannot_substitute_for_fold_in_roof(self):
        lowfold=[(a,b,1. if abs(a)>.3 else .2) for a,b,_ in PASS]
        self.assertFalse(detect_tunnel_passages(trajectory(lowfold))['passed_tunnel'])

    def test_late_samples_or_unobserved_intersection_are_not_confirmed(self):
        trace=trajectory(PASS);trace[3]['time']+=1.
        self.assertFalse(detect_tunnel_passages(trace)['passed_tunnel'])
        sparse=trajectory([(-.3,0.,1.),(.3,0.,1.),(.6,0.,1.)])
        self.assertFalse(detect_tunnel_passages(sparse)['passed_tunnel'])

    def test_missing_or_invalid_geometry_has_no_claim(self):
        trace=trajectory(PASS)
        for row in trace:row['navigation']={}
        self.assertFalse(detect_tunnel_passages(trace)['passed_tunnel'])
        self.assertEqual(detect_tunnel_passages(trace)['tested_passages'],0)


if __name__=='__main__':unittest.main()
