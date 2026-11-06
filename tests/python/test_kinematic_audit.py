import unittest
from rmuc2027_research.rl.warp_kinematic.audit import audit


class MotionAuditTests(unittest.TestCase):
    def rows(self, speed):
        return [dict(time=i*.5,robots=[dict(id='red-3',x=0.,y=0.,z=0.,vx=speed,vy=0.,alive=True)])
                for i in range(8)]

    def test_stationary_telemetry_contradiction_is_reported(self):
        report=audit(self.rows(1.))
        self.assertEqual(len(report['findings']),1)
        self.assertEqual(report['findings'][0]['id'],'red-3')

    def test_stationary_or_moving_consistent_state_is_valid(self):
        self.assertFalse(audit(self.rows(0.))['findings'])
        rows=self.rows(1.)
        for row in rows:row['robots'][0]['x']=row['time']
        self.assertFalse(audit(rows)['findings'])

    def test_nonfinite_and_death_are_handled(self):
        rows=self.rows(1.)
        for row in rows:row['robots'][0]['alive']=False
        self.assertFalse(audit(rows)['findings'])
        rows[0]['robots'][0]['x']=float('nan')
        self.assertEqual(audit(rows)['findings'][0]['kind'],'nonfinite-motion')
