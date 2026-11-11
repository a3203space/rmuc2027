import copy
import json
import unittest

from rmuc2027_research.rl.v4_campaign import _evaluation_diagnostics


class EvaluationDiagnosticsTests(unittest.TestCase):
    def test_ground_truth_is_trace_only_and_all_fields_are_detached(self):
        robot=dict(id='red-7',team='red',slot=7,vx=.2,vy=.1,vz=0.,quaternion=[0.,0.,0.,1.],
                   angular_velocity=[0.,0.,.4],grounded=True,rfid=[1,0],powered=True)
        obs=dict(robots=[robot],referee=dict(sources={'0x0209':dict(valid=True,age=.1,source_time=1.,receipt_time=1.1)},
                                          control={'red_outpost':1}))
        info=dict(metrics={'capture_seconds':2.},control={'red':3.},reward_terms={'capture_seconds':.02},
                  action_trace=dict(execution_trace={'fold':1.},executed_action={'move':[.2,0.],'brake':False}))
        private=dict(time=1.1,robots=[dict(team='red',slot=7,weak=True,invulnerable=False,
                    zone_contacts={'red_outpost':1.},last_raw_zones=[],secret_target=[12,5]),
                    dict(team='blue',slot=7,weak=False)],
                    zones={'red_outpost':dict(owner='blue',progress=2.,occupants=['blue-3'],contested=False)},
                    positions={'red':dict(owner='red',controller='blue',progress=2.,active=['blue-3'])},
                    training_capabilities={'secret':'never copied'})
        original=copy.deepcopy((obs,info,private,robot))
        result=_evaluation_diagnostics(obs,info,private,robot)
        self.assertEqual((obs,info,private,robot),original)
        self.assertNotIn('weak',result['legal_self'])
        self.assertTrue(result['evaluation_ground_truth']['robot']['weak'])
        self.assertEqual(result['evaluation_ground_truth']['zones']['red_outpost']['occupants'],['blue-3'])
        self.assertNotIn('training_capabilities',json.dumps(result));self.assertNotIn('secret_target',json.dumps(result))
        result['legal_self']['quaternion'][0]=99
        result['evaluation_ground_truth']['robot']['zone_contacts'].clear()
        result['execution']['executed_action']['move'][0]=99
        self.assertEqual((obs,info,private,robot),original)

    def test_missing_legal_packets_are_explicitly_empty(self):
        result=_evaluation_diagnostics({}, {'metrics':{}}, {},dict(team='blue',slot=7))
        self.assertEqual(result['legal_referee']['sources']['0x0209'],{})
        self.assertEqual(result['evaluation_ground_truth']['robot'],{})
        self.assertIsNone(result['evaluation_metrics']['capture_seconds'])


if __name__=='__main__':unittest.main()
