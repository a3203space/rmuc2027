import unittest
from rmuc2027_research.rl.session import load_config,resource
from rmuc2027_research.strategies.battle import Battle


class FixedAssemblyDispatchTests(unittest.TestCase):
    def test_fixed_opening_skips_only_single_heavy_controller(self):
        config=load_config('configs/rl/sentry-mjwarp-hour.json')
        with Battle(resource(config['friendly']),resource(config['opponent']),duration=2,
                    sandbox='trusted-local',timeout=5,battle_settings={'timed_assembly':True,'drones_enabled':False}) as battle:
            calls={}
            for team,slots,policy in battle.policies:
                key=(team,tuple(slots or []));calls[key]=0;original=policy.decide
                def counted(obs,key=key,original=original):
                    calls[key]+=1
                    return original(obs)
                policy.decide=counted
            while battle.match.status!='finished':battle.step()
            self.assertEqual(calls[('red',(1,))],0)
            self.assertGreater(calls[('red',(3,))],0)
            self.assertGreater(calls[('blue',())],0)
            for team in ('red','blue'):
                self.assertFalse(battle.fixed_assembly_controls_slot(team,None))
                robot=next(r for r in battle._latest_frame['robots'] if r['team']==team and r['slot']==1)
                robot['launcher']='unlocked'
                self.assertFalse(battle.fixed_assembly_controls_slot(team,[1]))
            self.assertFalse(battle.diagnostics)
