"""Complete spawn-to-objective traversal with parked teammates present."""
import math
import unittest
from rmuc2027_core.engine import Match
from rmuc2027_core.sdk import navigate_action, action
from rmuc2027_research.rl.warp_kinematic.arena import ArenaMotion


class NavigationIntegrationTests(unittest.TestCase):
    def test_special_chassis_reach_goal_with_parked_teammates(self):
        for model,slot in [(2731,3),(2732,4),(2773,7),(2714,1)]:
            with self.subTest(model=model):
                self.traverse(model,slot,ArenaMotion())

    def test_fold_sentry_detour_also_works_in_real_physics(self):
        self.traverse(2773,7,None)

    def traverse(self,model,slot,adapter):
        lineups={team:[dict(slot=s,model=model if s==slot else m) for s,m in
                      [(1,2711),(3,2731),(4,2732),(6,2761),(7,2771)]] for team in ['red','blue']}
        states={team:{} for team in lineups};reached=set();final={}
        with Match(lineups=lineups,physics_adapter=adapter,battle_settings={'drones_enabled':False,'timed_assembly':True}) as match:
            for tick in range(800):
                commands={}
                for team in lineups:
                    obs=match.observation(team,[slot]);r=next(r for r in obs['robots'] if r['slot']==slot)
                    goal=(12,7.5) if team=='red' else (16,7.5)
                    fields=navigate_action(r,goal,obs,states[team]);commands[team]={str(slot):action(**fields)}
                    if math.dist((r['x'],r['y']),goal)<.25:reached.add(team)
                    final[team]=(r['x'],r['y'],r['z'])
                if len(reached)==2:break
                match.step(commands)
        self.assertEqual(reached,{'red','blue'},dict(model=model,states=states,positions=final))
