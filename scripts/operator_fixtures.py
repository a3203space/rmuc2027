"""Produce host-derived HUD fixtures without starting a server."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'packages/core'), str(ROOT / 'apps/server')]
from rmuc2027_core.engine import Match
from rmuc2027_server.projection import project
from rmuc2027_server.catalog import Catalog


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    match = Match()
    try:
        source = copy.deepcopy(match._frame)
        fixtures = {}
        catalog = Catalog()
        configs = {t: {str(slot): {"model": model, "strategy": "baseline"} for slot,model in ((1,2711),(3,2731),(4,2732),(6,2761),(7,2771))} for t in ("red","blue")}
        team_configs = {t: {"mode":"independent","strategy":"baseline"} for t in ("red","blue")}
        for name, slot in [('heavy',1), ('infantry',3), ('infantry4',4), ('hangar',6), ('sentry',7), ('damaged',3), ('contested',3), ('dead',3), ('overheat',3), ('penalty',3), ('resume',3), ('finished',3)]:
            f = copy.deepcopy(source); f.update(status='running', time=65, remaining=355, tick=1300)
            f['teams']['red']['resources'] = 620
            f['teams']['blue']['resources'] = 450
            f['teams']['red']['control'] = 42.5
            r = next(r for r in f['robots'] if r['team']=='red' and r['slot']==slot)
            r.update(last_damage_time=0, last_shot_time=0)
            if name=='damaged':
                f['bases']['red'].update(hp=875,max_hp=1500,shield_hp=80)
                f['outposts']['blue'].update(hp=0,rebuild={'robot':'red-3','elapsed':2,'required':5,'progress':.4})
            if name=='contested':
                f['positions']['red_outpost'].update(progress=2.5,active=['red','blue'])
                f['zones']['red_outpost'].update(occupants=['red-3','blue-7'],contested=True,progress=.5)
            if name=='dead':r.update(alive=False,hp=0,respawn_progress=7,respawn_required=15)
            if name=='overheat':r.update(overheated=True,heat=300)
            if name=='penalty':r['fpv_blocked_until']=80
            if name=='finished':f.update(status='finished',remaining=0,winner='red')
            p = project(f,match.compact_observation('red',[slot]),'red',slot,'operator-fixture')
            p['control'] = {'needs_resume':name=='resume','status':'AI 接管中' if name=='resume' else '手动控制','ack_seq':0,'manual':name!='resume'}
            fixture = {'player':p,'static':{k:f[k] for k in ('arena','robot_models','simulation_assumptions')},'player_id':'operator','room':{'status':f['status'],'match_id':'operator-fixture','players':[{'id':'operator','nickname':'操作手','team':'red','slot':slot,'host':True,'ready':False,'connected':True}]},'operations':[]}
            fixture['room'].update(seat_configs=configs,team_configs=team_configs)
            if name=='finished':fixture['room']['result']={'winner':'red','teams':f['teams'],'robots':[{k:r.get(k) for k in ('id','team','slot','model_slot','design_name','role','kills','damage','hp','max_hp')} for r in f['robots']]}
            fixtures[name] = fixture
        fixtures['_catalog'] = catalog.public()
        fixtures['lobby'] = {'player':None,'player_id':'operator','room':{**fixtures['infantry']['room'],'status':'lobby'},'operations':[]}
        for phase in ('loading','intro','countdown'):
            fixtures[phase] = {'player':None,'player_id':'operator','room':{**fixtures['lobby']['room'],'status':phase,'stage_elapsed':0,'stage_duration':60 if phase=='loading' else 38.3 if phase=='intro' else 5},'static':fixtures['infantry']['static'],'presentation':{k:v for k,v in source.items() if k not in ('arena','robot_models','simulation_assumptions')},'operations':[]}
        (output/'fixtures.json').write_text(json.dumps(fixtures,ensure_ascii=False),encoding='utf-8')
        print(output/'fixtures.json')
    finally:match.close()

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'build/operator-ui');generate(p.parse_args().output)
