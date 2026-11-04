"""The copied Arena web application runs against this workspace's engine and assets."""
import base64
import importlib.util
import json
import io
import zipfile
from pathlib import Path
import struct
import subprocess
import tempfile
import time
import unittest

from fastapi.testclient import TestClient
from rmuc2027_core.models import DEFAULT_TEAM_LINEUP
from rmuc2027_core.submissions import load_submission
from rmuc2027_research.web.paths import ROOT, STRATEGIES
from rmuc2027_research.web.server import create_app


class WebArenaTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.app = create_app('trusted-local', self.root/'submissions', self.root/'store',
                              test_mode=True, workspace_registry=self.root/'registry.json')
        options = {'use_uvloop': True} if importlib.util.find_spec('uvloop') else {}
        self.client = TestClient(self.app, backend_options=options)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def wait(self, predicate, seconds=15):
        deadline = time.monotonic()+seconds
        while time.monotonic()<deadline:
            value = predicate()
            if value:
                return value
            time.sleep(.03)
        self.fail(f'timed out; state error={self.app.state.arena.error}')

    def post(self, path, body):
        response = self.client.post(path, json=body)
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def test_original_pages_catalog_and_only_2027_resources(self):
        for path in ['/api/health','/api/maps','/api/models','/api/catalog','/api/robot-catalog',
                     '/api/strategies','/api/custom-models','/api/deck-presets','/api/matches',
                     '/api/leaderboard','/api/battle-settings','/api/agent-guide','/api/showcase-guide']:
            self.assertEqual(self.client.get(path).status_code, 200, path)
        health = self.client.get('/api/health').json()
        self.assertEqual(set(health['seasons']), {'2027'})
        hardware = self.client.get('/api/models').json()['models']
        self.assertEqual(len(hardware), 18)
        self.assertEqual(len(self.client.get('/api/strategies').json()), 94)
        for model in hardware.values():
            self.assertTrue((ROOT/'assets/web'/model['asset'].lstrip('/')).is_file())
        if (ROOT/'build/arena-web/index.html').exists():
            self.assertEqual(self.client.get('/').status_code, 200)
            self.assertEqual(self.client.get('/assets/map-rmuc2027.glb').status_code, 200)
        self.assertEqual(self.client.post('/api/start', json={'map':'rmuc2026'}).status_code, 422)
        self.assertEqual(self.client.get('/api/battle-settings?season=2026').status_code, 422)

    def test_live_standard_match_websocket_history_and_replay(self):
        self.post('/api/start', {'duration':1})
        self.wait(lambda: not self.app.state.arena.busy)
        self.assertIsNone(self.app.state.arena.error)
        with self.client.websocket_connect('/api/ws') as socket:
            state = socket.receive_json()['state']
            self.assertGreater(state['frame']['tick'], 0)
            self.assertEqual(state['frame']['status'], 'finished')
            self.assertEqual(state['frame']['arena']['name'], 'rmuc2027')
        record = self.client.get('/api/matches').json()[0]
        self.assertTrue(record['has_replay'])
        replay = self.client.get('/api/replays/'+record['id']).json()
        self.assertEqual(replay['format'], 'rmuc-replay-v4')
        self.assertEqual(replay['metadata']['rules_hash'], state['frame']['rules_hash'])
        self.assertGreater(len(replay['actions']), 0)
        self.assertEqual(self.client.delete('/api/matches/'+record['id']).status_code, 200)
        self.assertEqual(self.client.get('/api/matches').json(), [])

    def test_intro_pause_skip_speed_stop_and_duplicate_start(self):
        service = self.app.state.arena
        service.prematch_stages = (('self_check', 2), ('countdown', .2))
        self.post('/api/start', {'duration':10})
        self.wait(lambda: service.prematch)
        self.assertEqual(service.frame['tick'], 0)
        self.assertEqual(self.client.post('/api/start',json={'duration':1}).status_code, 409)
        self.post('/api/control', {'command':'pause'})
        before = service.prematch['remaining']
        time.sleep(.15)
        self.assertEqual(service.prematch['remaining'], before)
        self.post('/api/control', {'command':'pause'})
        self.post('/api/control', {'command':'skip_prematch'})
        self.post('/api/control', {'command':'speed','speed':8})
        self.wait(lambda: service.frame['tick']>0)
        self.post('/api/control', {'command':'stop'})
        self.wait(lambda: not service.busy)
        self.assertEqual(service.frame['reason'], 'stopped')
        self.assertIsNone(service.error)

    def test_free_battle_models_and_team_controller(self):
        self.post('/api/start', {'mode':'deck','duration':1,
            'red_deck':['claude_opus55_xhigh_2027_adaptive_2731'],
            'blue_deck':['claude_opus55_xhigh_2027_adaptive_2732'],
            'red_control':'team','red_team_strategy':'claude_opus55_xhigh_2027_team'})
        self.wait(lambda: not self.app.state.arena.busy)
        state = self.app.state.arena.state()
        self.assertIsNone(state['error'])
        self.assertEqual([r['model_slot'] for r in state['frame']['robots']], [2731,2732])
        self.assertEqual(state['metadata']['teams']['red']['controller'], 'team')
        self.assertEqual(state['metadata']['teams']['blue']['controller'], 'independent')

    def test_lineup_mismatch_is_rejected(self):
        lineup = {**DEFAULT_TEAM_LINEUP, '3':301}
        response = self.client.post('/api/start',json={'duration':1,'team_lineup':lineup})
        self.assertEqual(response.status_code, 409)
        self.assertFalse(self.app.state.arena.busy)

    def test_battle_settings_save_restore_and_match_metadata(self):
        data = self.client.get('/api/battle-settings').json()
        field = next(f for f in data['fields'] if data['defaults'][f['path']]<f['max'])
        value = min(field['max'],data['defaults'][field['path']]+field['step'])
        self.post('/api/battle-settings', {'season':'2027','values':{field['path']:value}})
        self.post('/api/start',{'duration':1})
        self.wait(lambda: not self.app.state.arena.busy)
        self.assertEqual(self.app.state.arena.metadata['battle_settings'][field['path']],value)
        self.assertTrue(self.app.state.arena.metadata['test_match'])
        self.assertEqual(self.post('/api/battle-settings', {'season':'2027','values':{}})['overrides'],{})

    def test_training_workspaces_practice_and_submission(self):
        workspaces = {}
        for mode in ('team','robot','distributed','adaptive'):
            body = {'name':mode,'mode':mode,'minutes':5,'season':'2027'}
            if mode in ('robot','adaptive'):
                body['robot'] = 3 if mode=='robot' else 2731
            if mode=='team':
                body['team_lineup'] = DEFAULT_TEAM_LINEUP
            workspaces[mode] = self.post('/api/training-workspaces',body)
            path = Path(workspaces[mode]['workspace'])
            self.assertTrue((path/'AGENT_TASK.md').is_file())
            self.assertTrue((path/'sdk.py').is_file())
            self.assertIn('2027',(path/'AGENT_GUIDE.md').read_text())
        task = workspaces['adaptive']
        path = Path(task['workspace'])
        metadata = json.loads((path/'agent.json').read_text())
        headers = {'Authorization':'Bearer '+metadata['training']['token']}
        base = '/api/training/'+task['session_id']
        response = self.client.post(base+'/tests',headers=headers,json={'duration':1,'games':1})
        self.assertEqual(response.status_code,200,response.text)
        job = response.json()['job_id']
        def finished():
            data=self.client.get(base+'/tests/'+job,headers=headers).json()
            return data if data.get('status') in ('finished','error','expired') else None
        result=self.wait(finished)
        self.assertEqual(result['status'],'finished',result)
        self.assertEqual(self.client.get(base+'/tests/'+job+'/replays/0',headers=headers).status_code,200)
        (path/'showcase.json').write_text((STRATEGIES/'claude_opus55_xhigh_2027_team/showcase.json').read_text())
        response=self.client.post(base+'/submit',headers=headers,json={})
        self.assertEqual(response.status_code,200,response.text)
        bundle=self.root/'submissions'/response.json()['entry']
        self.assertEqual(load_submission(bundle)[0]['mode'],'adaptive')
        self.assertIn(response.json()['entry'],self.client.get('/api/robot-catalog').json())
        self.assertEqual(self.client.get(base).status_code,403)

    def test_mixed_online_snapshot_keeps_only_2027_strategies(self):
        from rmuc2027_research.web.strategy_library import unpack_snapshot
        raw=io.BytesIO()
        source=STRATEGIES/'claude_opus55_xhigh_2027_team'
        with zipfile.ZipFile(raw,'w') as archive:
            for path in source.iterdir():
                if path.is_file():
                    archive.writestr('strategies/current/'+path.name,path.read_bytes())
            archive.writestr('strategies/old/manifest.json',json.dumps({'rules_profile':'rmuc-2026-v1.0.0-arena-v3'}))
            archive.writestr('strategies/old/strategy.py','raise RuntimeError("old season")')
        target=self.root/'snapshot'
        self.assertEqual(unpack_snapshot(raw.getvalue(),target,expected_count=2),1)
        self.assertTrue((target/'strategies/current/manifest.json').is_file())
        self.assertFalse((target/'strategies/old').exists())
        with self.assertRaisesRegex(ValueError,'count mismatch'):
            unpack_snapshot(raw.getvalue(),self.root/'bad-count',expected_count=3)

    def test_original_video_conversion_and_match_download(self):
        from rmuc2027_research.web.video_export import ffmpeg_executable
        executable=ffmpeg_executable()
        if not executable:
            self.skipTest('ffmpeg is not installed')
        source=self.root/'sample.webm'
        subprocess.run([executable,'-hide_banner','-loglevel','error','-f','lavfi','-i',
                        'color=c=red:s=96x64:r=10','-t','0.4','-c:v','libvpx','-threads','1',str(source)],
                       check=True,timeout=15,capture_output=True)
        headers={'Content-Type':'video/webm'}
        response=self.client.post('/api/video-export?duration=0.4',headers=headers,content=source.read_bytes())
        self.assertEqual(response.status_code,200,response.text[:150] if response.status_code!=200 else '')
        self.assertEqual(response.content[4:8],b'ftyp')
        self.post('/api/start',{'duration':1})
        self.wait(lambda:not self.app.state.arena.busy)
        match_id=self.app.state.arena.match_id
        response=self.client.post('/api/matches/'+match_id+'/video?duration=0.4',headers=headers,content=source.read_bytes())
        self.assertEqual(response.status_code,200,response.text)
        saved=self.client.get(response.json()['url'])
        self.assertEqual(saved.content[4:8],b'ftyp')
        self.assertTrue(self.client.get('/api/matches').json()[0]['has_video'])

    def test_custom_model_upload_read_delete_and_invalid_video(self):
        document={'asset':{'version':'2.0'},'meshes':[{'primitives':[{'attributes':{'POSITION':0}}]}],
                  'accessors':[{'count':3,'type':'VEC3','componentType':5126}]}
        body=json.dumps(document).encode();body+=b' '*((-len(body))%4)
        glb=b'glTF'+struct.pack('<II',2,len(body)+20)+struct.pack('<II',len(body),0x4E4F534A)+body
        result=self.post('/api/custom-models',{'name':'test model','base_model':2731,'glb_base64':base64.b64encode(glb).decode()})
        self.assertEqual(self.client.get(result['asset']).content,glb)
        self.assertEqual(self.client.delete('/api/custom-models/'+result['id']).status_code,200)
        self.assertEqual(self.client.get('/api/custom-models').json(),[])
        self.assertEqual(self.client.post('/api/video-export',content=b'invalid',headers={'Content-Type':'video/webm'}).status_code,400)


if __name__=='__main__':
    unittest.main()
