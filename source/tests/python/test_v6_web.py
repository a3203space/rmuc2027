"""Research UI contracts use fakes only: no physics, optimization or GPU context."""
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from rmuc2027_research.rl.capabilities import CapabilitySampler
from rmuc2027_research.rl.scenarios import ScenarioSpec
from rmuc2027_research.rl.v6_web import ResearchCatalog, FrozenResearchMatch, file_hash
from rmuc2027_research.web.server import ArenaService, StartRequest, create_app
from rmuc2027_research.web.store import Store


class Fixture:
    def __init__(self, root):
        self.root = Path(root)
        self.directory = self.root/'run-one'
        self.directory.mkdir(parents=True)
        self.case = ScenarioSpec(learner_side='blue')
        self.capabilities = CapabilitySampler(11).sample(self.case.lineups(), 'blue')
        config = dict(policy_version=6, team='red', friendly='friendly',opponent='opponent',
                      controllers={'red:7':dict(kind='learning')}, _v6_sampler_path='/do-not-write',
                      _v6_replay_dir='/do-not-write', recovery_kind='death',decision_window_seconds=5,
                      sentry_guidance_path='/do-not-use',rl_seconds=.1)
        for name in ('friendly','opponent'):
            (self.directory/name).mkdir()
        checkpoint = self.write('main/baseline.zip', b'fake immutable weights')
        self.checkpoint_id = 'baseline-'+checkpoint['sha256'][:12]
        self.manifest = dict(format='rmuc-research-run-v1',run_id='run-one',title='测试研究',
            runtime_config=self.write('main/runtime-config.json',config),
            checkpoints={self.checkpoint_id:dict(**checkpoint,label='baseline',policy_version=6)},
            helpers={role:self.write(role+'/actor.json',dict(role=role)) for role in ('infantry','heavy')},
            scenarios={self.case.scenario_id:self.case.to_dict()},
            capability_profiles={'blue-fixed':dict(**self.write('capabilities.json',self.capabilities),learner_side='blue',label='固定能力')},
            replays={})
        self.save()
        self.catalog=ResearchCatalog(self.root)
        self.request=dict(run_id='run-one',checkpoint_id=self.checkpoint_id,scenario_id=self.case.scenario_id,
                          capability_profile_id='blue-fixed',backend='rapier')

    def write(self, name, value):
        path=self.directory/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(value if isinstance(value,bytes) else json.dumps(value).encode())
        return dict(path=name,sha256=file_hash(path))

    def save(self):
        (self.directory/'run-manifest.json').write_text(json.dumps(self.manifest))

    def replay(self, *, full=True, ranked=False):
        frame=dict(tick=0,time=0,robots=[],projectiles=[],events=[],teams={}) if full else dict(tick=0,time=0,positions=[])
        payload=dict(format='rmuc-replay-v4',metadata=dict(map='rmuc2027',test_match=not ranked),static={},frames=[frame],actions=[],diagnostics=[],result={})
        self.manifest['replays']['episode-000']=dict(**self.write('replays/episode.json',payload),kind='training',backend='warp_kinematic',truncated=True,seconds=12.)
        self.save()


class FakeMatch:
    def __init__(self,*args,**kwargs):
        self.time=0.
        self.status='running'
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def snapshot(self):return dict(time=self.time,tick=round(self.time*20),status=self.status,robots=[])


class FakeEnv:
    def __init__(self,config,record_replay):
        assert record_replay
        self.config=config
        self.closed=False
        battle=SimpleNamespace(match=FakeMatch(),metadata=dict(teams={side:dict(name=side,version=side,mode='team') for side in ('red','blue')}))
        battle.result=lambda:dict(reason='timeout',winner='red',time=battle.match.time,policy_errors=0)
        self.session=SimpleNamespace(battle=battle,config={'rl_seconds':.1},save_replay=lambda p:Path(p).write_text('{}'))
    def reset(self,*,seed,options):
        self.reset_args=(seed,options)
        return {'actor':[0.]},{}
    def step(self,action):
        self.session.battle.match.time=round(self.session.battle.match.time+.1,5)
        return {'actor':[0.]},0.,False,self.session.battle.match.time>=.2,{}
    def close(self):self.closed=True


class FakeModel:
    def __init__(self):self.calls=[]
    def predict(self,observation,*,deterministic):
        self.calls.append(deterministic)
        return [0.]*7,None


class ResearchWebTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture=Fixture(Path(self.temp.name)/'research')

    def test_empty_catalog_and_safe_public_fields(self):
        self.assertEqual(ResearchCatalog(Path(self.temp.name)/'none').list_runs(),dict(runs=[],errors=[]))
        data=self.fixture.catalog.list_runs()
        self.assertEqual(len(data['runs']),1)
        self.assertNotIn('path',json.dumps(data))
        self.assertEqual(data['runs'][0]['scenarios'][0]['lineups']['blue']['7'],2773)

    def test_frozen_runtime_relocates_and_isolates_training(self):
        f=self.fixture
        resolved=f.catalog.resolve_start(f.request,seed=71,duration=20)
        cfg=resolved['config']
        self.assertEqual(cfg['team'],'blue')
        self.assertEqual(cfg['controllers'],{'blue:7':{'kind':'learning'}})
        self.assertEqual(cfg['auxiliary_pool']['infantry'],str(f.directory/'infantry/actor.json'))
        self.assertEqual(cfg['training_capabilities'],f.capabilities)
        self.assertTrue(cfg['evaluation'])
        self.assertFalse(cfg['scenario_sampling'])
        self.assertEqual(cfg['decision_window_seconds'],20)
        for key in ('_v6_sampler_path','_v6_replay_dir','sentry_guidance_path','recovery_kind'):
            self.assertNotIn(key,cfg)
        self.assertEqual(resolved['provenance']['gradient_updates'],0)

    def test_hash_paths_ids_and_model_conflicts_are_rejected(self):
        f=self.fixture
        checkpoint=f.manifest['checkpoints'][f.checkpoint_id]
        original=checkpoint['path']
        for name in ('../../outside.zip','/tmp/anything.zip'):
            checkpoint['path']=name;f.save()
            with self.assertRaises(ValueError):f.catalog.resolve_start(f.request,seed=1,duration=60)
        checkpoint['path']=original;f.save()
        (f.directory/original).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'SHA256'):f.catalog.resolve_start(f.request,seed=1,duration=60)
        f.manifest['checkpoints']={'selected':checkpoint};f.save()
        self.assertFalse(f.catalog.list_runs()['runs'])

    def test_symlink_escape_and_duplicate_ids_fail_closed(self):
        f=self.fixture
        outside=Path(self.temp.name)/'external';outside.write_bytes(b'fake immutable weights')
        (f.directory/'link').symlink_to(outside)
        f.manifest['checkpoints'][f.checkpoint_id]['path']='link';f.save()
        with self.assertRaisesRegex(ValueError,'outside'):f.catalog.resolve_start(f.request,seed=1,duration=60)
        other=f.root/'duplicate';other.mkdir();(other/'run-manifest.json').write_text(json.dumps(f.manifest))
        self.assertFalse(f.catalog.list_runs()['runs'])

    def test_missing_helper_and_wrong_capability_cannot_fall_back(self):
        f=self.fixture
        f.manifest['helpers'].pop('heavy');f.save()
        with self.assertRaisesRegex(ValueError,'Both frozen'):f.catalog.resolve_start(f.request,seed=1,duration=60)
        bad=copy.deepcopy(f.capabilities);bad['blue:7']['mobility']['speed_reference_mps']=1.2
        with self.assertRaisesRegex(ValueError,'protected'):f.catalog._validate_capabilities(bad,f.case)
        bad=copy.deepcopy(f.capabilities);bad['red:3']['model']=303
        with self.assertRaisesRegex(ValueError,'hardware'):f.catalog._validate_capabilities(bad,f.case)

    def test_only_hashed_canonical_test_replays(self):
        f=self.fixture;f.replay()
        row=f.catalog.list_replays('run-one')[0]
        self.assertTrue(row['truncated']);self.assertNotIn('path',row)
        self.assertTrue(f.catalog.replay('run-one','episode-000').is_file())
        f.replay(full=False)
        with self.assertRaisesRegex(ValueError,'full frames'):f.catalog.replay('run-one','episode-000')
        f.replay(ranked=True)
        with self.assertRaisesRegex(ValueError,'Elo'):f.catalog.replay('run-one','episode-000')
        with self.assertRaises(ValueError):f.catalog.replay('run-one','../x')

    def test_adapter_uses_deterministic_ten_hz_and_marks_partial(self):
        f=self.fixture;resolved=f.catalog.resolve_start(f.request,seed=71,duration=20)
        model=FakeModel()
        frozen=FrozenResearchMatch(resolved,env_factory=FakeEnv,model_loader=lambda *a,**k:model)
        self.assertEqual(frozen.step_seconds,.1)
        self.assertEqual(frozen.step()['tick'],2)
        self.assertEqual(frozen.step()['tick'],4)
        self.assertEqual(model.calls,[True,True])
        self.assertTrue(frozen.done)
        frozen.save(Path(self.temp.name)/'saved.json')
        self.assertTrue(frozen.metadata['test_match'])
        self.assertTrue(frozen.metadata['recording']['truncated'])
        self.assertEqual(frozen.metadata['recording']['frame_hz'],20)
        frozen.close();self.assertTrue(frozen.env.closed)

    def test_full_horizon_research_never_awards_elo(self):
        f=self.fixture;resolved=f.catalog.resolve_start(f.request,seed=71,duration=420)
        frozen=FrozenResearchMatch(resolved,env_factory=FakeEnv,model_loader=lambda *a,**k:FakeModel())
        store=Store(Path(self.temp.name)/'journal');store.create('test',frozen.metadata)
        store.finish('test',dict(reason='timeout',winner='red',policy_errors=0))
        self.assertEqual(store.leaderboard(),[])
        frozen.close()

    def test_api_has_no_path_input_and_returns_registered_replay(self):
        f=self.fixture;f.replay()
        with patch('rmuc2027_research.web.server.Match',FakeMatch):
            app=create_app(test_mode=True,data_root=Path(self.temp.name)/'web',research_root=f.root)
        client=TestClient(app)
        self.assertEqual(client.get('/api/research/runs').status_code,200)
        self.assertEqual(client.get('/api/research/runs/run-one/replays/episode-000').status_code,200)
        self.assertEqual(client.get('/api/research/runs/unknown/replays/episode-000').status_code,404)
        payload=dict(mode='research',research=dict(f.request,path='/tmp/model.zip'),duration=60)
        self.assertEqual(client.post('/api/start',json=payload).status_code,422)
        payload['research']=f.request;payload['battle_settings']={}
        self.assertEqual(client.post('/api/start',json=payload).status_code,409)
        self.assertFalse(app.state.arena.busy)

    def test_web_loop_paces_env_step_at_point_one(self):
        f=self.fixture;resolved=f.catalog.resolve_start(f.request,seed=71,duration=20)
        frozen=FrozenResearchMatch(resolved,env_factory=FakeEnv,model_loader=lambda *a,**k:FakeModel())
        service=ArenaService.__new__(ArenaService)
        service.lock=threading.RLock();service.paused=False;service.stopping=False;service.speed=1
        service.revision=0;service.busy=True;service.error=None
        service.store=Store(Path(self.temp.name)/'paced');service.store.create('paced',frozen.metadata)
        with patch('rmuc2027_research.web.server.FrozenResearchMatch',return_value=frozen),patch('rmuc2027_research.web.server.time.monotonic',return_value=10),patch('rmuc2027_research.web.server.time.sleep') as sleep:
            service._run_research('paced',resolved)
        self.assertEqual([call.args[0] for call in sleep.call_args_list],[.1,.1])
        self.assertIsNone(service.error)
        self.assertFalse(service.busy);self.assertTrue(frozen.env.closed)
        self.assertIsNone(service.battle)
        self.assertEqual(service.store.leaderboard(),[])


if __name__=='__main__':unittest.main()
