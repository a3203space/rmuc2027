import json
import builtins
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from rmuc2027_research.rl.v4_bundle import (
    create_bundle, extract_bundle, load_runtime_config, relocate_model_runtime, verify_bundle, preflight,
)


class BundleTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)/'repo'; run = root/'data/run'
        root.mkdir()
        (root/'pyproject.toml').write_text('[project]\nname="fixture"\n')
        (root/'Cargo.toml').write_text('[workspace]\nmembers=[]\n')
        assets = root/'assets/web/assets'; assets.mkdir(parents=True)
        (assets/'robot-2773.glb').write_bytes(b'fixture geometry')
        (root/'assets/manifest.json').write_text('{}')
        cache = root/'research/.cache'; cache.mkdir(parents=True)
        (cache/'compiled-kernel.bin').write_bytes(b'not source')
        (root/'scratch.sqlite').write_bytes(b'not source')
        baseline = root/'research/strategies/examples/base'
        baseline.mkdir(parents=True); (baseline/'strategy.py').write_text('def act(o,s): return {},s\n')
        for stage in ('main','infantry','heavy'):
            folder = run/stage; folder.mkdir(parents=True)
            (folder/'model.zip').write_bytes(b'fake checkpoint; tests do not load neural models')
            (folder/'actor.json').write_text('{}')
            (folder/'resume-state.pt').write_bytes(b'fake rng')
            (folder/'samplers.json').write_text('{}')
        radar = run/'radar-calibration'; radar.mkdir()
        (radar/'classifier.json').write_text('{"version":1}')
        (radar/'rows.npz').write_bytes(b'fake observable calibration rows')
        (radar/'report.json').write_text('{"fit_split":"train"}')
        for name in ('evaluation','radar-calibration-next','selected-export'):
            folder = run/name; folder.mkdir(); (folder/'report.json').write_text('{}')
        for name in ('selection.json','preflight.json','initialization.json'):
            (run/name).write_text('{}')
        cfg = dict(policy_version=4, friendly='examples/base', opponent=str(baseline),
                   auxiliary_pool={role:str(run/role/'actor.json') for role in ('infantry','heavy')},
                   radar_classifier_path=str(radar/'classifier.json'),
                   _training_deadline=1, _episode_log_dir='/old/machine/logs')
        for stage in ('main','infantry','heavy'):
            (run/stage/'experiment.json').write_text(json.dumps({'config':{**cfg,'policy_version':4 if stage=='main' else 5}}))
        guidance = run/'main/guidance'; guidance.mkdir(); (guidance/'demonstrations.npz').write_bytes(b'fake demos')
        dataset = root/'rmuc2026shuj'; dataset.mkdir(); (dataset/'private.sqlite').write_bytes(b'external database')
        return root, run, dataset/'private.sqlite'

    def test_bundle_relative_resources_optimizer_files_and_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, database = self.fixture(directory)
            archive = Path(directory)/'bundle.zip'
            result = create_bundle(run, archive, source_root=root, external_database=database)
            self.assertFalse(result['manifest']['resume_contract']['bitwise_equivalent'])
            self.assertFalse(result['manifest']['external_database']['embedded'])
            with zipfile.ZipFile(archive) as z:
                self.assertIn('source/Cargo.toml',z.namelist())
                self.assertIn('source/pyproject.toml',z.namelist())
                self.assertIn('source/assets/web/assets/robot-2773.glb',z.namelist())
                self.assertIn('source/assets/manifest.json',z.namelist())
                self.assertFalse(any('/.cache/' in name for name in z.namelist()))
                self.assertFalse(any(name.endswith('.sqlite') for name in z.namelist()))
            moved = Path(directory)/'another-machine'
            extract_bundle(archive,moved)
            config = load_runtime_config(moved,'main')
            self.assertTrue(config['friendly'].startswith(str(moved)))
            self.assertEqual(config['auxiliary_pool']['infantry'],str(moved/'models/infantry/actor.json'))
            self.assertEqual(config['radar_classifier_path'],str(moved/'radar-calibration/classifier.json'))
            self.assertTrue((moved/'radar-calibration/rows.npz').is_file())
            self.assertTrue((moved/'radar-calibration/report.json').is_file())
            for name in ('evaluation','radar-calibration-next','selected-export'):
                self.assertTrue((moved/name/'report.json').is_file())
            for name in ('selection.json','preflight.json','initialization.json'):
                self.assertTrue((moved/'run-state'/name).is_file())
            self.assertIn('cargo build --release --manifest-path source/packages/simulation/Cargo.toml',
                          (moved/'RESUME.txt').read_text())
            self.assertNotIn('_training_deadline',config)
            model = SimpleNamespace(sentry_guidance_path='/old/demo.npz',learning_rate_control_path='/old/rate.json',
                                    skill_guidance_path='/old/skill.npz',skill_guidance_datasets=['/old/skill.npz'],
                                    _sentry_guidance_cache={'old':True})
            paths = relocate_model_runtime(model,moved,'main',output_dir=Path(directory)/'next-run/main')
            self.assertIsNone(model.sentry_guidance_path)
            self.assertIsNone(model.skill_guidance_path)
            self.assertIsNone(model.skill_guidance_datasets)
            self.assertTrue((moved/'models/main/guidance/demonstrations.npz').is_file())
            auxiliary = SimpleNamespace(sentry_guidance_path='/old/demo.npz',skill_guidance_path='/old/skill.npz',
                                        skill_guidance_datasets=['/old/skill.npz'])
            relocate_model_runtime(auxiliary,moved,'infantry')
            self.assertIsNone(auxiliary.sentry_guidance_path)
            self.assertIsNone(auxiliary.skill_guidance_path)
            self.assertIsNone(auxiliary.skill_guidance_datasets)
            self.assertFalse(hasattr(model,'_sentry_guidance_cache'))
            self.assertNotIn('/old/',json.dumps(paths))
            verify_bundle(moved)

    def test_legacy_configuration_keeps_relocated_guidance_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root,run,_ = self.fixture(directory)
            path = run/'main/experiment.json'
            metadata = json.loads(path.read_text()); metadata['config']['policy_version'] = 3
            path.write_text(json.dumps(metadata))
            archive = Path(directory)/'bundle.zip'; create_bundle(run,archive,source_root=root)
            moved = Path(directory)/'unpacked'; extract_bundle(archive,moved)
            model = SimpleNamespace(sentry_guidance_path='/old/demo.npz')
            relocate_model_runtime(model,moved,'main')
            self.assertEqual(model.sentry_guidance_path,str(moved/'models/main/guidance/demonstrations.npz'))

    def test_integrity_tampering_and_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root,run,_ = self.fixture(directory)
            archive = Path(directory)/'bundle.zip'
            create_bundle(run,archive,source_root=root)
            with self.assertRaises(FileExistsError):
                create_bundle(run,archive,source_root=root)
            moved = Path(directory)/'unpacked'; extract_bundle(archive,moved)
            (moved/'models/main/model.zip').write_bytes(b'corrupted')
            with self.assertRaises(ValueError):
                verify_bundle(moved)

    def test_preflight_reports_missing_inference_dependency_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root, run, _ = self.fixture(directory)
            archive = Path(directory)/'bundle.zip'
            create_bundle(run, archive, source_root=root)
            moved = Path(directory)/'unpacked'; extract_bundle(archive, moved)
            normal_import = builtins.__import__
            def import_without_sb3(name, *args, **kwargs):
                if name == 'stable_baselines3':
                    raise ImportError('test target machine lacks stable_baselines3')
                return normal_import(name, *args, **kwargs)
            with patch('builtins.__import__', side_effect=import_without_sb3):
                report = preflight(moved)
            self.assertFalse(report['ready'])
            self.assertFalse(report['training_ready'])
            self.assertFalse(report['inference_ready'])
            self.assertFalse(report['native_simulator']['available'])
            self.assertEqual(report['gradient_updates'], 0)
            self.assertEqual(set(report['inference']), {'main', 'infantry', 'heavy'})
            self.assertIn('Missing inference dependency', report['inference']['main']['error'])

    def test_zip_path_escape_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'bad.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('../escape','bad')
            target = Path(directory)/'new'
            with self.assertRaises(ValueError):
                extract_bundle(archive,target)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
