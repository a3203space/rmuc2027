"""The LAN product must stay independent of research and removed real-robot stacks."""
import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class ArchitectureTests(unittest.TestCase):
    def test_default_python_dependencies_do_not_install_neural_training(self):
        text = (ROOT / 'apps/server/pyproject.toml').read_text()
        for package in ('torch', 'stable-baselines3', 'gymnasium', 'rmuc2027-research'):
            self.assertNotIn(package, text)

    def test_lan_and_core_do_not_import_research_or_removed_platform(self):
        for folder in ('apps/server', 'packages/core'):
            for path in (ROOT / folder).rglob('*.py'):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    names = [node.module or ''] if isinstance(node, ast.ImportFrom) else [n.name for n in node.names] if isinstance(node, ast.Import) else []
                    for name in names:
                        self.assertFalse(name.startswith(('rmuc2027_research', 'torch', 'gymnasium', 'stable_baselines3')), path)

    def test_only_2027_map_rulebook_and_legal_models_are_shipped(self):
        rules = json.loads((ROOT / 'configs/rules.json').read_text())
        self.assertEqual(set(rules['maps']), {'rmuc2027'})
        self.assertEqual(set(rules['seasons']), {'2027'})
        self.assertEqual([r['slot'] for r in rules['roster']], [1, 3, 4, 6, 7])
        self.assertNotIn('rulebook_2027', rules)  # One canonical rulebook.
        for model in rules['robot_designs'].values():
            self.assertEqual(set(model['seasons']), {'2027'})
            self.assertTrue(model['seasons']['2027']['legal'])
            self.assertIn(model['role'], ('heavy', 'infantry', 'aerial', 'sentry'))

    def test_client_has_no_real_robot_transport_or_legacy_mode(self):
        sources = '\n'.join(p.read_text() for p in (ROOT / 'apps/client').rglob('*') if p.suffix in ('.cpp', '.h', '.qml', '.txt'))
        for symbol in ('GameData', 'NetworkManager', 'ProtocolSimulator', 'Protobuf', 'FFMPEG', 'Paho', '--competition', 'RM26_ENABLE_ARENA'):
            self.assertNotIn(symbol, sources)

        self.assertNotIn('QQuickWidget', sources)
        self.assertTrue((ROOT / 'apps/client/web/OperatorHud.tsx').is_file())
        for record in json.loads((ROOT / 'docs/legal/client-assets.json').read_text()):
            import hashlib
            data = (ROOT / record['path']).read_bytes()
            self.assertEqual(data[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(hashlib.sha256(data).hexdigest(), record['sha256'])

    def test_every_shipped_model_and_drone_asset_exists(self):
        rules = json.loads((ROOT / 'configs/rules.json').read_text())
        for model in rules['robot_designs'].values():
            self.assertTrue((ROOT / 'assets/web' / model['asset'].lstrip('/')).is_file(), model['asset'])
            if model['role'] == 'aerial':
                drone = model['asset'].replace('/robot-', '/drone-')
                self.assertTrue((ROOT / 'assets/web' / drone.lstrip('/')).is_file(), drone)
        arena = rules['maps']['rmuc2027']
        for asset in [arena['asset'], *arena.get('structure_assets', {}).values()]:
            self.assertTrue((ROOT / 'assets/web' / asset.lstrip('/')).is_file(), asset)

    def test_old_product_directories_and_entrypoints_are_gone(self):
        for name in ('Arena-1', 'RM26_Client', 'legacy', 'desktop', 'sim'):
            self.assertFalse((ROOT / name).exists())
        for path in (ROOT / 'apps').rglob('*.py'):
            self.assertNotIn('from rmuc_arena', path.read_text())


if __name__ == '__main__':
    unittest.main()
