"""Versioned runtime strategies, independent of the research application."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from rmuc2027_core.paths import ROOT, data_path
from rmuc2027_core.models import model_catalog, normalize_team_lineup
from rmuc2027_core.submissions import load_submission


class Catalog:
    def __init__(self):
        self.sources = {}
        self.entries = {}
        self.models = model_catalog()
        self.register('baseline', ROOT / 'packages/core/bots/default', baseline=True)
        for root, prefix in ((ROOT / 'packages/core/bots/catalog', 'builtin'), (data_path('strategies'), 'local')):
            if root.exists():
                for source in sorted(root.iterdir()):
                    if (source / 'manifest.json').is_file():
                        self.register(prefix + ':' + source.name, source)

    def register(self, key, source, baseline=False):
        manifest, _ = load_submission(source)
        profile = json.loads((source / 'profile.json').read_text(encoding='utf-8')) if (source / 'profile.json').is_file() else {}
        self.sources[key] = source
        self.entries[key] = dict(id=key, version=manifest['version'], mode=manifest['mode'], robot=manifest.get('robot'),
            lineup=manifest.get('team_lineup'), name=profile.get('display_name', '通用基线' if baseline else source.name),
            author=profile.get('author') or profile.get('agent', {}).get('model', ''), description=profile.get('description', ''),
            flexible=baseline)

    def public(self):
        return dict(models=self.models, strategies=list(self.entries.values()))

    def compatible(self, key, slot, model):
        e = self.entries.get(key)
        if not e: return False
        if e['flexible']: return True
        if e['mode'] == 'adaptive': return e['robot'] == model
        if e['mode'] == 'robot': return e['robot'] == slot
        return False

    def validate_team(self, key, lineup):
        if key not in self.entries: raise ValueError('策略不存在')
        e = self.entries[key]
        if e['flexible']: return
        if e['mode'] not in ('team', 'squad', 'distributed'): raise ValueError('请选择队伍策略')
        expected = e['lineup'] or normalize_team_lineup()
        if expected != lineup: raise ValueError('队伍策略与当前型号阵容不兼容，请先按策略阵容选择型号')

    def import_bundle(self, files):
        if not files or len(files) > 12: raise ValueError('无效策略包')
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory)
            for name, content in files.items():
                if Path(name).name != name or name not in ('manifest.json', 'profile.json', 'showcase.json', 'strategy.py', 'robot_1.py', 'robot_3.py', 'robot_4.py', 'robot_6.py', 'robot_7.py'):
                    raise ValueError('策略包包含不支持的文件')
                if len(content.encode('utf-8')) > 512 * 1024: raise ValueError('策略文件过大')
                (staged / name).write_text(content, encoding='utf-8')
            manifest, entries = load_submission(staged)
            for entry in entries.values(): compile(entry.read_bytes(), entry.name, 'exec')
            version = manifest['version']
            destination = data_path('strategies') / version
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(staged, destination)
            self.register('local:' + version, destination)
            return self.public()
