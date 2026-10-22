"""Retain complete dependency license texts alongside desktop binaries."""
import json
from pathlib import Path
import shutil
import subprocess


def collect(root, output, *, manifests=None, offline=False, target=None):
    records = {}

    def package(kind, name, version, directory, license_value):
        key = f"{kind}/{name.replace('/', '__')}@{version}"
        if key in records:
            return
        target = output / key
        files = []
        for source in directory.iterdir():
            if source.name.lower().startswith(("license", "licence", "copying", "notice", "copyright")):
                target.mkdir(parents=True, exist_ok=True)
                if source.is_file():
                    shutil.copy2(source, target / source.name)
                elif source.is_dir():
                    shutil.copytree(source, target / source.name, dirs_exist_ok=True)
                files.append(source.name)
        records[key] = {"name": name, "version": version, "license": license_value, "notices": files}

    modules = root / "node_modules"
    pending = [(name, modules) for name in json.loads((root/'package.json').read_text())['dependencies']]
    visited = set()
    while pending:
        name, parent = pending.pop()
        candidate = parent/name/'package.json'
        if not candidate.is_file(): candidate = modules/name/'package.json'
        if not candidate.is_file(): continue
        candidate = candidate.resolve()
        if candidate in visited: continue
        visited.add(candidate)
        info = json.loads(candidate.read_text(encoding='utf-8'))
        package('javascript',info['name'],info['version'],candidate.parent,info.get('license'))
        for child in {**info.get('dependencies',{}),**info.get('optionalDependencies',{})}:
            pending.append((child,candidate.parent/'node_modules'))
    for manifest in manifests or ("Cargo.toml",):
        command = ["cargo", "metadata", "--locked", "--format-version", "1", "--manifest-path", manifest]
        if offline:
            command.append("--offline")
        if target:
            command += ["--filter-platform", target]
        metadata = json.loads(subprocess.check_output(command, cwd=root))
        for info in metadata["packages"]:
            if info["source"] is not None:
                package("rust", info["name"], info["version"], Path(info["manifest_path"]).parent, info.get("license"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "index.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Dependency license records retained: {len(records)}", flush=True)
