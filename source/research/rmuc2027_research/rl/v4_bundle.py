"""Relocatable v4 checkpoints, source snapshots and offline preparation.

Packages are integrity-checked, not advertised as bitwise hardware-independent
replays. Resume only after a completed PPO update; discard partial rollouts and
restart episodes on a hardware/backend transfer. This module never trains.
"""
import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import subprocess
import sys
import zipfile

from .offline2026 import file_hash
from .scenarios import canonical_hash

VERSION = 1
STAGES = ("main", "infantry", "heavy")
EXCLUDED_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".cache",
                 ".mypy_cache", ".ruff_cache", ".hypothesis", ".tox", "target", "build", "dist",
                 ".git-backup", ".git-backups", "git-backups", "github-backup", "github-backups"}
EXCLUDED_SOURCE_SUFFIXES = (".sqlite", ".sqlite3", ".sqlite-wal", ".sqlite-shm", ".db", ".db-wal", ".db-shm")
RUNTIME_MODEL_PATHS = ("sentry_guidance_path", "skill_guidance_path", "learning_rate_control_path", "update_log_path")


def _relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Bundle paths must be safe relative POSIX paths")
    return path


def _walk(directory, *, source=False):
    directory = Path(directory)
    for base, dirs, names in os.walk(directory):
        relative = Path(base).relative_to(directory)
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRS and not Path(base, d).is_symlink()
                         and not (source and d.endswith((".egg-info", ".dist-info")))
                         and not (source and relative == Path('.') and d in {"data", "rmuc2026shuj"}))
        for name in sorted(names):
            path = Path(base, name)
            if path.is_symlink() or name.endswith((".pyc", ".tmp", ".partial")) or name in {".DS_Store", ".env"}:
                continue
            if name.lower().endswith(EXCLUDED_SOURCE_SUFFIXES) or name.endswith('.bundle'):
                continue
            yield path, path.relative_to(directory).as_posix()


def _resolve_submission(reference, source_root):
    path = Path(reference)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "examples":
        return source_root / "research/strategies" / path
    return source_root / path


def create_bundle(run_dir, output, *, preparation=None, source_root=None, external_database=None):
    """Copy completed stages and resources into a new ZIP with relative manifest paths.

    Call this after stopping updates/checkpoint writes. File hashes are checked
    again after packaging to reject a snapshot that changed while being copied.
    """
    if source_root is None:
        from rmuc2027_core.paths import ROOT
        source_root = ROOT
    source_root, run_dir, output = Path(source_root).resolve(), Path(run_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = {}; generated = {}; stage_info = {}

    def include(path, destination):
        destination = _relative(destination).as_posix()
        if destination in entries and entries[destination] != path:
            raise ValueError("Duplicate bundle destination")
        entries[destination] = Path(path)

    for path, relative in _walk(source_root, source=True):
        # Do not accidentally recursively archive an output under the source tree.
        if path.resolve() == output or run_dir == path or run_dir in path.parents:
            continue
        include(path, "source/" + relative)
    radar_directory = run_dir/"radar-calibration"
    if radar_directory.is_dir():
        for path, relative in _walk(radar_directory):
            include(path, "radar-calibration/"+relative)
    for stage in STAGES:
        directory = run_dir/stage
        checkpoint = directory/"model.zip"
        if not checkpoint.exists():
            checkpoint = directory/"initialized.zip"
        if not checkpoint.exists():
            raise ValueError(f"Missing {stage} model.zip/initialized.zip")
        for path, relative in _walk(directory):
            include(path, f"models/{stage}/{relative}")
        metadata_path = directory/"experiment.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        config = copy.deepcopy(metadata.get("config", {}))
        bindings = {}
        references = [(key, config.get(key)) for key in ("friendly", "opponent")]
        references += [("controller:"+key, spec.get("submission")) for key, spec in config.get("controllers", {}).items()]
        for key, reference in references:
            if not reference:
                continue
            path = _resolve_submission(reference, source_root).resolve()
            if not path.is_dir():
                raise ValueError(f"Missing frozen policy resource for {stage}:{key}")
            try:
                destination = "source/" + path.relative_to(source_root).as_posix()
                # Source resources under excluded directories must be explicitly included.
                if not any(name.startswith(destination + "/") for name in entries):
                    raise ValueError
            except ValueError:
                members = list(_walk(path))
                digest = canonical_hash([(rel, file_hash(p)) for p, rel in members])[:20]
                destination = "frozen-pool/" + digest
                for resource, relative in members:
                    include(resource, destination+"/"+relative)
            bindings[key] = destination
        for role, reference in config.get("auxiliary_pool", {}).items():
            path = _resolve_submission(reference, source_root).resolve()
            if not path.is_file():
                raise ValueError(f"Missing auxiliary actor resource: {role}")
            try:
                relative = path.relative_to(run_dir)
                if relative.parts[0] not in STAGES:
                    raise ValueError
                destination = "models/" + relative.as_posix()
            except ValueError:
                destination = "frozen-pool/actors/" + file_hash(path)[:20] + "-" + path.name
            include(path, destination)
            bindings["auxiliary_pool:"+role] = destination
        radar_reference = config.get("radar_classifier_path")
        if radar_reference:
            path = _resolve_submission(radar_reference, source_root).resolve()
            if not path.is_file():
                raise ValueError(f"Missing radar classifier resource for {stage}")
            try:
                destination = "radar-calibration/" + path.relative_to(radar_directory).as_posix()
            except ValueError:
                members = list(_walk(path.parent)) if path.parent.name == "radar-calibration" else [(path, path.name)]
                digest = canonical_hash([(relative, file_hash(resource)) for resource, relative in members])[:20]
                directory = "frozen-pool/radar/"+digest
                for resource, relative in members:
                    include(resource, directory+"/"+relative)
                destination = directory+"/"+path.name
            include(path, destination)
            bindings["radar_classifier_path"] = destination
        if config.get('policy_version') == 6:
            sampler = run_dir/'sampler-coordinator.json'
            if not sampler.is_file():
                raise ValueError('v6 bundle requires the frozen sampler-coordinator.json')
            include(sampler, 'run-state/sampler-coordinator.json')
            bindings['_v6_sampler_path'] = 'run-state/sampler-coordinator.json'
        # Portable runtime config has no active machine-local deadlines/log paths.
        for key in ("_training_deadline", "_episode_log_dir", "_v6_replay_dir", "_worker_id", "github_backup_repo"):
            config.pop(key, None)
        for key, destination in bindings.items():
            if key.startswith("controller:"):
                config["controllers"][key.split(":", 1)[1]]["submission"] = destination
            elif key.startswith("auxiliary_pool:"):
                config["auxiliary_pool"][key.split(":", 1)[1]] = destination
            else:
                config[key] = destination
        config_name = f"models/{stage}/runtime-config.json"
        generated[config_name] = (json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode()
        stage_info[stage] = dict(checkpoint=f"models/{stage}/{checkpoint.name}", config=config_name,
                                 policy_version=config.get('policy_version', 1),
                                 resource_bindings=bindings,
                                 rng_state=f"models/{stage}/resume-state.pt" if (directory/"resume-state.pt").exists() else None,
                                 sampler_state=f"models/{stage}/samplers.json" if (directory/"samplers.json").exists() else None)
    for name in ("resume-state.pt", "samplers.json", "campaign.json", "budget.json", "selection.json", "preflight.json", "initialization.json",
                 "requirements-runtime.txt", "runtime-environment.json", "intervention-hold.json", "interventions.json"):
        if (run_dir/name).is_file():
            include(run_dir/name, "run-state/"+name)
    for dirname in ("evaluation", "radar-calibration-next", "selected-export", "aux-intervention",
                    "radar-calibration-stratified-candidate", "radar-calibration-next-analysis", "report-tools", "review", "review-notes", "native-viewer-smoke"):
        if (run_dir/dirname).is_dir():
            for path, relative in _walk(run_dir/dirname):
                include(path, dirname+"/"+relative)
    for dirname in ("frozen-pool", "frozen_pool", "opponent-pool"):
        if (run_dir/dirname).is_dir():
            for path, relative in _walk(run_dir/dirname):
                include(path, "frozen-pool/run/"+relative)
    # New campaign reports, manifests, recovery probes and both kinds of
    # replay must not depend on a hand-maintained list of directory names.
    # Model ZIPs remain inside models/<stage>; never recursively package an
    # older release ZIP or the outside release index (which names this ZIP).
    for path, relative in _walk(run_dir):
        parts = PurePosixPath(relative).parts
        if parts[0] in STAGES or parts[0] in ('radar-calibration', 'frozen-pool', 'frozen_pool', 'opponent-pool'):
            continue
        if path.resolve() == output or path.suffix.lower() == '.zip' or path.name == 'release-index.json':
            continue
        include(path, ('run-state/' if len(parts) == 1 else '') + relative)
    preparation_info = None
    if preparation is not None:
        preparation = Path(preparation).resolve()
        prepared = json.loads((preparation/"manifest.json").read_text())
        for path, relative in _walk(preparation):
            include(path, "preparation/"+relative)
        preparation_info = dict(manifest="preparation/manifest.json", split_hash=prepared["split_hash"],
                                source_sha256=prepared["source"]["sha256"], database_embedded=False)
    external = None
    if external_database is not None:
        database = Path(external_database)
        external = dict(filename=database.name, bytes=database.stat().st_size, sha256=file_hash(database), embedded=False)
    elif preparation_info:
        external = dict(filename=prepared["source"]["filename"], bytes=prepared["source"]["bytes"],
                        sha256=prepared["source"]["sha256"], embedded=False)
    is_v6 = stage_info['main']['policy_version'] == 6
    bundle_module = 'v6_bundle' if is_v6 else 'v4_bundle'
    generated["preflight-ubuntu24.sh"] = ('''#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
python_bin="${PYTHON_BIN:-python3}"
export PYTHONPATH="$bundle_dir/source/packages/core:$bundle_dir/source/research:$bundle_dir/source/apps/server${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -m rmuc2027_research.rl.''' + bundle_module + ''' preflight "$bundle_dir" "$@"
''').encode()
    generated["RESUME.txt"] = ("Install the pinned source dependencies on the target machine, then run:\n"
        "  cargo build --release --manifest-path source/packages/simulation/Cargo.toml\n"
        "  PYTHON_BIN=/path/to/venv/bin/python bash preflight-ubuntu24.sh\n"
        "Run these commands from the extracted bundle root. Rust build outputs are intentionally excluded.\n"
        "Preflight distinguishes inference_ready from training_ready; training also needs native simulation and all backend dependencies.\n"
        "Do not reuse an in-flight PPO rollout or simulator state after hardware/backend transfer.\n"
        "Restore model/critic/optimizer counters, RNG and samplers; start fresh episodes.\n"
        "Missing stage RNG/sampler state is reported, never silently called exact resume.\n"
        "Inference and throughput must be checked locally; no GPU memory or speed is assumed.\n"
        "2026 SQLite is external and verified by hash. Preparation masks unknown actions and unconfirmed mappings.\n").encode()
    if is_v6:
        generated['RESUME.txt'] += (
            '\nV6 main must load with rmuc2027_research.rl.v6_ppo.HybridPPO.load(..., device="cpu").\n'
            'Do not load it with PPO or AsymmetricPPO. Infantry/heavy remain auxiliary v5 PPO.\n'
            'Restore tactical/revive/critic optimizers, counters, RNG and sampler coordinator as one checkpoint set.\n'
            'The actor has 775 legal inputs, critic 454 private inputs, revive_mask 3 and transport action 7.\n'
            'Skills and fold/jump standard deviations stay frozen; start a NEW rollout and NEW episodes.\n'
            'load_runtime_config resolves _v6_sampler_path under run-state; copy that coordinator to a new run\n'
            'before training, so source bundle files and their integrity hashes remain immutable.\n'
            'Target requested: Ubuntu 24, Ryzen 5600X, RTX 5070 Ti. THIS PACKAGE DOES NOT CLAIM TARGET VALIDATION.\n'
            'Use CPU Torch for PPO and actual Warp CUDA for physics; do not require Torch CUDA or silently use CPU physics.\n'
            'The current campaign measures actual Warp 1/2 environments and Rapier 1 before choosing parallelism;\n'
            '4/8-environment scaling is a later explicit benchmark, not a claimed result of this campaign.\n'
            'measure wall time, memory and failures under the real concurrent workload. No fixed throughput is assumed.\n'
            'This bundle preflight checks inference/integrity; it is not a substitute for that physical throughput test.\n'
            'run-results.json describes finite evidence/censoring. replays/training and replays/evaluation are recorded\n'
            'web replays; there is no live training stream. Release ZIP SHA, relocated verification and Git records\n'
            'belong in the OUTSIDE release-index.json, never inside a self-referential ZIP.\n'
            '\nConcrete target continuation (run from the extracted bundle root; no automatic run by extraction):\n'
            '  python3 -m venv .venv\n'
            '  .venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r run-state/requirements-runtime.txt\n'
            '  cargo build --release --manifest-path source/packages/simulation/Cargo.toml\n'
            '  export PYTHONPATH="$PWD/source/packages/core:$PWD/source/research:$PWD/source/apps/server"\n'
            '  export RMUC_RESOURCE_ROOT="$PWD/source"\n'
            '  PYTHON_BIN="$PWD/.venv/bin/python" bash preflight-ubuntu24.sh\n'
            '  .venv/bin/python -m rmuc2027_research.rl.v6_campaign run --resume-bundle "$PWD" --output /absolute/path/to/new-v6-round\n'
            'The run command starts a NEW bounded two-hour experiment, including physical load measurement.\n'
            'Resume materialization preserves all three optimizer states and RNG/sampler history, releases only\n'
            'interrupted leases, and makes the restored current model the new paired baseline. It never resumes\n'
            'an unfinished rollout, replaces weights with an older selected model, or silently retrains helpers.\n').encode()
    # Generated runtime bindings take precedence over an older copied runtime config.
    for name in generated:
        entries.pop(name, None)
    file_records = {name: dict(sha256=file_hash(path), bytes=path.stat().st_size) for name, path in entries.items()}
    file_records.update({name: dict(sha256=hashlib.sha256(value).hexdigest(), bytes=len(value)) for name, value in generated.items()})
    manifest = dict(version=VERSION, format="rmuc2027-sentry-v6-resume" if is_v6 else "rmuc2027-sentry-v4-resume", stages=stage_info,
                    preparation=preparation_info, external_database=external, files=file_records,
                    resume_contract=dict(bitwise_equivalent=False, checkpoint_boundary="after_completed_PPO_update",
                        discard_partial_rollout=True, reset_episodes=True, restore_optimizer=True,
                        restore_rng_if_compatible=True, normalization="restore stage state; explicit none if unused",
                        hardware_throughput="measure on destination; no preselected parallelism"),
                    source_revision_note="Snapshot of working tree, including authorized uncommitted changes")
    pending = output.with_suffix(output.suffix+".partial")
    try:
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for name, path in sorted(entries.items()):
                archive.write(path, name)
            for name, content in sorted(generated.items()):
                archive.writestr(name, content)
            archive.writestr("bundle-manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
        for name, path in entries.items():
            if file_hash(path) != file_records[name]["sha256"]:
                raise RuntimeError("A bundle input changed during packaging: "+name)
        os.replace(pending, output)
    except BaseException:
        pending.unlink(missing_ok=True)
        raise
    return dict(path=str(output), sha256=file_hash(output), bytes=output.stat().st_size, manifest=manifest)


def verify_bundle(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/"bundle-manifest.json").read_text())
    if manifest.get("version") != VERSION or manifest.get("format") not in ("rmuc2027-sentry-v4-resume", "rmuc2027-sentry-v6-resume"):
        raise ValueError("Unsupported bundle version")
    for name, record in manifest["files"].items():
        path = directory / _relative(name)
        if path.is_symlink() or not path.is_file() or directory not in path.resolve().parents:
            raise ValueError("Missing/unsafe bundle member: "+name)
        if path.stat().st_size != record["bytes"] or file_hash(path) != record["sha256"]:
            raise ValueError("Bundle member integrity mismatch: "+name)
    return manifest


def extract_bundle(archive, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    with zipfile.ZipFile(archive) as zipped:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP entries")
        for info in zipped.infolist():
            _relative(info.filename)
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Symlinks are not accepted in resume bundles")
        manifest = json.loads(zipped.read("bundle-manifest.json"))
        if set(names) != set(manifest["files"]) | {"bundle-manifest.json"}:
            raise ValueError("ZIP entries do not match the manifest")
        destination.mkdir(parents=True)
        for name in names:
            path = destination/_relative(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(name) as src, path.open("wb") as dst:
                while chunk := src.read(4 * 1024 * 1024):
                    dst.write(chunk)
    return verify_bundle(destination)


def load_runtime_config(directory, stage):
    directory = Path(directory).resolve()
    manifest = json.loads((directory/"bundle-manifest.json").read_text())
    if stage not in STAGES:
        raise ValueError("Unknown stage")
    info = manifest["stages"][stage]
    config = json.loads((directory/_relative(info["config"])).read_text())
    for key, reference in info["resource_bindings"].items():
        resolved = str(directory/_relative(reference))
        if key.startswith("controller:"):
            config["controllers"][key.split(":", 1)[1]]["submission"] = resolved
        elif key.startswith("auxiliary_pool:"):
            config["auxiliary_pool"][key.split(":", 1)[1]] = resolved
        else:
            config[key] = resolved
    for key in ('_training_deadline', '_episode_log_dir', '_v6_replay_dir', '_worker_id'):
        config.pop(key, None)
    return config


def relocate_model_runtime(model, directory, stage, *, output_dir=None):
    """Rebind loaded SB3 attributes; never follow paths embedded by another host."""
    root = Path(directory).resolve(); stage_dir = root/"models"/stage
    if stage not in STAGES or not stage_dir.is_dir():
        raise ValueError("Unknown bundled stage")
    output = Path(output_dir).resolve() if output_dir is not None else stage_dir
    bindings = {}
    config = load_runtime_config(root, stage)
    initialization_only_guidance = config.get("policy_version") in (4, 5, 6)
    for attribute, candidates in (("sentry_guidance_path", ("guidance/demonstrations.npz", "demonstrations.npz")),
                                  ("skill_guidance_path", ("guidance/demonstrations.npz", "demonstrations.npz"))):
        selected = None if initialization_only_guidance else next((stage_dir/name for name in candidates if (stage_dir/name).is_file()), None)
        # Explicit None disables missing guidance instead of accessing the old host.
        setattr(model, attribute, str(selected) if selected else None)
        bindings[attribute] = str(selected) if selected else None
    if initialization_only_guidance:
        # The v4 main and current auxiliary policies use BC only at initialization.
        # Artifacts remain packaged for provenance; PPO must not reopen them.
        model.skill_guidance_datasets = None
        bindings["skill_guidance_datasets"] = None
    for attribute, name in (("learning_rate_control_path", "learning-rate.json"), ("update_log_path", "updates.jsonl")):
        setattr(model, attribute, str(output/name)); bindings[attribute] = str(output/name)
    for attribute in ("_sentry_guidance_cache", "_skill_guidance_cache"):
        if hasattr(model, attribute):
            delattr(model, attribute)
    for attribute in ('_training_deadline', 'training_deadline', '_episode_log_dir', '_v6_replay_dir', '_worker_id'):
        if hasattr(model, attribute):
            delattr(model, attribute)
    if config.get('policy_version') == 6:
        model._v6_sampler_path = config['_v6_sampler_path']
        bindings['_v6_sampler_path'] = model._v6_sampler_path
    return bindings


def preflight(directory, *, inference=True):
    """Local integrity/device/inference checks only; no network or training."""
    directory = Path(directory).resolve()
    manifest = verify_bundle(directory)
    os_release = {}
    if Path("/etc/os-release").is_file():
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                key, value = line.split("=", 1); os_release[key] = value.strip('"')
    report = dict(integrity=True, system=platform.system(), python=platform.python_version(), os_release=os_release,
                  ubuntu24=os_release.get("ID") == "ubuntu" and os_release.get("VERSION_ID", "").startswith("24."),
                  dependencies={}, cuda_available=False, cuda_devices=[], inference={}, gradient_updates=0,
                  dependency_imports={}, warp_cuda_available=False, warp_cuda_devices=[],
                  policy_device="cpu", torch_cuda_available=False, torch_cuda_devices=[], physics_device=None,
                  bitwise_resume_claimed=False, reset_episodes=True, discard_partial_rollout=True,
                  missing_resume_state=[stage for stage, row in manifest["stages"].items() if not row["rng_state"] or not row["sampler_state"]])
    for package in ("torch", "numpy", "stable-baselines3", "gymnasium", "warp-lang", "mujoco", "mujoco-warp"):
        try:
            report["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][package] = None
    for package, module in (("torch","torch"), ("numpy","numpy"), ("stable-baselines3","stable_baselines3"),
                            ("gymnasium","gymnasium"), ("warp-lang","warp"), ("mujoco","mujoco"), ("mujoco-warp","mujoco_warp")):
        try:
            importlib.import_module(module)
            report["dependency_imports"][package] = dict(available=True)
        except Exception as error:
            report["dependency_imports"][package] = dict(available=False, error=str(error))
    build_command = "cargo build --release --manifest-path source/packages/simulation/Cargo.toml"
    native = directory/"source/target/release"/("rmuc-sim.exe" if os.name == "nt" else "rmuc-sim")
    native_report = dict(path=native.relative_to(directory).as_posix(), available=False, build_command=build_command)
    if native.is_file() and os.access(native, os.X_OK):
        try:
            environment = dict(os.environ)
            environment.pop("RMUC_EXTERNAL_PHYSICS", None); environment.pop("RMUC_KINEMATIC_PHYSICS", None)
            result = subprocess.run([str(native)], input=json.dumps(dict(op="init", seed=42, duration=1, map="rmuc2027"))+"\n",
                                    capture_output=True, text=True, timeout=15, env=environment)
            frame = json.loads(result.stdout.splitlines()[0])
            rules_hash = file_hash(directory/"source/configs/rules.json")
            native_report.update(available=result.returncode == 0 and frame.get("rules_hash") == rules_hash,
                                 exit_code=result.returncode, rules_hash_match=frame.get("rules_hash") == rules_hash)
            if frame.get("error"): native_report["error"] = frame["error"]
        except Exception as error:
            native_report["error"] = str(error)
    else:
        native_report["error"] = "Build the native simulator from the packaged source before resuming"
    report["native_simulator"] = native_report
    try:
        import torch
        report["torch_cuda_available"] = torch.cuda.is_available()
        report["torch_cuda_version"] = torch.version.cuda
        if report["torch_cuda_available"]:
            for i in range(torch.cuda.device_count()):
                properties = torch.cuda.get_device_properties(i)
                report["torch_cuda_devices"].append(dict(index=i, name=properties.name, memory_bytes=properties.total_memory,
                                                    capability=list(torch.cuda.get_device_capability(i))))
    except Exception as error:
        report["torch_error"] = str(error)
    try:
        import warp as wp
        wp.init()
        for device in wp.get_cuda_devices():
            report["warp_cuda_devices"].append(dict(device=str(device), name=device.name,
                architecture=device.arch, memory_bytes=device.total_memory, is_cuda=device.is_cuda))
        report["warp_cuda_available"] = bool(report["warp_cuda_devices"] and wp.get_device("cuda:0").is_cuda)
        report["physics_device"] = "cuda:0" if report["warp_cuda_available"] else None
        report["cuda_available"] = report["warp_cuda_available"]
        report["cuda_devices"] = report["warp_cuda_devices"]
    except Exception as error:
        report["warp_error"] = str(error)
    if inference:
        try:
            import numpy as np
            from stable_baselines3 import PPO
        except ImportError as error:
            report["inference"] = {stage: dict(finite=False, error="Missing inference dependency: "+str(error)) for stage in STAGES}
            report["ready"] = False
            report["inference_ready"] = False; report["training_ready"] = False
            report["resume_state_complete"] = not report["missing_resume_state"]
            return report
        for stage, info in manifest["stages"].items():
            try:
                config = load_runtime_config(directory, stage)
                algorithm = PPO
                if config.get("policy_version") == 4 and stage == "main":
                    from .asymmetric_ppo import AsymmetricPPO
                    algorithm = AsymmetricPPO
                elif config.get('policy_version') == 6 and stage == 'main':
                    from .v6_ppo import HybridPPO
                    algorithm = HybridPPO
                model = algorithm.load(directory/_relative(info["checkpoint"]), device="cpu")
                relocate_model_runtime(model, directory, stage)
                observation = model.observation_space.sample()
                # Independently sampling each categorical mask bit can produce
                # an impossible empty legal-action set. Inference preflight
                # uses a finite synthetic observation with all heads enabled;
                # actual environment masks are checked by the native tests.
                if isinstance(observation, dict) and 'action_mask' in observation:
                    observation['action_mask'] = np.ones_like(observation['action_mask'])
                if config.get('policy_version') == 6:
                    from .sentry_v6 import v6_feature_masks
                    observation['revive_mask'] = np.asarray(v6_feature_masks(observation['actor']), np.float32)
                action, _ = model.predict(observation, deterministic=True)
                row = dict(finite=bool(np.isfinite(action).all()), action_shape=list(np.asarray(action).shape),
                           algorithm=algorithm.__name__)
                actor_path = directory/"models"/stage/"actor.json"
                if actor_path.is_file():
                    payload = json.loads(actor_path.read_text())
                    if payload.get("format") in ("rmuc-aux-actor-v1", "rmuc-aux-actor-v2"):
                        from .aux_policy import PortableAuxActor
                        from .privileged_aux import schema
                        portable = PortableAuxActor(payload)
                        shape = schema()
                        actor_action, _ = portable.predict(dict(actor=np.zeros(shape["obs_size"], np.float32),
                                                                 action_mask=np.ones(shape["mask_size"], np.float32)))
                    elif payload.get("version") == 4 and "tactical" in payload:
                        from . import sentry_v4
                        actor_action = sentry_v4.portable_forward([0.] * sentry_v4.schema()["obs_size"], payload)
                    elif payload.get('version') == 6 and 'revive' in payload:
                        from . import sentry_v6
                        actor_action = sentry_v6.portable_forward([0.] * sentry_v6.schema()['obs_size'], payload)
                    else:
                        raise ValueError("Unrecognized portable actor format")
                    row["portable_actor_finite"] = bool(np.isfinite(actor_action).all())
                    row["finite"] &= row["portable_actor_finite"]
                if config.get('policy_version') == 6:
                    from .v6_bundle import check_main_model
                    row.update(check_main_model(model))
                    row['finite'] &= row['hybrid_contract_valid']
                report["inference"][stage] = row
            except Exception as error:
                report["inference"][stage] = dict(finite=False, error=str(error))
    report["inference_ready"] = bool(inference and len(report["inference"]) == len(STAGES) and all(row.get("finite") for row in report["inference"].values()))
    report["training_ready"] = bool(report["inference_ready"] and report["warp_cuda_available"]
                                    and native_report["available"] and all(row["available"] for row in report["dependency_imports"].values()))
    report["ready"] = report["training_ready"]
    report["resume_state_complete"] = not report["missing_resume_state"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("pack")
    pack.add_argument("--run-dir", type=Path, required=True); pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--preparation", type=Path); pack.add_argument("--source-root", type=Path); pack.add_argument("--database", type=Path)
    unpack = commands.add_parser("unpack"); unpack.add_argument("archive", type=Path); unpack.add_argument("--destination", type=Path, required=True)
    check = commands.add_parser("preflight"); check.add_argument("directory", type=Path); check.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    if args.command == "pack":
        result = create_bundle(args.run_dir, args.output, preparation=args.preparation, source_root=args.source_root, external_database=args.database)
        result.pop("manifest")
    elif args.command == "unpack":
        value = extract_bundle(args.archive, args.destination)
        result = dict(destination=str(args.destination), integrity=True, stages=list(value["stages"]))
    else:
        result = preflight(args.directory, inference=not args.skip_inference)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    if args.command == "preflight" and not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
