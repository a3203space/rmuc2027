"""Relocatable v6 hybrid PPO research release. Integrity/inference, never training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from . import v4_bundle


REQUIRED_RUN_FILES = ('run-results.json', 'run-manifest.json', 'sampler-coordinator.json', 'main/migration.json')


def create_bundle(run_dir, output, **kwargs):
    root = Path(run_dir)
    missing = [name for name in REQUIRED_RUN_FILES if not (root / name).is_file()]
    missing += [f'{stage}/{name}' for stage in v4_bundle.STAGES
                for name in ('resume-state.pt', 'samplers.json') if not (root / stage / name).is_file()]
    if missing:
        raise ValueError('Incomplete v6 frozen release: ' + ', '.join(missing))
    metadata = json.loads((root / 'main/experiment.json').read_text())
    if metadata.get('config', {}).get('policy_version') != 6:
        raise ValueError('v6 bundle requires a v6 main and auxiliary v5 policies')
    for stage in ('infantry', 'heavy'):
        if json.loads((root / stage / 'experiment.json').read_text()).get('config', {}).get('policy_version') != 5:
            raise ValueError('v6 release auxiliaries must retain their v5 contract')
    return v4_bundle.create_bundle(run_dir, output, **kwargs)


def verify_bundle(directory):
    manifest = v4_bundle.verify_bundle(directory)
    if manifest['format'] != 'rmuc2027-sentry-v6-resume':
        raise ValueError('Expected a v6 hybrid research bundle')
    expected = ['run-state/' + name for name in REQUIRED_RUN_FILES if '/' not in name]
    expected += ['models/main/migration.json']
    missing = [name for name in expected if name not in manifest['files']]
    if missing:
        raise ValueError('Missing frozen v6 release records: ' + ', '.join(missing))
    return manifest


def extract_bundle(archive, destination):
    v4_bundle.extract_bundle(archive, destination)
    return verify_bundle(destination)


load_runtime_config = v4_bundle.load_runtime_config
relocate_model_runtime = v4_bundle.relocate_model_runtime


def prepare_resume_run(bundle, output):
    """Materialize a new round without simulation, gradients or rewriting source.

    Returns (runtime_config, report). The campaign may already have created
    budget/case records in output. Existing model directories are never replaced.
    Continue from the current model, not the previous selection decision.
    """
    from .campaign import write_json
    from .learning import experiment
    from .v6_ppo import HybridPPO
    from .v6_sampling import TrainingCoordinator
    import torch
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    manifest = verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=True)
    for name in (*v4_bundle.STAGES, 'sampler-coordinator.json', 'capability-profiles',
                 'radar-calibration', 'preparation', 'bundle-resources'):
        if (output / name).exists():
            raise FileExistsError('Resume cannot overwrite an existing run resource: ' + str(output / name))
    model = HybridPPO.load(bundle / manifest['stages']['main']['checkpoint'], device='cpu')
    validation = check_main_model(model, bundle / 'models/main/initialized.zip')
    if not validation['hybrid_contract_valid']:
        raise ValueError('Source hybrid checkpoint contract/frozen-skill validation failed')
    state = torch.load(bundle / 'models/main/resume-state.pt', map_location='cpu', weights_only=False)
    if state.get('num_timesteps') != model.num_timesteps or state.get('optimizers') != [
            'tactical_optimizer', 'revive_optimizer', 'critic_optimizer']:
        raise ValueError('Source model and three-optimizer/RNG checkpoint set disagree')
    # Copy only current model state. Ancestor updates remain in the immutable
    # source bundle, avoiding exponential checkpoint/archive duplication.
    for stage in v4_bundle.STAGES:
        source, target = bundle / 'models' / stage, output / stage
        target.mkdir()
        for name in ('model.zip', 'resume-state.pt', 'samplers.json', 'actor.json'):
            source_path = source / name
            if name == 'model.zip':
                source_path = bundle / manifest['stages'][stage]['checkpoint']
            if not source_path.is_file():
                raise ValueError('Missing resumable stage member: ' + str(source_path))
            shutil.copy2(source_path, target / name)
        for name in ('initialized.zip', 'initialized-actor.json', 'training.json',
                     'training-radar-calibration-rows.json'):
            if (source / name).is_file():
                shutil.copy2(source / name, target / name)
    for name in ('preparation', 'radar-calibration', 'capability-profiles'):
        if (bundle / name).is_dir():
            shutil.copytree(bundle / name, output / name)
    for side in ('red', 'blue'):
        if not (output / f'capability-profiles/evaluation-{side}.json').is_file():
            raise ValueError('Source bundle lacks frozen side-specific evaluation capabilities')
    coordinator_source = bundle / 'models/main/coordinator-state.json'
    coordinator_kind = 'checkpoint_coordinator'
    if not coordinator_source.is_file():
        coordinator_source = bundle / 'run-state/sampler-coordinator.json'
        coordinator_kind = 'run_coordinator_fallback'
    if not coordinator_source.is_file():
        raise ValueError('Missing v6 coordinator state')
    source_coordinator = json.loads(coordinator_source.read_text())
    shutil.copy2(coordinator_source, output / 'sampler-coordinator.json')
    coordinator = TrainingCoordinator(output / 'sampler-coordinator.json')
    coordinator.release_interrupted_leases()
    restored_coordinator = coordinator.snapshot()
    if source_coordinator.get('counts') != restored_coordinator.get('counts'):
        raise ValueError('Releasing interrupted leases changed historical coverage counts')

    def rebind(reference):
        relative = v4_bundle._relative(reference)
        parts = relative.parts
        if parts[0] == 'models':
            target = output.joinpath(*parts[1:])
        elif parts[0] in ('radar-calibration', 'capability-profiles', 'preparation'):
            target = output.joinpath(*parts)
        elif relative.as_posix() == 'run-state/sampler-coordinator.json':
            return str(output / 'sampler-coordinator.json')
        else:
            target = output / 'bundle-resources' / relative
        source = bundle / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
        return str(target)

    configs = {}
    for stage in v4_bundle.STAGES:
        info = manifest['stages'][stage]
        config = json.loads((bundle / info['config']).read_text())
        for key, reference in info['resource_bindings'].items():
            target = rebind(reference)
            if key.startswith('controller:'):
                config['controllers'][key.split(':', 1)[1]]['submission'] = target
            elif key.startswith('auxiliary_pool:'):
                config['auxiliary_pool'][key.split(':', 1)[1]] = target
            else:
                config[key] = target
        for key in ('_training_deadline', '_episode_log_dir', '_v6_replay_dir', '_worker_id'):
            config.pop(key, None)
        if stage == 'main':
            config.update(_v6_sampler_path=str(output / 'sampler-coordinator.json'),
                _v6_replay_dir=str(output / 'replays/training'),
                _episode_log_dir=str(output / 'main/completed-episodes'))
        configs[stage] = config
        write_json(output / stage / 'experiment.json', experiment(config))
    main = output / 'main'
    shutil.copy2(main / 'model.zip', main / 'initialized.zip')
    shutil.copy2(main / 'actor.json', main / 'initialized-actor.json')
    write_json(main / 'coordinator-state.json', restored_coordinator)
    hash_file = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = dict(version=6, mode='resume_hybrid_checkpoint', source_bundle=str(bundle),
        source_manifest_sha256=hash_file(bundle / 'bundle-manifest.json'),
        source_model_sha256=hash_file(bundle / manifest['stages']['main']['checkpoint']),
        resumed_model_sha256=hash_file(main / 'model.zip'), optimizer_reset=False, optimizer_count=3,
        source_num_timesteps=model.num_timesteps, source_gradient_epochs=model._n_updates,
        source_completed_updates=getattr(model, 'completed_updates', 0),
        source_coordinator=coordinator_source.relative_to(bundle).as_posix(),
        coordinator_restore=coordinator_kind,
        source_coordinator_sha256=hash_file(coordinator_source),
        released_interrupted_leases=len(source_coordinator.get('leases', {})),
        historical_counts_preserved=True, ancestor_updates_retained_in_source_bundle=True,
        gradient_updates=0, physics_steps=0, rollout_reused=False, fresh_episodes_required=True,
        rng_file_copied_exact=True, new_round_baseline='restored_current_model',
        target_hardware_verified=False, frozen_contract_validation=validation)
    write_json(main / 'migration.json', report)
    write_json(output / 'initialization.json', report)
    generation = main / 'checkpoints/baseline'
    generation.mkdir(parents=True)
    names = ('model.zip', 'resume-state.pt', 'samplers.json', 'actor.json', 'experiment.json', 'coordinator-state.json')
    for name in names:
        shutil.copy2(main / name, generation / name)
    ready = dict(directory=generation.relative_to(output).as_posix(), samples=model.num_timesteps,
        update_id=getattr(model, 'completed_updates', 0), sha256={name: hash_file(generation / name) for name in names})
    write_json(generation / 'complete.json', ready)
    write_json(main / 'checkpoint-ready.json', ready)
    write_json(main / 'training.json', dict(actual_samples=0, total_samples=model.num_timesteps,
        accepted_actor_steps=0, skill_unchanged=True, status='not_started', demonstration_updates_during_ppo=0,
        resume_source=report['source_model_sha256']))
    return configs['main'], report


def check_main_model(model, baseline_path=None):
    """Exercise all legal revival regimes, with zero gradients and no simulator."""
    import numpy as np
    import torch
    from .sentry_v6 import FEATURE_INDEX, portable_forward, v6_feature_masks
    from .sentry_v6_policy import export_actor
    from .v6_ppo import HybridPPO
    if not isinstance(model, HybridPPO):
        raise ValueError('v6 checkpoint must load with HybridPPO')
    shapes = {name: tuple(space.shape) for name, space in model.observation_space.spaces.items()}
    if shapes != dict(actor=(775,), critic=(454,), revive_mask=(3,)) or model.action_space.shape != (7,):
        raise ValueError('Unexpected v6 actor/critic/mask/action schema')
    rng = np.random.default_rng(6021)
    actor = rng.uniform(-.25, .25, (6, 775)).astype(np.float32)
    for index in range(len(actor)):
        actor[index, FEATURE_INDEX['packet.0201.valid']] = 0 if index == 5 else 1
        actor[index, FEATURE_INDEX['packet.0201.age']] = 0
        actor[index, FEATURE_INDEX['self.hp']] = .3 if index in (0, 1) else 0
        actor[index, FEATURE_INDEX['packet.020D.valid']] = 1 if index in (1, 3, 4) else 0
        actor[index, FEATURE_INDEX['packet.020D.age']] = 0
        actor[index, FEATURE_INDEX['ack.paid_revive']] = 1 if index == 3 else 0
        actor[index, FEATURE_INDEX['ack.free_revive']] = 1 if index == 4 else 0
    masks = np.asarray([v6_feature_masks(row) for row in actor], np.float32)
    obs = dict(actor=actor, critic=np.zeros((6, 454), np.float32), revive_mask=masks)
    actions = model.predict(obs, deterministic=True)[0]
    portable = np.asarray([portable_forward(row, export_actor(model.policy)) for row in actor])
    portable[:, :6] = portable[:, :6].clip(-1, 1)
    parity = float(np.max(np.abs(actions - portable)))
    skills = list(model.policy.mlp_extractor.policy_net.skill.parameters())
    names = ('tactical_optimizer', 'revive_optimizer', 'critic_optimizer')
    optimizer_parameters = [{id(p) for group in getattr(model, name).param_groups for p in group['params']} for name in names]
    frozen = all(not parameter.requires_grad for parameter in skills)
    frozen &= not bool(set(map(id, skills)) & set.union(*optimizer_parameters))
    frozen &= 'skill_log_std' in dict(model.policy.named_buffers())
    disjoint = not any(optimizer_parameters[i] & optimizer_parameters[j] for i in range(3) for j in range(i))
    dead = masks[:, 1] > .5
    dead_noop = bool(np.all(actions[dead, :6] == 0))
    mask_valid = bool(all(float(action[6]).is_integer() and masks[index, int(action[6])] == 1
                          for index, action in enumerate(actions)))
    tensor = {key: torch.as_tensor(value, device=model.device) for key, value in obs.items()}
    with torch.no_grad():
        distribution = model.policy.get_distribution(tensor)
        logp, _ = distribution.log_prob_parts(torch.as_tensor(actions, device=model.device))
    # The continuous executed clip is irrelevant for dead rows, where the
    # recorded Gaussian term must be exactly absent.
    dead_zero_logp = bool((logp[torch.as_tensor(dead, device=model.device)] == 0).all())
    same_skills = None
    if baseline_path is not None and Path(baseline_path).is_file():
        baseline = HybridPPO.load(baseline_path, device='cpu')
        same_skills = all(torch.equal(value.detach().cpu(), baseline.policy.state_dict()[name].detach().cpu())
            for name, value in model.policy.state_dict().items() if '.skill.' in name or name == 'skill_log_std')
    valid = bool(parity <= 1e-5 and frozen and disjoint and dead_noop and mask_valid and dead_zero_logp
                 and same_skills is not False and np.isfinite(actions).all())
    return dict(hybrid_contract_valid=valid, actor_size=775, critic_size=454, revive_mask_size=3,
        portable_parity_max_error=parity, frozen_skill=frozen, frozen_skill_matches_initial=same_skills,
        optimizer_groups_disjoint=disjoint, optimizers={name: dict(parameters=len(parameters),
            states=len(getattr(model, name).state)) for name, parameters in zip(names, optimizer_parameters)},
        dead_continuous_noop=dead_noop, dead_continuous_logprob_zero=dead_zero_logp,
        categorical_legal=mask_valid, synthetic_regimes=6, gradient_updates=0)


def preflight(directory, *, inference=True):
    manifest = verify_bundle(directory)
    report = v4_bundle.preflight(directory, inference=inference)
    report.update(format=manifest['format'], target_hardware_claimed=False,
                  required_physical_parallelism_probe=[1, 2, 4, 8], physical_throughput_measured=False)
    if inference and report.get('inference', {}).get('main', {}).get('finite'):
        import torch
        from .v6_ppo import HybridPPO
        root = Path(directory)
        model = HybridPPO.load(root / manifest['stages']['main']['checkpoint'], device='cpu')
        row = check_main_model(model, root / 'models/main/initialized.zip')
        state = torch.load(root / 'models/main/resume-state.pt', map_location='cpu', weights_only=False)
        row['resume_state_matches_checkpoint'] = bool(state.get('num_timesteps') == model.num_timesteps
            and state.get('optimizers') == ['tactical_optimizer', 'revive_optimizer', 'critic_optimizer']
            and state.get('frozen_skill') is True)
        row['hybrid_contract_valid'] &= row['resume_state_matches_checkpoint']
        report['inference']['main'].update(row)
        report['inference']['main']['finite'] &= row['hybrid_contract_valid']
        report['inference_ready'] &= row['hybrid_contract_valid']
        report['training_ready'] &= row['hybrid_contract_valid']
        report['ready'] = report['training_ready']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pack = commands.add_parser('pack')
    pack.add_argument('--run-dir', type=Path, required=True)
    pack.add_argument('--output', type=Path, required=True)
    pack.add_argument('--source-root', type=Path)
    pack.add_argument('--preparation', type=Path)
    pack.add_argument('--database', type=Path)
    unpack = commands.add_parser('unpack')
    unpack.add_argument('archive', type=Path)
    unpack.add_argument('--destination', type=Path, required=True)
    check = commands.add_parser('preflight')
    check.add_argument('directory', type=Path)
    check.add_argument('--skip-inference', action='store_true')
    args = parser.parse_args()
    if args.command == 'pack':
        report = create_bundle(args.run_dir, args.output, source_root=args.source_root,
                               preparation=args.preparation, external_database=args.database)
        report.pop('manifest')
    elif args.command == 'unpack':
        manifest = extract_bundle(args.archive, args.destination)
        report = dict(integrity=True, destination=str(args.destination), format=manifest['format'])
    else:
        report = preflight(args.directory, inference=not args.skip_inference)
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if args.command == 'preflight' and not report['ready']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
