"""Materialize a published v6 generation, without simulation or optimization.

Unlike bundle default resume, the caller explicitly chooses the complete
generation. Rejected descendant weights, Adam moments and sampler counts are
never silently merged into the selected ancestor.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil


MEMBERS = ('model.zip', 'resume-state.pt', 'samplers.json', 'actor.json',
           'experiment.json', 'coordinator-state.json')
OPTIMIZERS = ('tactical_optimizer', 'revive_optimizer', 'critic_optimizer')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _skill_hash(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        if '.skill.' in name or name == 'skill_log_std':
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def materialize_generation(source_run, generation, output, *, rejected_candidate=None,
                           reason='Explicit continuation from a published generation'):
    """Return (relocated main config, provenance report); do not run an env.

    ``generation`` is a path relative to source_run, or an absolute child path.
    Output may already contain authorization/budget records, but model and
    runtime-resource directories must not exist. Optimizer/RNG bytes are copied
    exactly; only config paths and interrupted coordinator leases change.
    """
    import torch
    from .campaign import write_json
    from .learning import experiment
    from .v4_bundle import _resolve_submission, _walk
    from .v6_bundle import check_main_model
    from .v6_ppo import HybridPPO
    from .v6_sampling import TrainingCoordinator

    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    requested = Path(generation)
    source = (requested if requested.is_absolute() else source_run / requested).resolve()
    if source_run not in source.parents or source_run == output or source_run in output.parents:
        raise ValueError('Generation must belong to source run; output must be a separate run')
    complete = json.loads((source / 'complete.json').read_text())
    if (source_run / complete.get('directory', '')).resolve() != source:
        raise ValueError('Generation directory disagrees with publication record')
    if set(complete.get('sha256', {})) != set(MEMBERS):
        raise ValueError('Published generation must contain the six exact state members')
    for name in MEMBERS:
        if _sha(source / name) != complete['sha256'][name]:
            raise ValueError('Published generation hash mismatch: ' + name)
    metadata = json.loads((source / 'experiment.json').read_text())
    if metadata.get('config', {}).get('policy_version') != 6:
        raise ValueError('Continuation requires a v6 hybrid generation')
    model = HybridPPO.load(source / 'model.zip', device='cpu')
    state = torch.load(source / 'resume-state.pt', map_location='cpu', weights_only=False)
    if state.get('optimizers') != list(OPTIMIZERS) or state.get('policy_version') != 6:
        raise ValueError('Missing v6 three-optimizer resume contract')
    for name in ('num_timesteps', 'gradient_epochs', 'optimizer_steps',
                 'tactical_optimizer_steps', 'revive_optimizer_steps', 'critic_optimizer_steps'):
        attribute = '_n_updates' if name == 'gradient_epochs' else name
        if state.get(name) != getattr(model, attribute, 0):
            raise ValueError('Model and RNG-state counters disagree: ' + name)
    if (complete.get('samples') != model.num_timesteps or
            complete.get('update_id') != getattr(model, 'completed_updates', 0) or
            complete.get('optimizer_steps') != getattr(model, 'optimizer_steps', 0) or
            complete.get('skill_sha256') != _skill_hash(model.policy)):
        raise ValueError('Publication counters or frozen skill hash disagree')
    validation = check_main_model(model, source_run / 'main/initialized.zip')
    if not validation['hybrid_contract_valid']:
        raise ValueError('Selected generation violates hybrid/frozen-skill contract')
    resources = ('main', 'infantry', 'heavy', 'sampler-coordinator.json',
                 'capability-profiles', 'radar-calibration', 'preparation', 'bundle-resources')
    for name in resources:
        if (output / name).exists():
            raise FileExistsError('Continuation cannot overwrite: ' + str(output / name))
    for role in ('infantry', 'heavy'):
        for name in ('model.zip', 'resume-state.pt', 'samplers.json', 'actor.json', 'experiment.json'):
            if not (source_run / role / name).is_file():
                raise ValueError('Missing frozen auxiliary member: ' + role + '/' + name)
    for side in ('red', 'blue'):
        if not (source_run / f'capability-profiles/evaluation-{side}.json').is_file():
            raise ValueError('Missing frozen evaluation capability profile')

    output.mkdir(parents=True, exist_ok=True)
    (output / 'main').mkdir()
    for name in MEMBERS:
        shutil.copy2(source / name, output / 'main' / name)
    for role in ('infantry', 'heavy'):
        target = output / role
        target.mkdir()
        for name in ('model.zip', 'initialized.zip', 'resume-state.pt', 'samplers.json',
                     'actor.json', 'initialized-actor.json', 'experiment.json', 'training.json'):
            if (source_run / role / name).is_file():
                shutil.copy2(source_run / role / name, target / name)
    for name in ('capability-profiles', 'radar-calibration', 'preparation'):
        if (source_run / name).is_dir():
            shutil.copytree(source_run / name, output / name)
    shutil.copy2(source / 'coordinator-state.json', output / 'sampler-coordinator.json')
    old_coordinator = json.loads((source / 'coordinator-state.json').read_text())
    coordinator = TrainingCoordinator(output / 'sampler-coordinator.json')
    coordinator.release_interrupted_leases()
    restored_coordinator = coordinator.snapshot()
    if old_coordinator.get('counts') != restored_coordinator.get('counts'):
        raise ValueError('Lease release changed historical coverage counts')
    write_json(output / 'main/coordinator-state.json', restored_coordinator)

    source_root = Path(__file__).resolve().parents[3]
    bindings = {}

    def rebind(reference, preferred=None):
        path = _resolve_submission(reference, source_root).resolve()
        if not path.exists():
            raise ValueError('Missing configured continuation resource: ' + str(path))
        if preferred is not None and path.is_file() and preferred.is_file() and _sha(path) == _sha(preferred):
            target = output / preferred.relative_to(source_run)
        elif source_run in path.parents:
            target = output / path.relative_to(source_run)
        else:
            entries = list(_walk(path)) if path.is_dir() else [(path, path.name)]
            digest = hashlib.sha256(json.dumps([(rel, _sha(p)) for p, rel in entries]).encode()).hexdigest()[:20]
            target = output / 'bundle-resources' / digest / path.name
        if not target.exists():
            if path.is_dir():
                target.mkdir(parents=True)
                for item, relative in _walk(path):
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, destination)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        bindings[str(path)] = str(target)
        return str(target)

    configs = {}
    for stage in ('main', 'infantry', 'heavy'):
        config = copy.deepcopy(json.loads((output / stage / 'experiment.json').read_text())['config'])
        for key in ('friendly', 'opponent'):
            if config.get(key):
                config[key] = rebind(config[key])
        for controller in config.get('controllers', {}).values():
            if controller.get('submission'):
                controller['submission'] = rebind(controller['submission'])
        for role, reference in config.get('auxiliary_pool', {}).items():
            config['auxiliary_pool'][role] = rebind(reference, source_run / role / 'actor.json')
        if config.get('radar_classifier_path'):
            config['radar_classifier_path'] = rebind(config['radar_classifier_path'], source_run / 'radar-calibration/classifier.json')
        for key in ('_training_deadline', '_episode_log_dir', '_v6_replay_dir', '_worker_id', 'github_backup_repo'):
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
    rejected = None
    if rejected_candidate is not None:
        rejected_path = Path(rejected_candidate)
        if not rejected_path.is_absolute():
            rejected_path = source_run / rejected_path
        rejected = dict(path=str(rejected_path.resolve()), sha256=_sha(rejected_path),
                        retained_in_ancestor=True, state_imported=False)
    report = dict(version=6, mode='continue_published_generation', source_run=str(source_run),
        source_generation=str(source), source_complete_sha256=_sha(source / 'complete.json'),
        source_members_sha256=complete['sha256'], source_model_sha256=_sha(source / 'model.zip'),
        resumed_model_sha256=_sha(main / 'model.zip'), reason=str(reason), rejected_candidate=rejected,
        source_num_timesteps=model.num_timesteps, source_gradient_epochs=model._n_updates,
        source_completed_updates=getattr(model, 'completed_updates', 0),
        optimizer_steps={name: getattr(model, name, 0) for name in ('tactical_optimizer_steps',
            'revive_optimizer_steps', 'critic_optimizer_steps')}, optimizer_reset=False, optimizer_count=3,
        source_coordinator_sha256=_sha(source / 'coordinator-state.json'),
        released_interrupted_leases=len(old_coordinator.get('leases', {})),
        historical_counts_preserved=True, descendant_coverage_imported=False,
        rng_file_copied_exact=_sha(main / 'resume-state.pt') == complete['sha256']['resume-state.pt'],
        source_sampler_copied_exact=_sha(main / 'samplers.json') == complete['sha256']['samplers.json'],
        runtime_bindings=bindings, rollout_reused=False, fresh_episodes_required=True,
        gradient_updates=0, physics_steps=0, frozen_contract_validation=validation)
    write_json(main / 'migration.json', report)
    write_json(output / 'initialization.json', report)
    destination = main / 'checkpoints/baseline'
    destination.mkdir(parents=True)
    for name in MEMBERS:
        shutil.copy2(main / name, destination / name)
    ready = dict(directory=destination.relative_to(output).as_posix(), samples=model.num_timesteps,
        update_id=getattr(model, 'completed_updates', 0), optimizer_steps=getattr(model, 'optimizer_steps', 0),
        skill_sha256=_skill_hash(model.policy), sha256={name: _sha(destination / name) for name in MEMBERS})
    write_json(destination / 'complete.json', ready)
    write_json(main / 'checkpoint-ready.json', ready)
    write_json(main / 'training.json', dict(actual_samples=0, total_samples=model.num_timesteps,
        accepted_actor_steps=0, skill_unchanged=True, status='not_started', demonstration_updates_during_ppo=0,
        resume_source=report['source_model_sha256']))
    return configs['main'], report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rejected-candidate')
    parser.add_argument('--reason', required=True)
    args = parser.parse_args()
    _, report = materialize_generation(args.source_run, args.generation, args.output,
        rejected_candidate=args.rejected_candidate, reason=args.reason)
    print(json.dumps(report, ensure_ascii=False))
