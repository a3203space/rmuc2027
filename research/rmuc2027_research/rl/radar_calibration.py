"""Offline class calibration from legal tracks with separately attached labels.

The recorder is research-only. It never sends labels to the runtime estimator,
actor, or radar link. Track support is counted by episode/team/slot and the
validation split groups entire episodes, including both radar viewpoints.
"""
from collections import defaultdict
import copy
from fractions import Fraction
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

from rmuc2027_core.radar_inference import ChassisClassifier, ENEMY_SLOTS


def stratified_groups(group_counts, *, minimum_tracks=4):
    """Select complete match-seed groups using *only* class counts.

    The objective and search budget are fixed before fitting. Small datasets
    exhaust every partition; larger datasets use bounded deterministic prefix
    candidates. Failure to find a feasible split never enables a classifier.
    No trajectory features, predictions or validation metrics enter this API.
    """
    labels = ChassisClassifier.labels
    names = sorted(group_counts)
    vectors = {name: tuple(int(group_counts[name].get(c, 0)) for c in labels) for name in names}
    totals = tuple(sum(v[i] for v in vectors.values()) for i in range(len(labels)))
    method = 'exhaustive' if len(names) <= 16 else 'bounded_deterministic_prefixes'
    manifest = dict(version=2, method=method, exhaustive=method == 'exhaustive',
        target_validation_fraction='1/3', minimum_tracks_per_class_per_split=minimum_tracks,
        group_counts={name:dict(zip(labels, vectors[name])) for name in names},
        objective=['max_class_fraction_deviation', 'sum_class_fraction_deviation',
                   'total_track_fraction_deviation', 'group_fraction_deviation', 'sha256_tie_break'],
        candidates_evaluated=0, feasible_candidates=0, selection_inputs='match_seed_and_label_counts_only')
    if minimum_tracks < 4: raise ValueError('At least four independent tracks per class required')
    if len(names) < 2 or any(n < 2*minimum_tracks for n in totals):
        return None, manifest | dict(reason='insufficient_grouped_class_support')
    if any(sum(v[i] > 0 for v in vectors.values()) < 2 for i in range(len(labels))):
        return None, manifest | dict(reason='insufficient_grouped_class_support')
    best = None; heldout = None

    def consider(selected, counts):
        nonlocal best, heldout
        manifest['candidates_evaluated'] += 1
        if any(n < minimum_tracks or total-n < minimum_tracks for n,total in zip(counts,totals)): return
        manifest['feasible_candidates'] += 1
        deviations = [abs(Fraction(n,total)-Fraction(1,3)) for n,total in zip(counts,totals)]
        selected = tuple(sorted(selected))
        tie = hashlib.sha256(('radar-stratified-v2/'+','.join(selected)).encode()).hexdigest()
        score = (max(deviations),sum(deviations),abs(Fraction(sum(counts),sum(totals))-Fraction(1,3)),
                 abs(Fraction(len(selected),len(names))-Fraction(1,3)),tie)
        if best is None or score < best: best, heldout = score, selected

    if method == 'exhaustive':
        for size in range(1, len(names)):
            for selected in combinations(names, size):
                consider(selected, tuple(sum(vectors[name][i] for name in selected) for i in range(len(labels))))
    else:
        # 39 orders x <=105 prefixes = <=4095 candidates, independent of the
        # number of groups. Prefix counts are accumulated in O(39*n) time.
        orders = [names]
        for index in range(len(labels)):
            ordered = sorted(names, key=lambda name:(vectors[name][index], name))
            orders.extend((ordered, list(reversed(ordered))))
        for salt in range(32):
            orders.append(sorted(names, key=lambda name:hashlib.sha256(
                f'radar-stratified-v2/{salt}/{name}'.encode()).digest()))
        limit = 4096//len(orders)
        sizes = set(range(1,len(names))) if len(names)-1 <= limit else {
            1+round(i*(len(names)-2)/(limit-1)) for i in range(limit)}
        for ordered in orders:
            counts = [0]*len(labels)
            for size,name in enumerate(ordered,1):
                counts = [a+b for a,b in zip(counts,vectors[name])]
                if size in sizes: consider(ordered[:size], counts)
    if heldout is None:
        return None, manifest | dict(reason='no_feasible_group_split' if manifest['exhaustive'] else 'no_feasible_group_split_found')
    manifest.update(reason='label_support_satisfied', validation_groups=list(heldout),
        train_groups=[name for name in names if name not in heldout],
        score=[str(value) for value in best[:-1]], tie_break_sha256=best[-1])
    return set(heldout), manifest


class RadarCalibrationRecorder:
    def __init__(self):
        self._tracks = {}; self._last = {}

    def observe(self, match, episode_key):
        gateway = match.sentry_gateway
        if gateway is None: return 0
        episode = str(episode_key); now = float(gateway.snapshot['time'])
        if now-self._last.get(episode, -math.inf) < 1.-1e-8: return 0
        self._last[episode] = now
        # Private labels are read exclusively here, after feature construction.
        robots = {(r['team'], r['slot']): r for r in gateway.snapshot['robots']}
        count = 0
        for team, estimator in gateway.inference.items():
            for slot in ENEMY_SLOTS:
                features = estimator.calibration_features(slot, now)
                if features is None: continue
                robot = robots.get((estimator.other, slot))
                if robot is None: continue
                label = ('fold' if robot.get('transform_allowed') else
                         'wheel' if robot.get('chassis_type') == 'wheel_legged' else 'omni')
                key = (episode, estimator.other, slot)
                row = self._tracks.setdefault(key, dict(episode_key=episode,
                    track_id=f'{episode}/{estimator.other}/{slot}', label=label,
                    split_group=f'seed-{gateway.snapshot["seed"]}' if 'seed' in gateway.snapshot else episode,
                    windows=[], first_time=now, last_time=now))
                if row['label'] != label: raise ValueError('Chassis label changed within one trajectory')
                row['windows'].append(features); row['last_time'] = now; count += 1
        return count

    def rows(self):
        # One representative measured window per trajectory avoids pretending
        # overlapping windows are independent samples, and matches runtime scale.
        result = []
        for row in self._tracks.values():
            centers = [statistics.median(v[i] for v in row['windows']) for i in range(4)]
            chosen = min(row['windows'], key=lambda v: sum((a-b)**2 for a,b in zip(v, centers)))
            result.append({k: copy.deepcopy(v) for k,v in row.items() if k != 'windows'} |
                dict(features=list(chosen), window_count=len(row['windows']),
                     feature_source='legal_radar_coordinate_history', label_source='private_offline_chassis'))
        return sorted(result, key=lambda r:r['track_id'])

    def save(self, path):
        path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.rows(), indent=2), encoding='utf-8')
        return path


def fit_validate(rows, *, minimum_tracks=4):
    """Return (model, report); failed independent validation keeps unknown.

Requirements: >=4 independent tracks/class in train and heldout, >=25%
high-confidence supported coverage for every heldout class, >=70% selective
accuracy, >=50% macro accuracy and Brier <= the uniform-class baseline.
Thresholds are fixed before fitting; validation is never used to tune them.
"""
    labels = ChassisClassifier.labels; unique={}
    for original in rows:
        row=copy.deepcopy(original)
        if row['label'] not in labels: raise ValueError('Unknown calibration class')
        if len(row['features']) != 4 or not all(math.isfinite(float(x)) for x in row['features']):
            raise ValueError('Calibration needs four finite legal motion features')
        key=(str(row['episode_key']), str(row['track_id']))
        if key in unique: raise ValueError('Duplicate calibration trajectory')
        unique[key]=row
    group=lambda r:str(r.get('split_group',r['episode_key']))
    episodes=sorted({group(r) for r in unique.values()})
    group_counts={episode:{label:0 for label in labels} for episode in episodes}
    for row in unique.values(): group_counts[group(row)][row['label']] += 1
    heldout,split = stratified_groups(group_counts, minimum_tracks=minimum_tracks)
    train=[] if heldout is None else [r for r in unique.values() if group(r) not in heldout]
    validation=[] if heldout is None else [r for r in unique.values() if group(r) in heldout]
    counts=lambda group:{label:sum(r['label']==label for r in group) for label in labels}
    report=dict(version=2, valid=False, independent_tracks=len(unique), episodes=len(episodes), split=split,
        train_counts=counts(train), validation_counts=counts(validation),
        train_episodes=[] if heldout is None else [e for e in episodes if e not in heldout],
        validation_episodes=[] if heldout is None else sorted(heldout),
        thresholds=dict(minimum_tracks_per_class_per_split=minimum_tracks, selective_accuracy=.7,
            per_class_coverage=.25, macro_accuracy=.5, brier=2/3),
        label_source='private_offline_chassis', runtime_input='legal_radar_coordinate_history')
    if heldout is None:
        report['reason']=split['reason']; return dict(validated=False,calibration=report),report
    if any(n < minimum_tracks for n in report['train_counts'].values()):
        report['reason']='insufficient_training_tracks'; return dict(validated=False, calibration=report),report
    classifier=ChassisClassifier.fit(train, validated=False)
    classifier.model['temperature']=2.
    if any(n < minimum_tracks for n in report['validation_counts'].values()):
        report['reason']='insufficient_heldout_tracks'
        return classifier.model | dict(calibration=report),report
    # Evaluate raw scores without ever setting validated=True before acceptance.
    candidate=ChassisClassifier(classifier.model)
    by_class=defaultdict(list); accepted=[]; brier=[]
    for row in validation:
        probabilities,supported=candidate._likelihood(row['features'])
        ordering=sorted(range(3), key=lambda i:probabilities[i], reverse=True)
        confidence=bool(supported and probabilities[ordering[0]]>=.7 and
            probabilities[ordering[0]]-probabilities[ordering[1]]>=.2)
        correct=labels[ordering[0]]==row['label']
        by_class[row['label']].append((correct,confidence))
        if confidence: accepted.append(correct)
        brier.append(sum((p-int(label==row['label']))**2 for p,label in zip(probabilities,labels)))
    metrics={label:dict(accuracy=statistics.mean(x[0] for x in values),
        coverage=statistics.mean(x[1] for x in values)) for label,values in by_class.items()}
    selective=statistics.mean(accepted) if accepted else 0.
    macro=statistics.mean(m['accuracy'] for m in metrics.values()); score=statistics.mean(brier)
    valid=selective>=.7 and macro>=.5 and score<=2/3 and all(m['coverage']>=.25 for m in metrics.values())
    report.update(valid=valid, reason='passed' if valid else 'failed_independent_calibration',
        classes=metrics, selective_accuracy=selective, macro_accuracy=macro, brier=score,
        accepted_validation_tracks=len(accepted))
    model=classifier.model | dict(validated=valid, calibration=copy.deepcopy(report))
    return model,report


def save_classifier(rows, directory):
    directory=Path(directory); directory.mkdir(parents=True, exist_ok=True)
    model,report=fit_validate(rows)
    for name,value in (('classifier.json',model),('classifier-validation.json',report)):
        (directory/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    return directory/'classifier.json',report
