import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rmuc2027_core.radar_inference import ChassisClassifier, RadarInference
from rmuc2027_research.rl.radar_calibration import RadarCalibrationRecorder, fit_validate, stratified_groups


def rows(overlap=False):
    return [dict(episode_key=f'seed-{episode}', track_id=f'seed-{episode}/{label}', label=label,
        features=[(.5 if overlap else index*2+.5)+episode*.001, 1. if overlap else index*2+1.,
                  .1 if overlap else index*.4, .1])
        for episode in range(24) for index,label in enumerate(ChassisClassifier.labels)]


class RadarCalibrationTests(unittest.TestCase):
    def test_split_uses_only_labels_and_fit_never_receives_heldout_tracks(self):
        data=rows()
        # Different role/viewpoint names share the same physical match seed.
        for index,row in enumerate(data):
            row['split_group']=row['episode_key']
            row['episode_key']=f"role-{index%2}/{row['episode_key']}"
        with patch.object(ChassisClassifier,'fit',wraps=ChassisClassifier.fit) as fit:
            _,report=fit_validate(data)
            received=fit.call_args.args[0]
        self.assertEqual({r['split_group'] for r in received},set(report['train_episodes']))
        self.assertFalse({r['split_group'] for r in received} & set(report['validation_episodes']))
        changed=copy.deepcopy(list(reversed(data)))
        for row in changed:row['features']=[999.,-999.,0.,.5]
        _,other=fit_validate(changed)
        self.assertEqual(report['train_episodes'],other['train_episodes'])
        self.assertEqual(report['validation_episodes'],other['validation_episodes'])
        self.assertEqual(report['split'],other['split'])

    def test_rare_class_is_present_in_both_splits(self):
        counts={f'common-{i}':dict(wheel=1,fold=1,omni=0) for i in range(10)}
        counts.update(rare_a=dict(wheel=1,fold=1,omni=4),rare_b=dict(wheel=1,fold=1,omni=4))
        heldout,manifest=stratified_groups(counts)
        self.assertIsNotNone(heldout)
        self.assertEqual(len(heldout & {'rare_a','rare_b'}),1)
        for label in ChassisClassifier.labels:
            self.assertGreaterEqual(sum(counts[g][label] for g in heldout),4)
            self.assertGreaterEqual(sum(counts[g][label] for g in counts if g not in heldout),4)
        self.assertTrue(manifest['exhaustive'])

    def test_infeasible_grouped_class_never_fits_or_enables_classifier(self):
        data=rows()
        for row in data:
            # Plenty of rare tracks, but none can be divided without seed leak.
            row['split_group']='single-rare-seed' if row['label']=='omni' else row['episode_key']
        with patch.object(ChassisClassifier,'fit',side_effect=AssertionError('must not fit')):
            model,report=fit_validate(data)
        self.assertFalse(model['validated'])
        self.assertEqual(report['reason'],'insufficient_grouped_class_support')
        # Even two rare seeds can be indivisible at the four-track threshold.
        heldout,manifest=stratified_groups(dict(a=dict(wheel=4,fold=4,omni=7),b=dict(wheel=4,fold=4,omni=1)))
        self.assertIsNone(heldout); self.assertEqual(manifest['reason'],'no_feasible_group_split')

    def test_large_group_search_is_bounded_and_order_independent(self):
        counts={f'seed-{i}':dict(wheel=1,fold=1,omni=1) for i in range(128)}
        selected,manifest=stratified_groups(counts)
        other,again=stratified_groups(dict(reversed(list(counts.items()))))
        self.assertEqual(selected,other); self.assertEqual(manifest,again)
        self.assertLessEqual(manifest['candidates_evaluated'],4096)
        self.assertFalse(manifest['exhaustive']); self.assertIsNotNone(selected)

    def test_independent_episode_split_and_validated_separable_tracks(self):
        model,report=fit_validate(rows())
        self.assertTrue(model['validated'])
        self.assertFalse(set(report['train_episodes']) & set(report['validation_episodes']))
        self.assertEqual(sum(report['train_counts'].values())+sum(report['validation_counts'].values()),72)
        self.assertGreaterEqual(min(report['validation_counts'].values()),4)
        probabilities,supported=ChassisClassifier(model).predict([2.51,3.,.4,.1])
        self.assertTrue(supported); self.assertGreater(probabilities[1],.7)

    def test_unidentifiable_classes_and_insufficient_heldout_remain_unknown(self):
        model,report=fit_validate(rows(overlap=True))
        self.assertFalse(model['validated']); self.assertEqual(report['reason'],'failed_independent_calibration')
        self.assertEqual(ChassisClassifier(model).predict([.5,1.,.1,.1]),([1/3]*3,False))
        model,report=fit_validate(rows()[:9])
        self.assertFalse(model['validated']); self.assertIn('insufficient',report['reason'])

    def test_duplicate_tracks_cannot_inflate_support(self):
        data=rows()
        with self.assertRaisesRegex(ValueError,'Duplicate'): fit_validate(data+[copy.deepcopy(data[0])])

    def test_same_physical_seed_across_roles_stays_in_one_split(self):
        data=rows()
        for row in data:row['split_group']='shared-physical-seed'
        model,report=fit_validate(data)
        self.assertFalse(model['validated'])
        self.assertEqual(report['episodes'],1)
        self.assertFalse(report['train_episodes'])

    def test_recorder_copies_only_existing_legal_features_and_groups_windows(self):
        estimator=RadarInference('red')
        record=estimator.motion[('blue',3)]
        record.update(features=[1.,2.,.2,.3],features_time=20.)
        gateway=SimpleNamespace(inference={'red':estimator},snapshot=dict(time=20.,robots=[
            dict(team='blue',slot=3,transform_allowed=True,chassis_type='mecanum',max_speed=99.,true_accuracy=1.)]))
        match=SimpleNamespace(sentry_gateway=gateway); recorder=RadarCalibrationRecorder()
        self.assertEqual(recorder.observe(match,'episode42'),1)
        self.assertEqual(recorder.observe(match,'episode42'),0)
        gateway.snapshot['time']=21.;record['features_time']=21.
        self.assertEqual(recorder.observe(match,'episode42'),1)
        result=recorder.rows()
        self.assertEqual(len(result),1);self.assertEqual(result[0]['window_count'],2)
        self.assertEqual(result[0]['label'],'fold');self.assertEqual(result[0]['features'],[1.,2.,.2,.3])
        self.assertNotIn('true_accuracy',result[0]);self.assertNotIn('max_speed',result[0])
        self.assertFalse(estimator.classifier.model.get('validated',False))


if __name__=='__main__': unittest.main()
