import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from rmuc2027_research.rl.scenarios import ScenarioSpec
from rmuc2027_research.rl.v6_replay_catalog import metadata_header,recorded_scenario,enrich_replay_labels


class ReplayCatalogTests(unittest.TestCase):
    def test_records_both_sides_without_requiring_live_presets(self):
        for side in ('red','blue'):
            case=ScenarioSpec((303,303),(2732,2731),2771,side)
            meta={'rl':{'config':{'team':side,'lineups':case.lineups()}}}
            self.assertEqual(recorded_scenario(meta),case.scenario_id)
        with self.assertRaises(ValueError):recorded_scenario({'rl':{'config':{'team':'blue'}}})

    def test_header_reader_stops_before_unparseable_frame_array(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'replay.json'
            metadata={'rl':{'config':{'team':'red'}},'note':'中文 "metadata"'}
            path.write_text('{"format":"rmuc-replay-v4","metadata":'+json.dumps(metadata)+',"frames":[not-read')
            self.assertEqual(metadata_header(path),metadata)
            with self.assertRaises(ValueError):metadata_header(path,max_characters=10)

    def test_only_replay_display_rows_change_and_recordings_stay_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);case=ScenarioSpec((303,303),(2731,2731),2772,'blue')
            metadata={'map':'rmuc2027','test_match':True,'rl':{'config':{'team':'blue','lineups':case.lineups()}}}
            replay=root/'replay.json';replay.write_text(json.dumps({'format':'rmuc-replay-v4','metadata':metadata,'frames':[]}))
            digest=hashlib.sha256(replay.read_bytes()).hexdigest()
            manifest={'scenarios':{'registered-red':{'fixed':True}},'capability_profiles':{'only-red':{}},'checkpoints':{},'helpers':{},
                'replays':{'sample':{'path':'replay.json','sha256':digest,'kind':'training'}}}
            original=copy.deepcopy(manifest);(root/'run-manifest.json').write_text(json.dumps(manifest))
            (root/'campaign.json').write_text('{"status":"training"}')
            with self.assertRaisesRegex(ValueError,'web_acceptance'):enrich_replay_labels(root)
            (root/'campaign.json').write_text('{"status":"web_acceptance"}')
            result=enrich_replay_labels(root);updated=json.loads((root/'run-manifest.json').read_text())
            self.assertEqual(updated['replays']['sample']['scenario_id'],case.scenario_id)
            for key in ('scenarios','capability_profiles','checkpoints','helpers'):self.assertEqual(updated[key],original[key])
            self.assertEqual(hashlib.sha256(replay.read_bytes()).hexdigest(),digest)
            self.assertEqual(len(result['updated']),1)
            self.assertFalse(result['live_registry_changed'])
            self.assertEqual(enrich_replay_labels(root)['updated'],[])


if __name__=='__main__':unittest.main()
