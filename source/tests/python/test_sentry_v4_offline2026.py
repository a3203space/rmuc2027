import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from rmuc2027_research.rl.offline2026 import (
    SOURCE_FIELDS, clean_track, file_hash, prepare, readonly_database, split_series,
    track_speed_statistics, validate_calibration,
)


def rows(role='步兵3', n=10):
    result = []
    for i in range(n):
        row = dict.fromkeys(SOURCE_FIELDS, 0.)
        row.update(game_id=11, robot_id=3, 赛区='东部赛区', 场次号=1, 局号=1,
                   机器人类型=role, 阵营='红', 时刻秒=float(i+1), x=2.+i, y=2.,
                   当前血量=200., 最大血量=200., 底盘功率=50., 累计42mm发弹=None)
        result.append(row)
    return result


class Offline2026Tests(unittest.TestCase):
    def test_series_split_does_not_leak_rounds_or_regions(self):
        matches = [dict(game_id=1000*region+10*match+round_, 赛区=str(region), 场次号=match)
                   for region in range(3) for match in range(1, (88, 90, 88)[region]+1) for round_ in (1, 2)]
        split = split_series(matches)
        self.assertEqual(split['counts'], {'train': 186, 'validation': 40, 'test': 40})
        for row in split['series']:
            self.assertEqual({split['game_split'][str(g)] for g in row['game_ids']}, {row['split']})
        self.assertEqual(split, split_series(list(reversed(matches))))

    def test_unknown_actions_and_unconfirmed_semantics_stay_masked(self):
        result = clean_track(rows())
        self.assertGreater(result['future_source_mask'].sum(), 0)
        self.assertFalse(result['original_action_mask'].any())
        self.assertFalse(result['semantic_goal_mask'].any())
        self.assertFalse(result['field_position_valid'].any())
        self.assertTrue(np.isfinite(result['state']).all())

    def test_death_gap_sentinel_and_spike_break_route_labels(self):
        source = rows(n=12)
        source[3]['当前血量'] = 0.
        source[6]['x'], source[6]['y'] = -1., -1.
        source[9]['时刻秒'] += .25
        result = clean_track(source)
        self.assertFalse(result['future_source_mask'][2].any())
        self.assertFalse(result['source_position_valid'][6])
        self.assertFalse(result['future_source_mask'][8, 0])
        stationary = rows(n=12)
        for row in stationary:
            row['x'] = 5.
        stationary[6]['x'] = 20.
        result = clean_track(stationary)
        self.assertTrue(result['discontinuity'][6])

    def test_engineer_prefix_and_confirmed_semantic_mapping(self):
        source = rows('工程')
        self.assertFalse(clean_track(source)['future_source_mask'].any())
        calibration = dict(confirmed=True, evidence='test landmark fixture',
            source_to_field=[[1, 0, 0], [0, 1, 0]],
            landmarks=[dict(source_xy=p, field_xy=p) for p in ([0, 0], [28, 0], [0, 15])],
            semantic_regions=[dict(name='assembly_approach', field_polygon=[[0,0],[20,0],[20,10],[0,10]],
                                   target_2027_xy=[12.,7.5], target_verified=True)])
        result = clean_track(source, calibration, first_assembly=5.)
        self.assertTrue(result['semantic_goal_mask'][0, 0])
        self.assertFalse(result['semantic_goal_mask'][5:].any())
        self.assertFalse(result['state_valid'][:, 3].any())
        calibration['source_to_field'][0][2] = 5
        with self.assertRaises(ValueError):
            validate_calibration(calibration)

    def test_no_velocity_cap_hides_raw_maximum(self):
        source = rows(n=12)
        source[5]['x'] = 100.
        stats, _ = track_speed_statistics(source)
        self.assertGreater(stats['raw'].max(), 90.)
        self.assertEqual(stats['conditional'].max(), 1.)

    def test_readonly_etl_preserves_database_and_split_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.sqlite'
            c = sqlite3.connect(source)
            c.execute('CREATE TABLE matches (game_id INT,赛区 TEXT,场次号 INT,局号 INT,时长秒 INT,胜方 TEXT)')
            fields = ','.join('"'+f+'" REAL' for f in SOURCE_FIELDS)
            c.execute('CREATE TABLE timeseries (game_id INT,robot_id INT,赛区 TEXT,机器人类型 TEXT,阵营 TEXT,时刻秒 REAL,'+fields+')')
            c.execute('CREATE TABLE events (game_id INT,时刻秒 REAL,事件类型 TEXT,阵营 TEXT,数值 REAL)')
            for game in range(1, 5):
                c.execute('INSERT INTO matches VALUES (?,?,?,?,?,?)', (game,'东部赛区',game,1,11,'红'))
                for row in rows():
                    values = [game, row['robot_id'], row['赛区'], row['机器人类型'], row['阵营'], row['时刻秒'], *[row[f] for f in SOURCE_FIELDS]]
                    c.execute('INSERT INTO timeseries VALUES ('+','.join('?' for _ in values)+')', values)
            c.commit(); c.close()
            digest = file_hash(source)
            with readonly_database(source) as readonly:
                with self.assertRaises(sqlite3.OperationalError):
                    readonly.execute('DELETE FROM timeseries')
            report = prepare(source, Path(directory)/'prepared', split_counts=(2,1,1))
            self.assertEqual(file_hash(source), digest)
            self.assertTrue(report['source_unchanged'])
            self.assertEqual(report['gradient_updates'], 0)
            self.assertEqual(report['etl_source']['sha256'], file_hash(Path(directory)/'prepared'/report['etl_source']['path']))
            self.assertFalse(report['calibration_targets']['ready'])
            for artifact in report['files']:
                with np.load(Path(directory)/'prepared'/artifact['path'], allow_pickle=False) as data:
                    self.assertFalse(data['original_action_mask'].any())
                    self.assertFalse(data['semantic_goal_mask'].any())
            self.assertEqual(report['split_hash'], json.loads((Path(directory)/'prepared/split.json').read_text())['split_hash'])


if __name__ == '__main__':
    unittest.main()
