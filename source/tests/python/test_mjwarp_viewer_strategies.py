import json
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rmuc2027_core.submissions import load_submission, save_submission
from rmuc2027_research.rl.mjwarp.viewer import prepare_match
from rmuc2027_research.rl.session import load_config, resource


class ViewerStrategyTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(config=None, red=None, blue=None, rl_model=None, rl_team=None, device='cuda:0')
        return Namespace(**{**values, **overrides})

    def test_scripted_red_and_blue_are_selected_without_loading_rl(self):
        with ExitStack() as stack:
            cfg, sources, side = prepare_match(self.args(
                red='examples/claude_opus55_xhigh_2027_hold',
                blue='examples/claude_opus55_xhigh_2027_squad'), stack)
            self.assertIsNone(side)
            self.assertEqual(sources['red'].name, 'claude_opus55_xhigh_2027_hold')
            self.assertEqual(sources['blue'].name, 'claude_opus55_xhigh_2027_squad')
            self.assertEqual(set(cfg['lineups']), {'red', 'blue'})
            self.assertEqual(cfg['sim_backend'], 'mjwarp')

    def test_only_selected_side_uses_checkpoint_squad_and_temp_files_survive_view(self):
        cfg = load_config('configs/rl/sentry-mjwarp-hour.json')
        manifest, entries = load_submission(resource(cfg['friendly']))

        def export(model, output):
            save_submission(output, 'squad', entries, season='2027', team_lineup=manifest.get('team_lineup'))
            return {'learned_slots': [7]}

        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)/'model.zip'; checkpoint.touch()
            (Path(temp)/'experiment.json').write_text(json.dumps({'config': cfg}))
            for side in ('red', 'blue'):
                other = 'blue' if side == 'red' else 'red'
                args = self.args(rl_model=str(checkpoint), rl_team=side,
                                 **{other: 'examples/claude_opus55_xhigh_2027_hold'})
                with ExitStack() as stack, patch('rmuc2027_research.rl.exporting.export_model', side_effect=export):
                    actual, sources, learned_side = prepare_match(args, stack)
                    self.assertEqual(learned_side, side)
                    self.assertEqual(sources[other].name, 'claude_opus55_xhigh_2027_hold')
                    self.assertEqual(actual['learning_control_hz'], 2)
                    self.assertEqual(actual['rl_seconds'], .5)
                    _, selected = load_submission(sources[side])
                    self.assertEqual(set(selected), {'1', '3', '4', '6', '7'})
                    for slot in ('1', '3', '4', '6'):
                        self.assertEqual(selected[slot].read_bytes(), entries[slot].read_bytes())
                    temporary_squad = sources[side]
                    self.assertTrue(temporary_squad.exists())
                self.assertFalse(temporary_squad.exists())

    def test_rl_team_without_model_is_rejected(self):
        with ExitStack() as stack, self.assertRaisesRegex(ValueError, 'requires --rl-model'):
            prepare_match(self.args(rl_team='blue'), stack)

    def test_non_sentry_checkpoint_is_rejected_before_export(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'model.zip';checkpoint.touch()
            (Path(temp)/'experiment.json').write_text(json.dumps({'config': {'slot': 3}}))
            with ExitStack() as stack, self.assertRaisesRegex(ValueError, 'sentry'):
                prepare_match(self.args(rl_model=str(checkpoint)), stack)

    def test_explicit_scripted_source_on_rl_side_is_not_silently_ignored(self):
        cfg=load_config('configs/rl/sentry-mjwarp-hour.json')
        with tempfile.TemporaryDirectory() as temp:
            checkpoint=Path(temp)/'model.zip';checkpoint.touch()
            (Path(temp)/'experiment.json').write_text(json.dumps({'config': cfg}))
            with ExitStack() as stack, self.assertRaisesRegex(ValueError, 'cannot accompany'):
                prepare_match(self.args(rl_model=str(checkpoint),red=cfg['friendly']), stack)

    def test_exported_single_sentry_is_detected_and_two_rl_teams_are_rejected(self):
        cfg=load_config('configs/rl/sentry-mjwarp-hour.json')
        manifest, entries=load_submission(resource(cfg['friendly']))
        with tempfile.TemporaryDirectory() as temp:
            squad=Path(temp)/'export'
            save_submission(squad,'squad',entries,season='2027',team_lineup=manifest.get('team_lineup'))
            (squad/'rl-metadata.json').write_text(json.dumps({'learned_slots':[7]}))
            with ExitStack() as stack:
                _, sources, side=prepare_match(self.args(red=str(squad)),stack)
                self.assertEqual(side,'red')
                self.assertEqual(sources['red'],squad)
            with ExitStack() as stack, self.assertRaisesRegex(ValueError,'one RL sentry'):
                prepare_match(self.args(red=str(squad),blue=str(squad)),stack)
