"""CLI forwarding/rejection only; no bundle extraction, models or simulation."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rmuc2027_research.rl import v6_campaign


class ResumeCliTests(unittest.TestCase):
    def test_run_forwards_explicit_baseline_and_legacy_default(self):
        for generation in ('baseline',None):
            with self.subTest(generation=generation):
                argv=['v6_campaign','run','--output','/unused/new-run','--resume-bundle','/unused/bundle']
                if generation:argv+=['--resume-generation',generation]
                with patch('sys.argv',argv),patch.object(v6_campaign,'run') as run:
                    v6_campaign.main()
                run.assert_called_once_with(Path('/unused/new-run'),None,
                    resume_bundle=Path('/unused/bundle'),resume_generation=generation or 'current')

    def test_initialize_resume_forwards_generation_then_refreshes_registry(self):
        for generation in ('baseline',None):
            with self.subTest(generation=generation):
                argv=['v6_campaign','initialize-resume','--output','/unused/new-run','--resume-bundle','/unused/bundle']
                if generation:argv+=['--resume-generation',generation]
                with patch('sys.argv',argv),patch('rmuc2027_research.rl.v6_bundle.prepare_resume_run') as prepare,patch.object(v6_campaign,'refresh_manifest') as refresh:
                    v6_campaign.main()
                prepare.assert_called_once_with(Path('/unused/bundle'),Path('/unused/new-run'),generation=generation or 'current')
                refresh.assert_called_once_with(Path('/unused/new-run'))

    def test_v4_parent_cannot_silently_ignore_baseline_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'must-not-create'
            for command in ('run','initialize-resume'):
                with self.subTest(command=command):
                    argv=['v6_campaign',command,'--output',str(output),'--parent','/unused/v4-parent','--resume-generation','baseline']
                    if command=='initialize-resume':argv+=['--resume-bundle','/unused/v6-bundle']
                    with patch('sys.argv',argv),patch('rmuc2027_research.rl.v6_bundle.prepare_resume_run') as prepare,patch.object(v6_campaign,'refresh_manifest') as refresh,patch.object(v6_campaign,'child') as child:
                        with self.assertRaises((ValueError,SystemExit)):
                            v6_campaign.main()
                    prepare.assert_not_called();refresh.assert_not_called();child.assert_not_called()
                    self.assertFalse(output.exists())

    def test_initialize_resume_requires_bundle_and_never_ignores_parent(self):
        for extra in ([],['--parent','/unused/v4-parent'],
                      ['--resume-bundle','/unused/bundle','--parent','/unused/v4-parent']):
            with self.subTest(extra=extra):
                argv=['v6_campaign','initialize-resume','--output','/unused/new-run',*extra]
                with patch('sys.argv',argv),patch('rmuc2027_research.rl.v6_bundle.prepare_resume_run') as prepare,patch.object(v6_campaign,'refresh_manifest') as refresh:
                    with self.assertRaises((ValueError,SystemExit)):
                        v6_campaign.main()
                prepare.assert_not_called();refresh.assert_not_called()


if __name__=='__main__':unittest.main()
