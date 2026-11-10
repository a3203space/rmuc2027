import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from rmuc2027_research.rl import sentry_campaign


class SentryDeadlineTests(unittest.TestCase):
    def test_timestamped_intervention_is_recorded_and_worker_stopped(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            request=dict(model=2771,restart=True,time_unix=123.,reason='measured regression')
            (folder/'intervention.json').write_text(json.dumps(request))
            report=dict(phase='training',active_model=2771)
            process=MagicMock();process.poll.return_value=None
            with patch.object(sentry_campaign.subprocess,'Popen',return_value=process), patch.object(sentry_campaign,'stop_worker') as stop:
                code=sentry_campaign.child(['unused'],folder/'worker.log',time.time()+100,report,folder/'campaign.json')
            self.assertEqual(code,-77);self.assertTrue(stop.called)
            self.assertEqual(report['interventions'][0]['time_unix'],123.)
            self.assertIn('applied_unix',report['interventions'][0])
            self.assertTrue((folder/'intervention-applied-1.json').exists())

    def test_recovery_preserves_expired_budget_and_cannot_start_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);start=time.time()-7300
            budget=dict(started_unix=start,deadline_unix=start+7200,gradient_deadline_unix=start+5700,wall_budget_seconds=7200.,reserve_seconds=1500.)
            report=dict(**budget,phase='training',active_model=2771,active_directory=str(folder/'2771'),training=[],evaluations=[],models=[2771,2772,2773])
            original=json.dumps(budget);(folder/'budget.json').write_text(original)
            (folder/'campaign.json').write_text(json.dumps(report))
            with patch.object(sentry_campaign,'child',side_effect=AssertionError('expired worker started')), patch.object(sentry_campaign.zipfile,'ZipFile'), contextlib.redirect_stdout(io.StringIO()):
                sentry_campaign.run(folder,resume_run=True)
            after=json.loads((folder/'campaign.json').read_text())
            self.assertEqual((folder/'budget.json').read_text(),original)
            self.assertEqual(after['deadline_unix'],budget['deadline_unix'])
            self.assertFalse(after['all_models_have_checkpoints'])


if __name__=='__main__':unittest.main()
