import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import run_best_pair_joint as runner


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--resume'); p.add_argument('--epochs', type=int)
    p.add_argument('--smoke', action='store_true'); p.add_argument('--check-only', action='store_true')
    p.add_argument('--to', default='recipient@example.com')
    return p


class FakeChild:
    def __init__(self, code=0, output=None):
        self.stdout = output if output is not None else iter(['Training and analysis complete\n'])
        self.code = code; self.terminated = False
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def wait(self, timeout=None): return self.code
    def terminate(self): self.terminated = True
    def kill(self): self.terminated = True


def package():
    return dict(full=Path('reports.zip'), compact=None, attachment=Path('reports.zip'))


class BestPairEmailTests(unittest.TestCase):
    def fixture(self, root):
        suite = root/'suite'; suite.mkdir()
        for name in ('training', 'feature_analysis'): (suite/name).mkdir()
        (suite/'completion.json').write_text(json.dumps(dict(status='complete', outcomes={
            'training':dict(status='complete'), 'feature_analysis':dict(status='complete')})))
        module = SimpleNamespace(make_plan=MagicMock(return_value=dict(fusion_epochs=30)),
            create_suite=MagicMock(return_value=suite), validate_resume=MagicMock(return_value=dict(fusion_epochs=30)))
        return suite, module

    def test_success_emails_once_after_child_and_never_stops_instance(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, module = self.fixture(Path(temp)); events=[]
            with patch.dict(sys.modules, {'best_pair_joint':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'), \
                    patch.object(runner.subprocess,'Popen',side_effect=lambda *a,**k:(events.append('child') or FakeChild())), \
                    patch.object(runner,'bundle_results',side_effect=lambda *a:(events.append('package') or package())), \
                    patch.object(runner,'send_mail',side_effect=lambda *a,**k:events.append('mail')) as mail, \
                    patch('run_vit_augreg_overnight.stop_workbench') as stop, \
                    patch.object(runner.subprocess,'run') as command:
                with self.assertRaises(SystemExit) as exit:
                    runner.main([])
            self.assertEqual(exit.exception.code,0)
            self.assertEqual(events,['child','package','mail'])
            mail.assert_called_once(); stop.assert_not_called(); command.assert_not_called()
            delivery=json.loads((suite/'delivery_status.json').read_text())
            self.assertFalse(delivery['stop_requested']); self.assertTrue(delivery['delivery_complete'])

    def test_failed_training_still_emails_and_keeps_instance_running(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'best_pair_joint':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'), \
                    patch.object(runner.subprocess,'Popen',return_value=FakeChild(code=7)), \
                    patch.object(runner,'bundle_results',return_value=package()), \
                    patch.object(runner,'send_mail') as mail, \
                    patch('run_vit_augreg_overnight.stop_workbench') as stop:
                with self.assertRaises(SystemExit) as exit: runner.main([])
            self.assertEqual(exit.exception.code,7); mail.assert_called_once(); stop.assert_not_called()
            self.assertFalse(json.loads((suite/'delivery_status.json').read_text())['stop_requested'])

    def test_mail_failure_never_triggers_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp))
            with patch.object(runner,'bundle_results',return_value=package()), \
                    patch.object(runner,'send_mail',side_effect=RuntimeError('offline')) as mail, \
                    patch.object(runner.time,'sleep'), patch('run_vit_augreg_overnight.stop_workbench') as stop:
                self.assertFalse(runner.finish(suite,suite/'run.log','recipient@example.com',0))
            self.assertEqual(mail.call_count,3);stop.assert_not_called()
            delivery=json.loads((suite/'delivery_status.json').read_text())
            self.assertFalse(delivery['email_accepted']); self.assertFalse(delivery['stop_requested'])

    def test_package_failure_sends_failure_notice_without_stopping(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp))
            with patch.object(runner,'bundle_results',side_effect=OSError('fixture')), \
                    patch.object(runner,'send_mail') as mail, patch('run_vit_augreg_overnight.stop_workbench') as stop:
                self.assertFalse(runner.finish(suite,suite/'run.log','recipient@example.com',0))
            self.assertIsNone(mail.call_args.kwargs['attachment']);stop.assert_not_called()

    def test_check_only_has_no_training_email_or_cloud_control(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'best_pair_joint':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'), \
                    patch.object(runner.subprocess,'Popen') as child, \
                    patch.object(runner,'send_mail') as mail, \
                    patch('run_vit_augreg_overnight.stop_workbench') as stop, \
                    patch.object(runner.subprocess,'run') as command:
                runner.main(['--check-only'])
            module.create_suite.assert_not_called();child.assert_not_called();mail.assert_not_called()
            stop.assert_not_called();command.assert_not_called()

    def test_child_does_not_receive_mail_credentials(self):
        with patch.dict(os.environ,{'GMAIL_USER':'private','GMAIL_APP_PASSWORD':'secret','KEEP':'visible'}):
            env=runner.child_environment()
        self.assertNotIn('GMAIL_USER',env);self.assertNotIn('GMAIL_APP_PASSWORD',env)
        self.assertEqual(env['KEEP'],'visible')
        self.assertEqual(Path(runner.child_command('suite')[2]).name,'best_pair_joint.py')

    def test_real_parser_rejects_shutdown_flag(self):
        with self.assertRaises(SystemExit):
            runner.arguments().parse_args(['--fallback-os-shutdown'])


if __name__=='__main__': unittest.main()
