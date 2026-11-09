import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import fusion_mechanism_delivery as delivery
import run_fusion_mechanism as runner


def parser():
    value = argparse.ArgumentParser()
    value.add_argument('--resume')
    value.add_argument('--epochs', type=int)
    value.add_argument('--smoke', action='store_true')
    value.add_argument('--check-only', action='store_true')
    value.add_argument('--deadline-unix', type=float)
    value.add_argument('--to', default='recipient@example.com')
    value.add_argument('--fallback-os-shutdown', action='store_true')
    return value


class FakeChild:
    def __init__(self, code=0, output=None):
        self.stdout = iter(['All controls and diagnostics ended\n']) if output is None else output
        self.code = code

    def __enter__(self): return self
    def __exit__(self, *args): return False
    def wait(self, timeout=None): return self.code
    def poll(self): return self.code


def package():
    return dict(full=Path('reports.zip'), compact=None, attachment=Path('reports.zip'))


class FusionMechanismDeliveryTests(unittest.TestCase):
    def fixture(self, root, status='complete'):
        suite = root/'suite'; suite.mkdir()
        (suite/'training').mkdir(); (suite/'feature_analysis').mkdir()
        (suite/'suite.json').write_text(json.dumps(dict(protocol='fixture', epochs=30)))
        (suite/'completion.json').write_text(json.dumps(dict(status=status,
            outcomes=dict(training=dict(status=status), feature_analysis=dict(status=status)))))
        (suite/'training/summary.json').write_text('{"status":"complete"}')
        (suite/'feature_analysis/paired_summary.json').write_text('{"corrected":455,"harmed":390}')
        module = SimpleNamespace(make_plan=MagicMock(return_value=dict(epochs=30)),
            create_suite=MagicMock(return_value=suite),
            validate_resume=MagicMock(return_value=dict(epochs=30)))
        return suite, module

    def test_one_email_after_child_then_managed_stop_and_no_default_deadline(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, module = self.fixture(Path(temp)); events=[]
            with patch.dict(sys.modules, {'fusion_mechanism_experiment':module}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.object(runner, 'check_email_ready', side_effect=lambda *a:events.append('auth')), \
                    patch.object(runner, 'preflight', side_effect=lambda:events.append('preflight') or 'url'), \
                    patch.object(runner.subprocess, 'Popen', side_effect=lambda *a,**k:events.append('child') or FakeChild()) as child, \
                    patch.object(delivery, 'bundle_results', side_effect=lambda *a:events.append('package') or package()), \
                    patch.object(delivery, 'send_mail', side_effect=lambda *a,**k:events.append('email')) as mail, \
                    patch.object(delivery, 'stop_workbench', side_effect=lambda *a:events.append('stop')) as stop:
                with self.assertRaises(SystemExit) as result: runner.main([])
            self.assertEqual(result.exception.code, 0)
            self.assertEqual(events, ['auth','preflight','child','package','email','stop'])
            self.assertNotIn('--deadline-unix', child.call_args.args[0])
            stop.assert_called_once_with('url',False); mail.assert_called_once()
            record=json.loads((suite/'delivery_status.json').read_text())
            self.assertTrue(record['stop_requested']); self.assertTrue(record['delivery_complete'])
            self.assertFalse(record['managed_stop_confirmed'])
            self.assertIsNone(json.loads((suite/'runner.json').read_text())['default_total_hours'])

    def test_nonzero_child_status_is_preserved_after_delivery_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp),'failed')
            with patch.dict(sys.modules,{'fusion_mechanism_experiment':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'), patch.object(runner,'preflight',return_value='url'), \
                    patch.object(runner.subprocess,'Popen',return_value=FakeChild(7)), \
                    patch.object(delivery,'bundle_results',return_value=package()), \
                    patch.object(delivery,'send_mail') as mail, patch.object(delivery,'stop_workbench') as stop:
                with self.assertRaises(SystemExit) as result: runner.main([])
            self.assertEqual(result.exception.code,7); mail.assert_called_once(); stop.assert_called_once()
            self.assertIn('FAILED / INTERRUPTED',mail.call_args.args[1])

    def test_successful_child_but_failed_delivery_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'fusion_mechanism_experiment':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'), patch.object(runner,'preflight',return_value='url'), \
                    patch.object(runner.subprocess,'Popen',return_value=FakeChild()), \
                    patch.object(delivery,'finish',return_value=False) as finish:
                with self.assertRaises(SystemExit) as result: runner.main([])
            self.assertEqual(result.exception.code,2); finish.assert_called_once()

    def test_check_only_authenticates_and_checks_stop_but_never_arms_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'fusion_mechanism_experiment':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready') as auth, \
                    patch.object(runner,'preflight',return_value='url') as preflight, \
                    patch.object(runner.subprocess,'Popen') as child, \
                    patch.object(delivery,'finish') as finish:
                runner.main(['--check-only'])
            auth.assert_called_once();preflight.assert_called_once()
            module.create_suite.assert_not_called();child.assert_not_called();finish.assert_not_called()
            self.assertFalse((suite/'runner.json').exists())

    def test_preflight_failure_never_creates_suite_or_delivers(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'fusion_mechanism_experiment':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'), \
                    patch.object(runner,'preflight',side_effect=RuntimeError('permission missing')), \
                    patch.object(runner.subprocess,'Popen') as child, patch.object(delivery,'finish') as finish:
                with self.assertRaisesRegex(RuntimeError,'permission missing'): runner.main([])
            module.create_suite.assert_not_called(); child.assert_not_called(); finish.assert_not_called()

    def test_invalid_or_expired_explicit_deadline_never_arms_delivery(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'fusion_mechanism_experiment':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready') as auth, \
                    patch.object(runner,'preflight') as preflight, \
                    patch.object(delivery,'finish') as finish:
                for value in ('nan','inf','0'):
                    with self.assertRaises(ValueError):runner.main(['--deadline-unix',value])
                with self.assertRaisesRegex(RuntimeError,'expired'):
                    runner.main(['--deadline-unix','1'])
            auth.assert_not_called();preflight.assert_not_called();finish.assert_not_called()
            module.create_suite.assert_not_called()

    def test_spawn_failure_does_not_arm_stop_for_a_new_suite(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'fusion_mechanism_experiment':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'), patch.object(runner,'preflight',return_value='url'), \
                    patch.object(runner.subprocess,'Popen',side_effect=OSError('spawn failed')), \
                    patch.object(delivery,'finish') as finish:
                with self.assertRaises(SystemExit) as result: runner.main([])
            self.assertEqual(result.exception.code,1); finish.assert_not_called()
            self.assertFalse((suite/'execution_started.json').exists())

    def test_resumed_started_suite_still_delivers_on_spawn_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp))
            record=runner.runner_record(suite)
            (suite/'runner.json').write_text(json.dumps(record))
            runner.mark_execution_started(suite,record)
            with patch.dict(sys.modules,{'fusion_mechanism_experiment':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'), patch.object(runner,'preflight',return_value='url'), \
                    patch.object(runner.subprocess,'Popen',side_effect=OSError('spawn failed')), \
                    patch.object(delivery,'finish',return_value=False) as finish:
                with self.assertRaises(SystemExit) as result: runner.main(['--resume',str(suite)])
            self.assertEqual(result.exception.code,1); finish.assert_called_once()
            module.create_suite.assert_not_called()

    def test_keyboard_interrupt_stops_owned_child_then_delivers_partial_evidence(self):
        class Interrupted:
            def __iter__(self): raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp));child=FakeChild(output=Interrupted())
            with patch.dict(sys.modules,{'fusion_mechanism_experiment':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'), patch.object(runner,'preflight',return_value='url'), \
                    patch.object(runner.subprocess,'Popen',return_value=child), \
                    patch.object(runner,'_stop_child') as halt, \
                    patch.object(delivery,'finish',return_value=False) as finish:
                with self.assertRaises(SystemExit) as result: runner.main([])
            self.assertEqual(result.exception.code,130);halt.assert_called_once_with(child)
            self.assertEqual(finish.call_args.args[3],130)

    def test_runner_identity_rejects_manifest_source_policy_and_execution_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp));record=runner.runner_record(suite)
            (suite/'runner.json').write_text(json.dumps(record))
            self.assertEqual(runner.validate_runner(suite),record)
            runner.mark_execution_started(suite,record)
            self.assertTrue(runner.execution_has_started(suite,record))
            with patch.object(runner,'source_hashes',return_value={'different':'hash'}):
                with self.assertRaisesRegex(ValueError,'changed'):runner.validate_runner(suite)
            changed=dict(record,instance_policy='keep running')
            with self.assertRaisesRegex(ValueError,'different runner'):runner.execution_has_started(suite,changed)
            (suite/'suite.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'changed'):runner.validate_runner(suite)

    def test_report_zip_excludes_executables_weights_arrays_nested_archives_and_secrets(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp))
            for name in ('source.py','weights.pt','probabilities.npz','old.zip','email.txt','.env','credentials.json'):
                (suite/'training'/name).write_text('excluded')
            bundle=delivery.bundle_results(suite)
            with zipfile.ZipFile(bundle['attachment']) as archive:
                names=archive.namelist();self.assertIsNone(archive.testzip())
                self.assertIn('training_analysis/summary.json',names)
                self.assertIn('feature_analysis/paired_summary.json',names)
                for excluded in ('source.py','weights.pt','probabilities.npz','old.zip','email.txt','.env','credentials.json'):
                    self.assertFalse(any(name.endswith(excluded) for name in names))

    def test_package_failure_still_emails_and_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp));events=[]
            with patch.object(delivery,'bundle_results',side_effect=OSError('full')), \
                    patch.object(delivery,'send_mail',side_effect=lambda *a,**k:events.append('email')) as mail, \
                    patch.object(delivery,'stop_workbench',side_effect=lambda *a:events.append('stop')):
                self.assertFalse(delivery.finish(suite,suite/'run.log','recipient',0,'url'))
            self.assertEqual(events,['email','stop']);self.assertIsNone(mail.call_args.kwargs['attachment'])
            self.assertIn('PACKAGE FAILED',mail.call_args.args[1])

    def test_all_mail_retries_fail_then_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp));events=[]
            def fail(*a,**k):events.append('email');raise RuntimeError('offline')
            with patch.object(delivery,'bundle_results',return_value=package()), \
                    patch.object(delivery,'send_mail',side_effect=fail),patch.object(delivery.time,'sleep'), \
                    patch.object(delivery,'stop_workbench',side_effect=lambda *a:events.append('stop')):
                self.assertFalse(delivery.finish(suite,suite/'run.log','recipient',0,'url'))
            self.assertEqual(events,['email','email','email','stop'])
            self.assertFalse(json.loads((suite/'delivery_status.json').read_text())['email_accepted'])

    def test_record_or_sync_failure_cannot_block_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp))
            with patch.object(delivery,'bundle_results',return_value=package()), \
                    patch.object(delivery,'send_mail'), \
                    patch.object(delivery,'_record_delivery',side_effect=OSError('record full')), \
                    patch.object(delivery,'stop_workbench') as stop:
                with self.assertRaises(OSError):delivery.finish(suite,suite/'run.log','recipient',0,'url')
            stop.assert_called_once()

    def test_stop_failure_is_recorded_and_propagates(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp))
            with patch.object(delivery,'bundle_results',return_value=package()),patch.object(delivery,'send_mail'), \
                    patch.object(delivery,'stop_workbench',side_effect=RuntimeError('stop rejected')):
                with self.assertRaisesRegex(RuntimeError,'stop rejected'):
                    delivery.finish(suite,suite/'run.log','recipient',0,'url')
            record=json.loads((suite/'delivery_status.json').read_text())
            self.assertTrue(record['stop_requested']);self.assertFalse(record['managed_stop_confirmed'])
            self.assertEqual(record['stop_error_type'],'RuntimeError')

    def test_child_environment_strips_mail_credentials(self):
        with patch.dict(os.environ,{'GMAIL_USER':'private','GMAIL_APP_PASSWORD':'private','KEEP':'yes'}):
            env=runner.child_environment()
        self.assertNotIn('GMAIL_USER',env);self.assertNotIn('GMAIL_APP_PASSWORD',env)
        self.assertEqual(env['KEEP'],'yes')
        self.assertEqual(Path(runner.child_command('suite')[2]).name,runner.CORE)


if __name__=='__main__':unittest.main()
