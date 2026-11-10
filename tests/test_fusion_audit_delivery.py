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

import fusion_audit_delivery as delivery
import run_fusion_audit as runner


def parser():
    value = argparse.ArgumentParser()
    value.add_argument('--resume')
    value.add_argument('--to', default='recipient@example.com')
    value.add_argument('--check-only', action='store_true')
    value.add_argument('--fallback-os-shutdown', action='store_true')
    return value


class FakeChild:
    def __init__(self, code=0, output=None):
        self.code = code
        self.stdout = iter(['Fixed evaluation and analysis complete\n']) if output is None else output
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def wait(self, timeout=None): return self.code
    def poll(self): return self.code


def package():
    return dict(full=Path('reports.zip'), compact=None, attachment=Path('reports.zip'))


class FusionAuditDeliveryTests(unittest.TestCase):
    def fixture(self, root, status='complete'):
        suite = root/'registered_audit'; suite.mkdir()
        (suite/'feature_analysis').mkdir()
        (suite/'suite.json').write_text(json.dumps(dict(protocol='fixture', training_epochs=0)))
        (suite/'completion.json').write_text(json.dumps(dict(status=status,
            official_validation_evaluated=True, failure=None,
            primary_decision=dict(status='no_evidence_of_superiority'))))
        (suite/'feature_analysis'/'summary.json').write_text('{"corrected":455,"harmed":390}')
        module = SimpleNamespace(make_plan=MagicMock(return_value=dict(training_epochs=0)),
            create_suite=MagicMock(return_value=suite), validate_resume=MagicMock(return_value=dict(training_epochs=0)))
        return suite, module

    def test_one_email_after_fixed_evaluation_then_managed_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, module = self.fixture(Path(temp)); events=[]
            module.create_suite.side_effect=lambda *a,**k:events.append('register') or suite
            with patch.dict(sys.modules, {'fusion_confirmatory_audit':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready',side_effect=lambda *a:events.append('auth')), \
                    patch.object(runner,'preflight',side_effect=lambda:events.append('permission') or 'url'), \
                    patch.object(runner.subprocess,'Popen',side_effect=lambda *a,**k:events.append('evaluate') or FakeChild()) as child, \
                    patch.object(delivery,'bundle_results',side_effect=lambda *a:events.append('package') or package()), \
                    patch.object(delivery,'send_mail',side_effect=lambda *a,**k:events.append('email')) as mail, \
                    patch.object(delivery,'stop_workbench',side_effect=lambda *a:events.append('stop')) as stop:
                with self.assertRaises(SystemExit) as result: runner.main([])
            self.assertEqual(result.exception.code,0)
            self.assertEqual(events,['auth','permission','register','evaluate','package','email','stop'])
            mail.assert_called_once(); stop.assert_called_once_with('url',False)
            self.assertEqual(Path(child.call_args.args[0][2]).name,runner.CORE)
            self.assertNotIn('--epochs',child.call_args.args[0])
            record=json.loads((suite/'runner.json').read_text())
            self.assertEqual(record['training_epochs'],0)
            self.assertEqual(record['target_partition'],'official_validation')
            self.assertTrue(json.loads((suite/'delivery_status.json').read_text())['delivery_complete'])

    def test_no_evidence_of_superiority_still_means_successful_assessment(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp))
            self.assertEqual(delivery.completion_status(suite,0),'SUCCESS')
            data=json.loads((suite/'completion.json').read_text())
            data['official_validation_evaluated']=False
            (suite/'completion.json').write_text(json.dumps(data))
            self.assertEqual(delivery.completion_status(suite,0),'FAILED / AUDIT SCOPE CHANGED')

    def test_check_only_preflight_never_registers_evaluates_or_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'fusion_confirmatory_audit':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready') as auth, \
                    patch.object(runner,'preflight',return_value='url') as permission, \
                    patch.object(runner.subprocess,'Popen') as child, \
                    patch.object(delivery,'finish') as finish:
                runner.main(['--check-only'])
            auth.assert_called_once();permission.assert_called_once()
            module.create_suite.assert_not_called();child.assert_not_called();finish.assert_not_called()
            self.assertFalse((suite/'runner.json').exists())

    def test_preflight_failure_does_not_register_or_arm_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            _,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'fusion_confirmatory_audit':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'), \
                    patch.object(runner,'preflight',side_effect=RuntimeError('permission denied')), \
                    patch.object(runner.subprocess,'Popen') as child, \
                    patch.object(delivery,'finish') as finish:
                with self.assertRaisesRegex(RuntimeError,'permission denied'):runner.main([])
            module.create_suite.assert_not_called();child.assert_not_called();finish.assert_not_called()

    def test_new_spawn_failure_is_not_armed(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'fusion_confirmatory_audit':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'),patch.object(runner,'preflight',return_value='url'), \
                    patch.object(runner.subprocess,'Popen',side_effect=OSError('spawn failed')), \
                    patch.object(delivery,'finish') as finish:
                with self.assertRaises(SystemExit) as result:runner.main([])
            self.assertEqual(result.exception.code,1);finish.assert_not_called()
            self.assertFalse((suite/'execution_started.json').exists())

    def test_nonzero_child_preserved_after_failure_email_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp),'failed')
            with patch.dict(sys.modules,{'fusion_confirmatory_audit':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'),patch.object(runner,'preflight',return_value='url'), \
                    patch.object(runner.subprocess,'Popen',return_value=FakeChild(7)), \
                    patch.object(delivery,'bundle_results',return_value=package()), \
                    patch.object(delivery,'send_mail') as mail,patch.object(delivery,'stop_workbench') as stop:
                with self.assertRaises(SystemExit) as result:runner.main([])
            self.assertEqual(result.exception.code,7);mail.assert_called_once();stop.assert_called_once()
            self.assertIn('FAILED / INTERRUPTED',mail.call_args.args[1])

    def test_resumed_started_audit_delivers_on_spawn_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp));record=runner.runner_record(suite)
            (suite/'runner.json').write_text(json.dumps(record));runner.mark_execution_started(suite,record)
            with patch.dict(sys.modules,{'fusion_confirmatory_audit':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'),patch.object(runner,'preflight',return_value='url'), \
                    patch.object(runner.subprocess,'Popen',side_effect=OSError('spawn failed')), \
                    patch.object(delivery,'finish',return_value=False) as finish:
                with self.assertRaises(SystemExit) as result:runner.main(['--resume',str(suite)])
            self.assertEqual(result.exception.code,1);finish.assert_called_once()
            module.create_suite.assert_not_called()

    def test_successful_child_but_failed_delivery_has_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temp:
            _,module=self.fixture(Path(temp))
            with patch.dict(sys.modules,{'fusion_confirmatory_audit':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'),patch.object(runner,'preflight',return_value='url'), \
                    patch.object(runner.subprocess,'Popen',return_value=FakeChild()), \
                    patch.object(delivery,'finish',return_value=False):
                with self.assertRaises(SystemExit) as result:runner.main([])
            self.assertEqual(result.exception.code,2)

    def test_interrupt_halts_owned_child_and_delivers_partial_report(self):
        class Interrupted:
            def __iter__(self):raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temp:
            suite,module=self.fixture(Path(temp));child=FakeChild(output=Interrupted())
            with patch.dict(sys.modules,{'fusion_confirmatory_audit':module}), \
                    patch.object(runner,'arguments',side_effect=parser), \
                    patch('torch.cuda.is_available',return_value=True), \
                    patch.object(runner,'check_email_ready'),patch.object(runner,'preflight',return_value='url'), \
                    patch.object(runner.subprocess,'Popen',return_value=child), \
                    patch.object(runner,'_stop_child') as halt, \
                    patch.object(delivery,'finish',return_value=False) as finish:
                with self.assertRaises(SystemExit) as result:runner.main([])
            self.assertEqual(result.exception.code,130);halt.assert_called_once_with(child)
            self.assertEqual(finish.call_args.args[3],130)

    def test_runner_identity_rejects_changed_manifest_and_delivery_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp));record=runner.runner_record(suite)
            (suite/'runner.json').write_text(json.dumps(record))
            self.assertEqual(runner.validate_runner(suite),record)
            runner.mark_execution_started(suite,record)
            self.assertTrue(runner.execution_has_started(suite,record))
            with self.assertRaisesRegex(ValueError,'different runner'):
                runner.execution_has_started(suite,dict(record,training_epochs=30))
            (suite/'suite.json').write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'changed'):runner.validate_runner(suite)

    def test_zip_excludes_code_weights_arrays_secrets_and_has_no_training_component(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp))
            excluded=('source.py','weights.pt','probabilities.npz','nested.zip','email.txt','.env','credentials.json')
            for name in excluded:(suite/'feature_analysis'/name).write_text('excluded')
            bundle=delivery.bundle_results(suite)
            with zipfile.ZipFile(bundle['attachment']) as archive:
                names=archive.namelist();self.assertIsNone(archive.testzip())
                self.assertIn('feature_analysis/summary.json',names)
                self.assertIn('suite/suite.json',names)
                self.assertFalse(any(name.startswith('training') for name in names))
                for name in excluded:self.assertFalse(any(path.endswith(name) for path in names))
                manifest=json.loads(archive.read('delivery_manifest.json'))
                self.assertEqual(manifest['training_epochs'],0)
                self.assertEqual(manifest['target_partition'],'official_validation')

    def test_package_failure_still_sends_notice_then_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp));events=[]
            with patch.object(delivery,'bundle_results',side_effect=OSError('disk full')), \
                    patch.object(delivery,'send_mail',side_effect=lambda *a,**k:events.append('mail')) as mail, \
                    patch.object(delivery,'stop_workbench',side_effect=lambda *a:events.append('stop')):
                self.assertFalse(delivery.finish(suite,suite/'run.log','recipient',0,'url'))
            self.assertEqual(events,['mail','stop']);self.assertIsNone(mail.call_args.kwargs['attachment'])

    def test_all_mail_failures_still_request_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp));events=[]
            def failed(*a,**k):events.append('mail');raise RuntimeError('offline')
            with patch.object(delivery,'bundle_results',return_value=package()), \
                    patch.object(delivery,'send_mail',side_effect=failed),patch.object(delivery.time,'sleep'), \
                    patch.object(delivery,'stop_workbench',side_effect=lambda *a:events.append('stop')):
                self.assertFalse(delivery.finish(suite,suite/'run.log','recipient',0,'url'))
            self.assertEqual(events,['mail','mail','mail','stop'])

    def test_stop_failure_is_recorded_not_claimed_confirmed(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp))
            with patch.object(delivery,'bundle_results',return_value=package()),patch.object(delivery,'send_mail'), \
                    patch.object(delivery,'stop_workbench',side_effect=RuntimeError('stop rejected')):
                with self.assertRaisesRegex(RuntimeError,'stop rejected'):
                    delivery.finish(suite,suite/'run.log','recipient',0,'url')
            record=json.loads((suite/'delivery_status.json').read_text())
            self.assertFalse(record['managed_stop_confirmed']);self.assertEqual(record['stop_error_type'],'RuntimeError')

    def test_record_failure_cannot_block_authorized_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            suite,_=self.fixture(Path(temp))
            with patch.object(delivery,'bundle_results',return_value=package()),patch.object(delivery,'send_mail'), \
                    patch.object(delivery,'_record_delivery',side_effect=OSError('record failed')), \
                    patch.object(delivery,'stop_workbench') as stop:
                with self.assertRaises(OSError):delivery.finish(suite,suite/'run.log','recipient',0,'url')
            stop.assert_called_once()

    def test_child_environment_excludes_email_credentials(self):
        with patch.dict(os.environ,{'GMAIL_USER':'private','GMAIL_APP_PASSWORD':'private','KEEP':'yes'}):
            env=runner.child_environment()
        self.assertNotIn('GMAIL_USER',env);self.assertNotIn('GMAIL_APP_PASSWORD',env)
        self.assertEqual(env['KEEP'],'yes')


if __name__=='__main__':unittest.main()
