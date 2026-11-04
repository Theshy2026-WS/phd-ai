import argparse
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import torch

import run_generalization_overnight as runner


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('--parent-run')
    result.add_argument('--resume')
    result.add_argument('--fresh-epochs', type=int)
    result.add_argument('--warm-epochs', type=int)
    result.add_argument('--max-hours', type=float)
    result.add_argument('--smoke', action='store_true')
    result.add_argument('--to', default='recipient@example.com')
    result.add_argument('--check-only', action='store_true')
    result.add_argument('--fallback-os-shutdown', action='store_true')
    return result


class FakeChild:
    def __init__(self, output=None, code=0):
        self.stdout = output if output is not None else iter(['Both studies complete\n'])
        self.code = code
        self.terminated = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def wait(self, timeout=None):
        return self.code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def fake_suite_module(path=None):
    return SimpleNamespace(make_plan=MagicMock(return_value={'fresh_epochs': 100, 'warm_epochs': 30, 'max_hours': 12}),
        create_suite=MagicMock(return_value=path),
        validate_resume=MagicMock(return_value={'fresh_epochs': 100, 'warm_epochs': 30, 'max_hours': 12}))


def bundle_stub(attachment=Path('delivery.tar.gz')):
    return dict(full=Path('delivery.tar.gz'), compact=None, attachment=attachment,
        components={'data_audit': Path('data_audit_analysis.tar.gz'),
                    'previous_last': Path('previous_last_analysis.tar.gz'),
                    'training': Path('training_analysis.tar.gz')}, compact_level=0,
        omitted={})


class GeneralizationOvernightTests(unittest.TestCase):
    def test_budget_limit_is_reported_as_partial_not_complete_or_failed(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)
            (suite/'status.json').write_text(json.dumps(dict(status='complete', outcomes={
                'training': dict(status='budget_exhausted', deferred_jobs=['sam'])})))
            self.assertEqual(runner.completion_status(suite, 0), 'BUDGET LIMIT / PARTIAL')
            self.assertEqual(runner.completion_status(suite, 1), 'FAILED / INTERRUPTED')
            (suite/'status.json').write_text(json.dumps(dict(status='complete', max_hours=12)))
            self.assertEqual(runner.completion_status(suite, 0), 'SUCCESS')

    @patch.object(runner, '_record_delivery')
    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail')
    @patch.object(runner, 'bundle_results', return_value=bundle_stub())
    def test_budget_completion_sends_one_email_and_stops(self, bundle, mail, stop, record):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)
            (suite/'completion.json').write_text(json.dumps(dict(status='budget_limited')))
            self.assertTrue(runner.finish(suite, 'log', 'url', False, 'recipient', 0))
        mail.assert_called_once()
        self.assertIn('BUDGET LIMIT / PARTIAL', mail.call_args.args[1])
        self.assertEqual(record.call_args.args[1]['experiment_status'], 'BUDGET LIMIT / PARTIAL')
        stop.assert_called_once_with('url', False)

    def test_pdf_memo_is_packaged_and_inventory_is_compacted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'research.pdf').write_bytes(b'PDF memo')
            (folder/'image_inventory.csv').write_text('id,sha256\nimage,abc')
            files, omitted = runner._analysis_files(folder, compact=1)
            self.assertEqual([relative.as_posix() for _, relative in files], ['research.pdf'])
            self.assertEqual(omitted, ['image_inventory.csv'])

    def test_email_envelope_contains_three_reports_and_excludes_private_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            (suite / 'suite.json').write_text('{"epochs":30}')
            (suite / 'overnight.log').write_text('complete')
            for component in runner.COMPONENTS:
                folder = suite / component
                folder.mkdir()
                for name in ('report.md', 'summary.json', 'epochs.csv', 'per_class.csv', 'gallery.png'):
                    (folder / name).write_text('analysis')
                for name in ('best.pt', 'probabilities.npz', 'nested.tar.gz', 'email.txt', '.env', 'email.txt.bak'):
                    (folder / name).write_text('secret')
                (folder / 'data').mkdir()
                (folder / 'data' / 'looks_allowed.json').write_text('secret')
            with patch.object(runner, 'SOURCE_FILES', ()):
                bundle = runner.bundle_results(suite)
            self.assertEqual(bundle['attachment'], bundle['full'])
            self.assertEqual(bundle['compact_level'], 0)
            with tarfile.open(bundle['full']) as archive:
                self.assertIn('data_audit_analysis.tar.gz', archive.getnames())
                self.assertIn('training_analysis.tar.gz', archive.getnames())
                self.assertIn('previous_last_analysis.tar.gz', archive.getnames())
                self.assertIn('suite/overnight.log', archive.getnames())
                for component in runner.COMPONENTS:
                    with tarfile.open(fileobj=io.BytesIO(archive.extractfile(
                            f'{component}_analysis.tar.gz').read())) as material:
                        names = material.getnames()
                    self.assertIn(f'{component}_analysis/report.md', names)
                    self.assertIn(f'{component}_analysis/per_class.csv', names)
                    self.assertEqual(len(names), 6)
                    self.assertFalse(any('email.txt' in name or name.endswith(('.pt', '.npz', '.gz'))
                                         or '/data/' in name for name in names))

    def test_two_compact_levels_keep_curves_and_aggregate_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / 'suite'
            (suite / 'data_audit').mkdir(parents=True)
            (suite / 'training').mkdir()
            folder = suite / 'data_audit'
            (folder / 'report.md').write_text('report')
            (folder / 'paired_summary.json').write_text('{"corrected":5}')
            (folder / 'per_class.csv').write_text('label,n\n0,50')
            (folder / 'epochs.csv').write_text('epoch,accuracy\n1,0.64')
            (folder / 'predictions.csv').write_bytes(os.urandom(20000))
            (folder / 'paired_selection.csv').write_bytes(os.urandom(10000))
            (folder / 'image_inventory.csv').write_bytes(os.urandom(10000))
            (folder / 'gallery.png').write_bytes(os.urandom(12000))
            with patch.object(runner, 'SOURCE_FILES', ()), patch.object(runner, 'ATTACHMENT_LIMIT', 4000):
                bundle = runner.bundle_results(suite)
            self.assertEqual(bundle['compact_level'], 2)
            self.assertTrue(bundle['full'].exists())
            self.assertEqual(bundle['attachment'], bundle['compact'])
            self.assertEqual(set(bundle['omitted']['data_audit']),
                             {'predictions.csv', 'paired_selection.csv', 'gallery.png', 'image_inventory.csv'})
            with tarfile.open(bundle['attachment']) as archive:
                with tarfile.open(fileobj=io.BytesIO(archive.extractfile('data_audit_analysis.tar.gz').read())) as material:
                    names = material.getnames()
            for name in ('report.md', 'paired_summary.json', 'epochs.csv', 'per_class.csv'):
                self.assertIn(f'data_audit_analysis/{name}', names)
            with patch.object(runner, 'SOURCE_FILES', ()), patch.object(runner, 'ATTACHMENT_LIMIT', 1):
                self.assertIsNone(runner.bundle_results(suite)['attachment'])

    def test_missing_component_still_has_explicit_partial_material(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            with patch.object(runner, 'SOURCE_FILES', ()):
                bundle = runner.bundle_results(suite)
            with tarfile.open(bundle['components']['data_audit']) as material:
                manifest = json.load(material.extractfile('data_audit_analysis/analysis_bundle.json'))
            self.assertFalse(manifest['component_directory_exists'])

    def test_symlink_components_and_ancestors_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            component = suite / 'data_audit'
            component.mkdir()
            original = Path.is_symlink
            with patch.object(Path, 'is_symlink', lambda p: p == component or original(p)):
                with self.assertRaisesRegex(ValueError, 'symlink'):
                    runner.bundle_results(suite)
            with patch.object(Path, 'is_symlink', lambda p: p == suite or original(p)):
                with self.assertRaisesRegex(ValueError, 'symlink'):
                    runner.bundle_results(suite)

    def test_email_preflight_authenticates_without_sending(self):
        smtp = MagicMock()
        with patch.object(runner, 'load_email_credentials'), \
                patch.dict(os.environ, {'GMAIL_USER': 'sender', 'GMAIL_APP_PASSWORD': 'app pass'}), \
                patch.object(runner.smtplib, 'SMTP_SSL', return_value=smtp), \
                patch.object(runner, 'send_mail') as send:
            runner.check_email_ready('recipient@example.com')
        smtp.__enter__.return_value.login.assert_called_once_with('sender', 'apppass')
        smtp.__enter__.return_value.send_message.assert_not_called()
        send.assert_not_called()

    def test_email_preflight_failure_does_not_expose_exception_details(self):
        with patch.object(runner, 'load_email_credentials'), \
                patch.dict(os.environ, {'GMAIL_USER': 'sender', 'GMAIL_APP_PASSWORD': 'hidden'}), \
                patch.object(runner.smtplib, 'SMTP_SSL', side_effect=RuntimeError('hidden private token')):
            with self.assertRaisesRegex(RuntimeError, 'Email authentication failed') as result:
                runner.check_email_ready('recipient@example.com')
        self.assertNotIn('hidden', str(result.exception))

    @patch.object(runner, '_record_delivery')
    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail')
    @patch.object(runner, 'bundle_results', return_value=bundle_stub())
    def test_one_completion_email_precedes_stop(self, bundle, mail, stop, record):
        events = []
        mail.side_effect = lambda *args, **kwargs: events.append('mail')
        stop.side_effect = lambda *args: events.append('stop')
        self.assertTrue(runner.finish('suite', 'log', 'url', True, 'recipient', 0))
        self.assertEqual(events, ['mail', 'stop'])
        mail.assert_called_once()
        self.assertEqual(mail.call_args.kwargs['attachment'], Path('delivery.tar.gz'))
        self.assertIn('data_audit_analysis.tar.gz', mail.call_args.args[2])
        self.assertIn('training_analysis.tar.gz', mail.call_args.args[2])
        self.assertIn('previous_last_analysis.tar.gz', mail.call_args.args[2])
        self.assertTrue(record.call_args.args[1]['email_accepted'])

    @patch.object(runner, '_record_delivery')
    @patch.object(runner.time, 'sleep')
    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail', side_effect=RuntimeError('unavailable'))
    @patch.object(runner, 'bundle_results', side_effect=OSError('disk full'))
    def test_package_and_email_failure_still_stop(self, bundle, mail, stop, sleep, record):
        self.assertFalse(runner.finish('suite', 'log', 'url', False, 'recipient', 1))
        self.assertEqual(mail.call_count, 3)
        self.assertIn('ANALYSIS PACKAGE FAILED', mail.call_args.args[1])
        self.assertIsNone(mail.call_args.kwargs['attachment'])
        stop.assert_called_once_with('url', False)

    @patch.object(runner, '_record_delivery')
    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail')
    @patch.object(runner, 'bundle_results', return_value=bundle_stub(attachment=None))
    def test_oversize_email_reports_disk_paths(self, bundle, mail, stop, record):
        runner.finish('suite', 'log', 'url', False, 'recipient', 0)
        self.assertIn('ATTACHMENT TOO LARGE', mail.call_args.args[1])
        self.assertIn('delivery.tar.gz', mail.call_args.args[2])
        self.assertIsNone(mail.call_args.kwargs['attachment'])
        stop.assert_called_once()

    def test_preflight_failure_never_creates_child_or_stops(self):
        fake = fake_suite_module()
        with patch.dict(sys.modules, {'generalization_suite': fake}), \
                patch.object(runner, 'arguments', side_effect=parser), \
                patch('torch.cuda.is_available', return_value=True), \
                patch.object(runner, 'check_email_ready'), \
                patch.object(runner, 'preflight', side_effect=RuntimeError('no permission')), \
                patch.object(runner, 'send_mail') as mail, \
                patch.object(runner, 'stop_workbench') as stop, \
                patch.object(runner.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'permission'):
                runner.main([])
        fake.create_suite.assert_not_called()
        process.assert_not_called()
        mail.assert_not_called()
        stop.assert_not_called()

    def test_check_only_sends_no_email_and_creates_no_suite(self):
        fake = fake_suite_module()
        with patch.dict(sys.modules, {'generalization_suite': fake}), \
                patch.object(runner, 'arguments', side_effect=parser), \
                patch('torch.cuda.is_available', return_value=True), \
                patch.object(runner, 'check_email_ready') as ready, \
                patch.object(runner, 'preflight', return_value='url'), \
                patch.object(runner, 'send_mail') as mail, \
                patch.object(runner, 'stop_workbench') as stop, \
                patch.object(runner.subprocess, 'Popen') as process:
            runner.main(['--check-only', '--resume', 'suite'])
        ready.assert_called_once()
        fake.validate_resume.assert_called_once_with('suite')
        fake.create_suite.assert_not_called()
        process.assert_not_called()
        mail.assert_not_called()
        stop.assert_not_called()

    def test_full_wrapper_sends_only_at_completion_and_logs_child(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            fake = fake_suite_module(suite)
            child = FakeChild()
            events = []
            with patch.dict(sys.modules, {'generalization_suite': fake}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.dict(os.environ, {'GMAIL_USER': 'sender', 'GMAIL_APP_PASSWORD': 'hidden'}), \
                    patch.object(runner, 'check_email_ready'), \
                    patch.object(runner, 'preflight', return_value='url'), \
                    patch.object(runner, 'bundle_results', return_value=bundle_stub()), \
                    patch.object(runner, 'send_mail', side_effect=lambda *a, **k: events.append('mail')) as mail, \
                    patch.object(runner, 'stop_workbench', side_effect=lambda *a: events.append('stop')), \
                    patch.object(runner.subprocess, 'Popen', return_value=child) as process:
                with self.assertRaises(SystemExit) as result:
                    runner.main([])
                self.assertEqual(result.exception.code, 0)
            self.assertEqual(events, ['mail', 'stop'])
            mail.assert_called_once()
            self.assertNotIn('GMAIL_USER', process.call_args.kwargs['env'])
            self.assertNotIn('GMAIL_APP_PASSWORD', process.call_args.kwargs['env'])
            self.assertEqual(process.call_args.args[0][-2:], ['--resume', str(suite)])
            self.assertIn('Both studies complete', (suite / 'overnight.log').read_text())
            self.assertTrue(json.loads((suite / 'delivery_status.json').read_text())['email_accepted'])

    def test_spawn_failure_never_arms_finish(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            fake = fake_suite_module(suite)
            with patch.dict(sys.modules, {'generalization_suite': fake}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.object(runner, 'check_email_ready'), \
                    patch.object(runner, 'preflight', return_value='url'), \
                    patch.object(runner.subprocess, 'Popen', side_effect=OSError('cannot start')), \
                    patch.object(runner, 'finish') as finish:
                with self.assertRaisesRegex(OSError, 'cannot start'):
                    runner.main([])
            finish.assert_not_called()

    def test_interrupt_terminates_child_and_triggers_failed_finish(self):
        def interrupted():
            yield 'Epoch in progress\n'
            raise KeyboardInterrupt()
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            fake = fake_suite_module(suite)
            child = FakeChild(output=interrupted())
            with patch.dict(sys.modules, {'generalization_suite': fake}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.object(runner, 'check_email_ready'), \
                    patch.object(runner, 'preflight', return_value='url'), \
                    patch.object(runner.subprocess, 'Popen', return_value=child), \
                    patch.object(runner, 'finish') as finish:
                with self.assertRaises(KeyboardInterrupt):
                    runner.main([])
            self.assertTrue(child.terminated)
            self.assertEqual(finish.call_args.args[-1], 1)
            self.assertIn('KeyboardInterrupt', (suite / 'overnight.log').read_text())

    def test_resume_horizon_change_is_rejected_before_remote_preflight(self):
        fake = fake_suite_module()
        with patch.dict(sys.modules, {'generalization_suite': fake}), \
                patch.object(runner, 'arguments', side_effect=parser), \
                patch('torch.cuda.is_available', return_value=True), \
                patch.object(runner, 'check_email_ready') as email, \
                patch.object(runner, 'preflight') as preflight:
            for extra in (['--fresh-epochs', '200'], ['--warm-epochs', '99'], ['--max-hours','6'], ['--smoke']):
                with self.assertRaisesRegex(ValueError, 'saved smoke/full'):
                    runner.main(['--resume', 'suite', *extra])
        email.assert_not_called()
        preflight.assert_not_called()


if __name__ == '__main__':
    unittest.main()
