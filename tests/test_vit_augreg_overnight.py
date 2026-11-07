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

import torch

import run_vit_augreg_overnight as runner


TEST_ROOT = Path(__file__).resolve().parents[1] / 'runs' / 'vit_augreg_wrapper_tests_tmp'


def temporary_directory():
    TEST_ROOT.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEST_ROOT)


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('--source-suite')
    result.add_argument('--resume')
    result.add_argument('--vit-epochs', type=int)
    result.add_argument('--fusion-epochs', type=int)
    result.add_argument('--profiles', nargs='+')
    result.add_argument('--complete-all', action='store_true')
    result.add_argument('--smoke', action='store_true')
    result.add_argument('--to', default='recipient@example.com')
    result.add_argument('--check-only', action='store_true')
    result.add_argument('--fallback-os-shutdown', action='store_true')
    return result


class FakeChild:
    def __init__(self, output=None, code=0):
        self.stdout = output if output is not None else iter(['All follow-up studies complete\n'])
        self.code, self.terminated = code, False

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
    plan = dict(vit_epochs=300, fusion_epochs=30, complete_all=True, profiles=['tiny', 'small'])
    return SimpleNamespace(make_plan=MagicMock(return_value=plan),
        create_suite=MagicMock(return_value=path), validate_resume=MagicMock(return_value=plan))


def bundle_stub(attachment=Path('reports.zip')):
    return dict(full=Path('reports.zip'), compact=None, attachment=attachment,
        components={'training': dict(directory='training', exists=True),
                    'feature_analysis': dict(directory='feature_analysis', exists=True)},
        compact_level=0, omitted={})


def complete_suite(suite):
    for component in runner.COMPONENTS:
        (suite / component).mkdir(exist_ok=True)
    (suite / 'completion.json').write_text(json.dumps(dict(status='complete', outcomes={
        'training': dict(status='complete'), 'feature_analysis': dict(status='complete')})))


class ViTAugRegOvernightTests(unittest.TestCase):
    def test_atomic_replace_retries_transient_file_lock(self):
        source = MagicMock()
        source.replace.side_effect = [PermissionError('temporary lock'), None]
        with patch.object(runner.time, 'sleep'):
            runner._atomic_replace(source, Path('reports.zip'))
        self.assertEqual(source.replace.call_count, 2)

    def test_report_zip_is_single_layer_and_excludes_sources_credentials_and_weights(self):
        with temporary_directory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            complete_suite(suite)
            (suite / 'source_hashes.json').write_text('{"model.py":"sha256"}')
            (suite / 'overnight.log').write_text('done')
            for component in runner.COMPONENTS:
                folder = suite / component
                for name in ('report.md', 'summary.json', 'epochs.csv', 'per_class.csv',
                             'paired_summary.csv', 'confusion_matrix.csv', 'gallery.png', 'memo.pdf'):
                    (folder / name).write_text('analysis')
                for name in ('script.py', 'script.ps1', 'source.sh', 'best.pt', 'probabilities.npz',
                             'nested.zip', 'nested.tar.gz', 'email.txt', 'email.txt.bak', '.env',
                             'credentials.json', 'client_secret_test.json', 'service_account.json'):
                    (folder / name).write_text('private')
                for name in ('data', 'source', 'credentials'):
                    (folder / name).mkdir()
                    (folder / name / 'allowed.json').write_text('private')
            bundle = runner.bundle_results(suite)
            self.assertEqual(bundle['attachment'], bundle['full'])
            self.assertEqual(bundle['compact_level'], 0)
            with zipfile.ZipFile(bundle['full']) as archive:
                self.assertIsNone(archive.testzip())
                names = archive.namelist()
                self.assertEqual(len(names), len(set(names)))
                self.assertIn('training_analysis/report.md', names)
                self.assertIn('feature_analysis/report.md', names)
                self.assertIn('suite/source_hashes.json', names)
                self.assertIn('suite/overnight.log', names)
                self.assertFalse(any('private' in archive.read(name).decode(errors='ignore') for name in names))
                self.assertTrue(all(Path(name).suffix in runner.ALLOWED_SUFFIXES for name in names))
                self.assertFalse(any(name.endswith(('.py', '.ps1', '.sh', '.pt', '.npz', '.gz', '.zip')) for name in names))
                manifest = json.loads(archive.read('delivery_manifest.json'))
                self.assertTrue(manifest['components']['training']['exists'])
                self.assertIn('executable source', manifest['excluded_types'])

    def test_compaction_preserves_aggregate_paired_class_confusion_and_epoch_data(self):
        with temporary_directory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            complete_suite(suite)
            folder = suite / 'feature_analysis'
            for name in ('report.md', 'paired_summary.json', 'paired_summary.csv', 'per_class.csv',
                         'confusion_matrix.csv', 'epochs.csv'):
                (folder / name).write_text('aggregate')
            for name in ('predictions.csv', 'paired_selection.csv', 'image_inventory.csv', 'gallery.png'):
                (folder / name).write_bytes(os.urandom(12000))
            with patch.object(runner, 'ATTACHMENT_LIMIT', 8000):
                bundle = runner.bundle_results(suite)
            self.assertTrue(bundle['full'].exists())
            self.assertEqual(bundle['compact_level'], 2)
            self.assertEqual(bundle['attachment'], bundle['compact'])
            self.assertEqual(set(bundle['omitted']['feature_analysis']), {
                'predictions.csv', 'paired_selection.csv', 'image_inventory.csv', 'gallery.png'})
            with zipfile.ZipFile(bundle['attachment']) as archive:
                names = archive.namelist()
                for name in ('paired_summary.json', 'paired_summary.csv', 'per_class.csv',
                             'confusion_matrix.csv', 'epochs.csv'):
                    self.assertIn(f'feature_analysis/{name}', names)
            with patch.object(runner, 'ATTACHMENT_LIMIT', 1):
                self.assertIsNone(runner.bundle_results(suite)['attachment'])

    def test_per_image_header_detection_does_not_discard_paired_aggregate(self):
        with temporary_directory() as temp:
            folder = Path(temp)
            (folder / 'decisions.csv').write_text('sample_id,correct\na,1\n')
            (folder / 'paired_summary.csv').write_text('model,corrected,harmed\na,5,2\n')
            files, omitted = runner._analysis_files(folder, compact=1)
            self.assertEqual([relative.as_posix() for _, relative in files], ['paired_summary.csv'])
            self.assertEqual(omitted, ['decisions.csv'])

    def test_probability_and_prediction_aggregates_survive_email_compaction(self):
        with temporary_directory() as temp:
            folder = Path(temp)
            (folder / 'prediction_mixture_summary.csv').write_text('vit_weight,top1,nll\n0.2,.6472,1.70\n')
            (folder / 'probability_mixtures.csv').write_text('model,accuracy\nmixture,.6472\n')
            (folder / 'threeway_summary.csv').write_text('cnn_correct,vit_correct,joint_correct,count\n1,1,1,4040\n')
            (folder / 'prediction_membership.csv').write_text('"image_id",correct\na,1\n')
            files, omitted = runner._analysis_files(folder, compact=1)
            self.assertEqual({relative.as_posix() for _, relative in files}, {
                'prediction_mixture_summary.csv', 'probability_mixtures.csv', 'threeway_summary.csv'})
            self.assertEqual(omitted, ['prediction_membership.csv'])

    def test_missing_components_are_marked_in_manifest_and_status(self):
        with temporary_directory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            (suite / 'completion.json').write_text('{"status":"complete"}')
            bundle = runner.bundle_results(suite)
            with zipfile.ZipFile(bundle['full']) as archive:
                manifest = json.loads(archive.read('feature_analysis/analysis_bundle.json'))
            self.assertFalse(manifest['component_directory_exists'])
            self.assertEqual(manifest['report_file_count'], 0)
            self.assertEqual(runner.completion_status(suite, 0), 'PARTIAL / MISSING ANALYSIS')

    def test_symlink_components_are_rejected_and_links_to_files_are_excluded(self):
        with temporary_directory() as temp:
            folder = Path(temp)
            suite = folder / 'suite'
            suite.mkdir()
            outside = folder / 'outside'
            outside.mkdir()
            (outside / 'private.json').write_text('private')
            try:
                (suite / 'feature_analysis').symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest('Symlink privilege unavailable')
            with self.assertRaisesRegex(ValueError, 'must not be a symlink'):
                runner.bundle_results(suite)
            (suite / 'feature_analysis').unlink()
            (suite / 'feature_analysis').mkdir()
            (suite / 'feature_analysis' / 'linked.json').symlink_to(outside / 'private.json')
            files, _ = runner._analysis_files(suite / 'feature_analysis')
            self.assertEqual(files, [])

    def test_email_preflight_authenticates_without_transmitting(self):
        smtp = MagicMock()
        with patch.dict(os.environ, {'GMAIL_USER': 'sender@example.com', 'GMAIL_APP_PASSWORD': 'app pass'}), \
                patch.object(runner, 'load_email_credentials'), \
                patch.object(runner.smtplib, 'SMTP_SSL') as factory:
            factory.return_value.__enter__.return_value = smtp
            runner.check_email_ready('recipient@example.com')
        smtp.login.assert_called_once_with('sender@example.com', 'apppass')
        smtp.send_message.assert_not_called()

    def test_preflight_authentication_error_does_not_reveal_credentials(self):
        with patch.dict(os.environ, {'GMAIL_USER': 'sender', 'GMAIL_APP_PASSWORD': 'hidden'}), \
                patch.object(runner, 'load_email_credentials'), \
                patch.object(runner.smtplib, 'SMTP_SSL', side_effect=RuntimeError('hidden private password')):
            with self.assertRaisesRegex(RuntimeError, 'Email authentication failed') as result:
                runner.check_email_ready('recipient@example.com')
        self.assertNotIn('hidden', str(result.exception))

    def test_completion_mail_uses_actual_zip_mime_and_no_start_notification(self):
        with temporary_directory() as temp:
            attachment = Path(temp) / 'reports.zip'
            with zipfile.ZipFile(attachment, 'w') as archive:
                archive.writestr('report.json', '{"status":"complete"}')
            smtp = MagicMock()
            with patch.dict(os.environ, {'GMAIL_USER': 'sender@example.com', 'GMAIL_APP_PASSWORD': 'hidden'}), \
                    patch.object(runner, 'load_email_credentials'), \
                    patch.object(runner.smtplib, 'SMTP_SSL') as factory:
                factory.return_value.__enter__.return_value = smtp
                runner.send_mail('recipient@example.com', 'Completed', 'Reports', attachment)
            message = smtp.send_message.call_args.args[0]
            parts = list(message.iter_attachments())
            self.assertEqual(len(parts), 1)
            self.assertEqual(parts[0].get_content_type(), 'application/zip')
            self.assertEqual(parts[0].get_filename(), 'reports.zip')
            self.assertEqual(parts[0].get_payload(decode=True), attachment.read_bytes())
            smtp.send_message.assert_called_once()

    def test_no_success_claim_without_explicit_complete_status(self):
        with temporary_directory() as temp:
            suite = Path(temp)
            self.assertEqual(runner.completion_status(suite, 0), 'PARTIAL / STATUS UNAVAILABLE')
            complete_suite(suite)
            self.assertEqual(runner.completion_status(suite, 0), 'SUCCESS')
            (suite / 'completion.json').write_text('{"status":"complete","outcomes":{"training":{"status":"failed"}}}')
            self.assertEqual(runner.completion_status(suite, 0), 'PARTIAL / INCOMPLETE')
            self.assertEqual(runner.completion_status(suite, 1), 'FAILED / INTERRUPTED')

    @patch.object(runner, '_record_delivery')
    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail')
    @patch.object(runner, 'bundle_results', return_value=bundle_stub())
    @patch.object(runner, 'completion_status', return_value='SUCCESS')
    def test_one_completion_email_precedes_stop(self, status, bundle, mail, stop, record):
        events = []
        mail.side_effect = lambda *args, **kwargs: events.append('mail')
        stop.side_effect = lambda *args: events.append('stop')
        self.assertTrue(runner.finish('suite', 'log', 'url', True, 'recipient', 0))
        self.assertEqual(events, ['mail', 'stop'])
        mail.assert_called_once()
        self.assertEqual(mail.call_args.kwargs['attachment'], Path('reports.zip'))
        self.assertIn('single-layer ZIP', mail.call_args.args[2])
        self.assertTrue(record.call_args.args[1]['email_accepted'])

    @patch.object(runner, '_record_delivery')
    @patch.object(runner.time, 'sleep')
    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail', side_effect=RuntimeError('unavailable'))
    @patch.object(runner, 'bundle_results', side_effect=OSError('disk full'))
    def test_package_and_mail_failure_still_stop_after_three_send_attempts(self, bundle, mail, stop, sleep, record):
        self.assertFalse(runner.finish('suite', 'log', 'url', False, 'recipient', 1))
        self.assertEqual(mail.call_count, 3)
        self.assertIn('ANALYSIS PACKAGE FAILED', mail.call_args.args[1])
        self.assertIsNone(mail.call_args.kwargs['attachment'])
        stop.assert_called_once_with('url', False)

    @patch.object(runner, '_record_delivery')
    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail')
    @patch.object(runner, 'bundle_results', return_value=bundle_stub(attachment=None))
    def test_oversize_notification_keeps_local_zip_path_and_still_stops(self, bundle, mail, stop, record):
        self.assertFalse(runner.finish('suite', 'log', 'url', False, 'recipient', 0))
        self.assertIn('ATTACHMENT TOO LARGE', mail.call_args.args[1])
        self.assertIn('reports.zip', mail.call_args.args[2])
        self.assertIsNone(mail.call_args.kwargs['attachment'])
        stop.assert_called_once()

    def test_preflight_failure_does_not_start_child_send_or_stop(self):
        fake = fake_suite_module()
        with patch.dict(sys.modules, {'vit_augreg_suite': fake}), \
                patch.object(runner, 'arguments', side_effect=parser), \
                patch('torch.cuda.is_available', return_value=True), \
                patch.object(runner, 'check_email_ready'), \
                patch.object(runner, 'preflight', side_effect=RuntimeError('permission denied')), \
                patch.object(runner, 'send_mail') as mail, \
                patch.object(runner, 'stop_workbench') as stop, \
                patch.object(runner.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'permission'):
                runner.main([])
        fake.create_suite.assert_not_called()
        process.assert_not_called()
        mail.assert_not_called()
        stop.assert_not_called()

    def test_check_only_makes_no_suite_email_training_or_stop(self):
        fake = fake_suite_module()
        with patch.dict(sys.modules, {'vit_augreg_suite': fake}), \
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

    def test_full_wrapper_logs_child_strips_credentials_and_finishes_once(self):
        with temporary_directory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            complete_suite(suite)
            fake = fake_suite_module(suite)
            child = FakeChild()
            events = []
            with patch.dict(sys.modules, {'vit_augreg_suite': fake}), \
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
            self.assertIn('All follow-up studies complete', (suite / 'overnight.log').read_text())
            self.assertTrue(json.loads((suite / 'delivery_status.json').read_text())['email_accepted'])

    @patch.object(runner, '_record_delivery')
    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail')
    @patch.object(runner, 'bundle_results', side_effect=OSError('disk full'))
    @patch.object(runner, 'completion_status', return_value='SUCCESS')
    def test_accepted_email_without_package_is_not_complete_delivery(self, status, bundle, mail, stop, record):
        self.assertFalse(runner.finish('suite', 'log', 'url', False, 'recipient', 0))
        value = record.call_args.args[1]
        self.assertTrue(value['email_accepted'])
        self.assertFalse(value['analysis_package_succeeded'])
        self.assertFalse(value['analysis_attachment_available'])
        self.assertFalse(value['delivery_complete'])
        stop.assert_called_once()

    def test_child_success_delivery_failure_exits_two_but_keeps_stop_path(self):
        with temporary_directory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            fake = fake_suite_module(suite)
            with patch.dict(sys.modules, {'vit_augreg_suite': fake}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.object(runner, 'check_email_ready'), \
                    patch.object(runner, 'preflight', return_value='url'), \
                    patch.object(runner.subprocess, 'Popen', return_value=FakeChild()), \
                    patch.object(runner, 'finish', return_value=False) as finish:
                with self.assertRaises(SystemExit) as result:
                    runner.main([])
            self.assertEqual(result.exception.code, 2)
            finish.assert_called_once_with(suite, suite / 'overnight.log', 'url', False,
                                           'recipient@example.com', 0)

    def test_child_failure_keeps_original_exit_code_when_delivery_also_fails(self):
        with temporary_directory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            fake = fake_suite_module(suite)
            with patch.dict(sys.modules, {'vit_augreg_suite': fake}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.object(runner, 'check_email_ready'), \
                    patch.object(runner, 'preflight', return_value='url'), \
                    patch.object(runner.subprocess, 'Popen', return_value=FakeChild(code=7)), \
                    patch.object(runner, 'finish', return_value=False) as finish:
                with self.assertRaises(SystemExit) as result:
                    runner.main([])
            self.assertEqual(result.exception.code, 7)
            self.assertEqual(finish.call_args.args[-1], 7)

    def test_spawn_failure_never_arms_finish(self):
        with temporary_directory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            fake = fake_suite_module(suite)
            with patch.dict(sys.modules, {'vit_augreg_suite': fake}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.object(runner, 'check_email_ready'), \
                    patch.object(runner, 'preflight', return_value='url'), \
                    patch.object(runner.subprocess, 'Popen', side_effect=OSError('cannot start')), \
                    patch.object(runner, 'finish') as finish:
                with self.assertRaisesRegex(OSError, 'cannot start'):
                    runner.main([])
            finish.assert_not_called()

    def test_interrupt_stops_child_and_calls_failed_finish(self):
        def interrupted():
            yield 'Epoch in progress\n'
            raise KeyboardInterrupt()
        with temporary_directory() as temp:
            suite = Path(temp) / 'suite'
            suite.mkdir()
            fake = fake_suite_module(suite)
            child = FakeChild(output=interrupted())
            with patch.dict(sys.modules, {'vit_augreg_suite': fake}), \
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

    def test_resume_cannot_change_horizon_or_smoke_before_preflight(self):
        fake = fake_suite_module()
        with patch.dict(sys.modules, {'vit_augreg_suite': fake}), \
                patch.object(runner, 'arguments', side_effect=parser), \
                patch('torch.cuda.is_available', return_value=True), \
                patch.object(runner, 'check_email_ready') as email, \
                patch.object(runner, 'preflight') as preflight:
            for extra in (['--vit-epochs', '50'], ['--fusion-epochs', '20'], ['--smoke']):
                with self.assertRaisesRegex(ValueError, 'saved smoke/full'):
                    runner.main(['--resume', 'suite', *extra])
        email.assert_not_called()
        preflight.assert_not_called()

    def test_compatible_explicit_resume_horizons_pass_check_only(self):
        fake = fake_suite_module()
        with patch.dict(sys.modules, {'vit_augreg_suite': fake}), \
                patch.object(runner, 'arguments', side_effect=parser), \
                patch('torch.cuda.is_available', return_value=True), \
                patch.object(runner, 'check_email_ready') as ready, \
                patch.object(runner, 'preflight', return_value='url'), \
                patch.object(runner, 'send_mail') as mail, \
                patch.object(runner, 'stop_workbench') as stop:
            runner.main(['--resume', 'suite', '--vit-epochs', '300', '--fusion-epochs', '30', '--check-only'])
        ready.assert_called_once()
        mail.assert_not_called()
        stop.assert_not_called()

    def test_resume_cannot_silently_change_profile_search_before_preflight(self):
        fake = fake_suite_module()
        with patch.dict(sys.modules, {'vit_augreg_suite': fake}), \
                patch.object(runner, 'arguments', side_effect=parser), \
                patch('torch.cuda.is_available', return_value=True), \
                patch.object(runner, 'check_email_ready') as email, \
                patch.object(runner, 'preflight') as preflight:
            with self.assertRaisesRegex(ValueError, 'saved profile search'):
                runner.main(['--resume', 'suite', '--profiles', 'small'])
        email.assert_not_called()
        preflight.assert_not_called()


if __name__ == '__main__':
    unittest.main()
