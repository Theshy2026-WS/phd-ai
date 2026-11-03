import argparse
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

import run_spatial_overnight as runner


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('--base-run')
    result.add_argument('--resume')
    result.add_argument('--epochs', type=int)
    result.add_argument('--smoke', action='store_true')
    result.add_argument('--to', default='recipient@example.com')
    result.add_argument('--check-only', action='store_true')
    result.add_argument('--fallback-os-shutdown', action='store_true')
    return result


class FakeChild:
    def __init__(self, output=None, code=0):
        self.stdout = output if output is not None else iter(['Training complete\n'])
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


class SpatialOvernightTests(unittest.TestCase):
    def test_archive_keeps_analysis_and_excludes_private_or_large_types(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)/'suite'
            suite.mkdir()
            for name in ('summary.json', 'epochs.csv', 'paired_summary.json', 'gallery.png',
                         'report.md', 'overnight.log'):
                (suite/name).write_bytes(b'analysis')
            for name in ('best.pt', 'last.pth', 'predictions.npz', 'nested.tar.gz',
                         'email.txt', 'email.txt.bak', '.env'):
                (suite/name).write_bytes(b'private')
            for directory in ('datasets', 'data', 'checkpoints', 'email.txt.private'):
                (suite/directory).mkdir()
                (suite/directory/'looks_allowed.json').write_bytes(b'private')
            with patch.object(runner, 'SOURCE_FILES', ()):
                bundle = runner.bundle_results(suite)
            with tarfile.open(bundle['full']) as archive:
                names = archive.getnames()
            self.assertIn('suite/gallery.png', names)
            self.assertIn('suite/epochs.csv', names)
            self.assertIn('suite/analysis_bundle.json', names)
            self.assertEqual(len(names), 7)
            self.assertFalse(any('private' in name or 'predictions.npz' in name for name in names))

    def test_compact_attachment_and_too_large_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)/'suite'
            suite.mkdir()
            (suite/'summary.json').write_text('{"complete": true}')
            (suite/'gallery.png').write_bytes(b'small picture')
            (suite/'predictions.csv').write_bytes(os.urandom(30000))
            (suite/'paired_selection_seed42.csv').write_bytes(os.urandom(10000))
            with patch.object(runner, 'SOURCE_FILES', ()), patch.object(runner, 'ATTACHMENT_LIMIT', 5000):
                bundle = runner.bundle_results(suite)
            self.assertTrue(bundle['full'].exists())
            self.assertEqual(bundle['attachment'], bundle['compact'])
            with tarfile.open(bundle['compact']) as archive:
                names = archive.getnames()
                manifest = json.load(archive.extractfile('suite/analysis_bundle.json'))
            self.assertIn('suite/gallery.png', names)
            self.assertNotIn('suite/predictions.csv', names)
            self.assertNotIn('suite/paired_selection_seed42.csv', names)
            self.assertEqual(set(manifest['omitted_analysis_files']),
                             {'predictions.csv', 'paired_selection_seed42.csv'})
            with patch.object(runner, 'SOURCE_FILES', ()), patch.object(runner, 'ATTACHMENT_LIMIT', 1):
                self.assertIsNone(runner.bundle_results(suite)['attachment'])

    def test_symlink_file_and_ancestor_are_not_archived(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            suite = root/'suite'
            suite.mkdir()
            target = root/'private.json'
            target.write_text('secret')
            try:
                (suite/'linked.json').symlink_to(target)
                (root/'linked_suite').symlink_to(suite, target_is_directory=True)
            except OSError:
                # Windows may forbid local symlinks; exercise the same guard
                # without requesting privileges or touching the real filesystem.
                (suite/'mock_link.json').write_text('private')
                (root/'mock_linked_suite').mkdir()
                original = Path.is_symlink
                links = {suite/'mock_link.json', root/'mock_linked_suite'}
                with patch.object(Path, 'is_symlink', lambda p: p in links or original(p)), \
                        patch.object(runner, 'SOURCE_FILES', ()):
                    bundle = runner.bundle_results(suite)
                    with tarfile.open(bundle['full']) as archive:
                        self.assertNotIn('suite/mock_link.json', archive.getnames())
                    with self.assertRaisesRegex(ValueError, 'symlink'):
                        runner.bundle_results(root/'mock_linked_suite')
                return
            with patch.object(runner, 'SOURCE_FILES', ()):
                bundle = runner.bundle_results(suite)
            with tarfile.open(bundle['full']) as archive:
                self.assertNotIn('suite/linked.json', archive.getnames())
            with self.assertRaisesRegex(ValueError, 'symlink'):
                runner.bundle_results(root/'linked_suite')

    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail')
    @patch.object(runner, 'bundle_results')
    def test_email_precedes_stop(self, bundle, mail, stop):
        bundle.return_value = dict(full=Path('full.tar.gz'), compact=None,
                                   attachment=Path('full.tar.gz'), omitted=[])
        events = []
        mail.side_effect = lambda *a, **k: events.append('mail')
        stop.side_effect = lambda *a: events.append('stop')
        self.assertTrue(runner.finish('suite', 'log', 'url', True, 'recipient', 0))
        self.assertEqual(events, ['mail', 'stop'])
        self.assertEqual(mail.call_args.kwargs['attachment'], Path('full.tar.gz'))

    @patch.object(runner.time, 'sleep')
    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail', side_effect=RuntimeError('SMTP unavailable'))
    @patch.object(runner, 'bundle_results', side_effect=OSError('disk full'))
    def test_packaging_and_email_failure_still_stop(self, bundle, mail, stop, sleep):
        self.assertFalse(runner.finish('suite', 'log', 'url', False, 'recipient', 1))
        self.assertEqual(mail.call_count, 3)
        self.assertTrue(all(call.kwargs['attachment'] is None for call in mail.call_args_list))
        self.assertIn('ANALYSIS PACKAGE FAILED', mail.call_args.args[1])
        stop.assert_called_once_with('url', False)

    @patch.object(runner, 'stop_workbench')
    @patch.object(runner, 'send_mail')
    @patch.object(runner, 'bundle_results')
    def test_oversized_archive_sends_location_without_attachment(self, bundle, mail, stop):
        bundle.return_value = dict(full=Path('full.tar.gz'), compact=Path('compact.tar.gz'),
                                   attachment=None, omitted=['predictions.csv'])
        runner.finish('suite', 'log', 'url', True, 'recipient', 0)
        self.assertIsNone(mail.call_args.kwargs['attachment'])
        self.assertIn('ATTACHMENT TOO LARGE', mail.call_args.args[1])
        self.assertIn('full.tar.gz', mail.call_args.args[2])
        stop.assert_called_once()

    def test_preflight_failure_and_check_only_never_start_or_stop(self):
        fake = SimpleNamespace(make_plan=MagicMock(return_value={'test': True}),
            create_suite=MagicMock(), validate_resume=MagicMock(return_value={'test': True}))
        with patch.dict(sys.modules, {'spatial_experiment': fake}), \
                patch.object(runner, 'arguments', side_effect=parser), \
                patch('torch.cuda.is_available', return_value=True), \
                patch.dict(os.environ, {'GMAIL_USER': 'sender', 'GMAIL_APP_PASSWORD': 'private'}), \
                patch.object(runner, 'load_email_credentials'), \
                patch.object(runner, 'preflight', side_effect=RuntimeError('no permission')), \
                patch.object(runner, 'send_mail') as mail, \
                patch.object(runner, 'stop_workbench') as stop, \
                patch.object(runner.subprocess, 'Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'permission'):
                runner.main([])
            fake.create_suite.assert_not_called()
            process.assert_not_called()
            stop.assert_not_called()
            mail.assert_not_called()
        with patch.dict(sys.modules, {'spatial_experiment': fake}), \
                patch.object(runner, 'arguments', side_effect=parser), \
                patch('torch.cuda.is_available', return_value=True), \
                patch.dict(os.environ, {'GMAIL_USER': 'sender', 'GMAIL_APP_PASSWORD': 'private'}), \
                patch.object(runner, 'load_email_credentials'), \
                patch.object(runner, 'preflight', return_value='url'), \
                patch.object(runner, 'send_mail') as mail, \
                patch.object(runner, 'stop_workbench') as stop, \
                patch.object(runner.subprocess, 'Popen') as process:
            runner.main(['--check-only', '--resume', 'suite'])
            fake.validate_resume.assert_called_once_with('suite')
            fake.create_suite.assert_not_called()
            process.assert_not_called()
            stop.assert_not_called()
            mail.assert_called_once()

    def test_child_is_logged_and_only_started_child_arms_finish(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)/'suite'
            suite.mkdir()
            fake = SimpleNamespace(make_plan=MagicMock(return_value={'test': True}),
                create_suite=MagicMock(return_value=suite), validate_resume=MagicMock())
            child = FakeChild()
            with patch.dict(sys.modules, {'spatial_experiment': fake}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.dict(os.environ, {'GMAIL_USER': 'sender', 'GMAIL_APP_PASSWORD': 'private'}), \
                    patch.object(runner, 'load_email_credentials'), \
                    patch.object(runner, 'preflight', return_value='url'), \
                    patch.object(runner, 'send_mail'), \
                    patch.object(runner.subprocess, 'Popen', return_value=child) as process, \
                    patch.object(runner, 'finish') as finish:
                with self.assertRaises(SystemExit) as result:
                    runner.main([])
                self.assertEqual(result.exception.code, 0)
                env = process.call_args.kwargs['env']
                self.assertNotIn('GMAIL_USER', env)
                self.assertNotIn('GMAIL_APP_PASSWORD', env)
                self.assertEqual(process.call_args.args[0][-2:], ['--resume', str(suite)])
                finish.assert_called_once_with(suite, suite/'overnight.log', 'url', False,
                                                'recipient@example.com', 0)
            log = (suite/'overnight.log').read_text()
            self.assertIn('Training complete', log)
            self.assertIn('Child exit code: 0', log)

    def test_resume_protocol_changes_fail_before_email_or_stop(self):
        fake = SimpleNamespace(make_plan=MagicMock(), create_suite=MagicMock(),
            validate_resume=MagicMock(return_value={'config': {'training': {'epochs': 100}}}))
        with patch.dict(sys.modules, {'spatial_experiment': fake}), \
                patch.object(runner, 'arguments', side_effect=parser), \
                patch('torch.cuda.is_available', return_value=True), \
                patch.dict(os.environ, {'GMAIL_USER': 'sender', 'GMAIL_APP_PASSWORD': 'private'}), \
                patch.object(runner, 'load_email_credentials'), \
                patch.object(runner, 'preflight') as preflight, \
                patch.object(runner, 'send_mail') as mail, \
                patch.object(runner, 'stop_workbench') as stop:
            for extra in (['--epochs', '200'], ['--smoke']):
                with self.assertRaisesRegex(ValueError, 'saved smoke/full'):
                    runner.main(['--resume', 'suite', *extra])
            preflight.assert_not_called()
            mail.assert_not_called()
            stop.assert_not_called()

    def test_spawn_failure_leaves_shutdown_unarmed(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)/'suite'
            suite.mkdir()
            fake = SimpleNamespace(make_plan=MagicMock(return_value={}),
                create_suite=MagicMock(return_value=suite), validate_resume=MagicMock())
            with patch.dict(sys.modules, {'spatial_experiment': fake}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.dict(os.environ, {'GMAIL_USER': 'sender', 'GMAIL_APP_PASSWORD': 'private'}), \
                    patch.object(runner, 'load_email_credentials'), \
                    patch.object(runner, 'preflight', return_value='url'), \
                    patch.object(runner, 'send_mail'), \
                    patch.object(runner.subprocess, 'Popen', side_effect=OSError('Cannot start')), \
                    patch.object(runner, 'finish') as finish:
                with self.assertRaisesRegex(OSError, 'Cannot start'):
                    runner.main([])
                finish.assert_not_called()

    def test_interrupt_terminates_child_then_notifies_and_stops(self):
        def interrupted_output():
            yield 'Epoch in progress\n'
            raise KeyboardInterrupt()

        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)/'suite'
            suite.mkdir()
            fake = SimpleNamespace(make_plan=MagicMock(return_value={}),
                create_suite=MagicMock(return_value=suite), validate_resume=MagicMock())
            child = FakeChild(output=interrupted_output())
            with patch.dict(sys.modules, {'spatial_experiment': fake}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.dict(os.environ, {'GMAIL_USER': 'sender', 'GMAIL_APP_PASSWORD': 'private'}), \
                    patch.object(runner, 'load_email_credentials'), \
                    patch.object(runner, 'preflight', return_value='url'), \
                    patch.object(runner, 'send_mail'), \
                    patch.object(runner.subprocess, 'Popen', return_value=child), \
                    patch.object(runner, 'finish') as finish:
                with self.assertRaises(KeyboardInterrupt):
                    runner.main([])
                self.assertTrue(child.terminated)
                self.assertEqual(finish.call_args.args[-1], 1)
            self.assertIn('KeyboardInterrupt', (suite/'overnight.log').read_text())


if __name__ == '__main__':
    unittest.main()
