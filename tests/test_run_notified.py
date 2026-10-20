import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_notified


class NotificationTests(unittest.TestCase):
    def test_attachment_mime(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory)/'analysis.tar.gz'
            archive.write_bytes(b'archive')
            with patch.dict(os.environ, {'GMAIL_USER': 'sender@example.com',
                                        'GMAIL_APP_PASSWORD': 'test-app-password'}), \
                    patch.object(run_notified.smtplib, 'SMTP_SSL') as smtp:
                run_notified.send_mail('user@example.com', 'done', 'body', archive)
                message = smtp.return_value.__enter__.return_value.send_message.call_args.args[0]
                parts = list(message.iter_attachments())
                self.assertEqual(len(parts), 1)
                self.assertEqual(parts[0].get_filename(), 'analysis.tar.gz')
                self.assertEqual(parts[0].get_payload(decode=True), b'archive')

    def test_completion_attaches_analysis_without_shutdown(self):
        import overnight_correction
        for exit_code in (0, 7):
            with tempfile.TemporaryDirectory() as directory:
                suite = Path(directory)/'suite'
                suite.mkdir()
                (suite/'summary.json').write_text('{}')
                with patch.object(run_notified, 'send_mail') as mail, \
                        patch.object(overnight_correction, 'stop_workbench') as stop:
                    with self.assertRaises(SystemExit) as result:
                        run_notified.main(['--to', 'user@example.com', '--log', str(suite/'training.log'),
                                           '--analysis-suite', str(suite), '--', sys.executable,
                                           '-c', f'print("done"); raise SystemExit({exit_code})'])
                    self.assertEqual(result.exception.code, exit_code)
                    self.assertTrue(mail.call_args.kwargs['attachment'].is_file())
                    stop.assert_not_called()

    def test_daytime_wrapper_only_schedules_correction(self):
        import run_correction_notified as daytime
        import correction_experiment
        import torch
        with patch.object(sys, 'argv', ['daytime', '--epochs', '25']), \
                patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(correction_experiment, 'create_suite', return_value=Path('runs/test')) as create, \
                patch.object(daytime, 'notify_run') as notify:
            daytime.main()
            self.assertEqual(create.call_args.args[0].variants, ['attention_correction'])
            self.assertEqual(create.call_args.args[0].correction_epochs, 25)
            command = notify.call_args.args[0]
            self.assertIn('--analysis-suite', command)
            self.assertFalse(any('overnight' in s for s in command))

    def test_missing_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                run_notified.send_mail('user@example.com', 'test', 'body')

    def test_test_only_does_not_start_process(self):
        with patch.object(sys, 'argv', ['run_notified', '--to', 'user@example.com', '--test-email']), \
                patch.object(run_notified, 'send_mail') as mail, \
                patch.object(run_notified.subprocess, 'Popen') as process:
            run_notified.main()
            mail.assert_called_once()
            process.assert_not_called()

    def test_success_and_failure(self):
        for code in (0, 7):
            with self.subTest(code=code), tempfile.TemporaryDirectory() as directory:
                log = Path(directory)/'run.log'
                argv = ['run_notified', '--to', 'user@example.com', '--log', str(log),
                        '--', sys.executable, '-c', f'print("finished"); raise SystemExit({code})']
                with patch.object(sys, 'argv', argv), \
                        patch.object(run_notified, 'send_mail') as mail:
                    with self.assertRaises(SystemExit) as result:
                        run_notified.main()
                    self.assertEqual(result.exception.code, code)
                    self.assertEqual(mail.call_count, 2)
                    self.assertIn(f'Exit code: {code}', mail.call_args.args[2])
                    self.assertIn('finished', log.read_text())

    def test_preflight_failure_blocks_training(self):
        with patch.object(sys, 'argv', ['run_notified', '--to', 'user@example.com', '--', 'python']), \
                patch.object(run_notified, 'send_mail', side_effect=RuntimeError('secret')), \
                patch.object(run_notified.subprocess, 'Popen') as process:
            with self.assertRaises(SystemExit) as result:
                run_notified.main()
            self.assertNotIn('secret', str(result.exception))
            process.assert_not_called()


if __name__ == '__main__':
    unittest.main()
