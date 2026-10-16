import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import run_notified


class NotificationTests(unittest.TestCase):
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
