from pathlib import Path
import unittest
from unittest.mock import patch

from run_dual_overnight import finish


class OvernightTests(unittest.TestCase):
    @patch('run_dual_overnight.stop_workbench')
    @patch('run_dual_overnight.send_mail')
    @patch('run_dual_overnight.bundle_results', return_value=Path('results.tar.gz'))
    def test_mail_before_stop(self, bundle, mail, stop):
        events = []
        mail.side_effect = lambda *a, **k: events.append('mail')
        stop.side_effect = lambda *a: events.append('stop')
        finish('suite', 'log', 'url', True, 'recipient', 0)
        self.assertEqual(events, ['mail', 'stop'])
        self.assertEqual(mail.call_args.kwargs['attachment'], Path('results.tar.gz'))

    @patch('run_dual_overnight.time.sleep')
    @patch('run_dual_overnight.stop_workbench')
    @patch('run_dual_overnight.send_mail', side_effect=RuntimeError('SMTP failure'))
    @patch('run_dual_overnight.bundle_results', side_effect=OSError('disk full'))
    def test_failures_still_stop(self, bundle, mail, stop, sleep):
        finish('suite', 'log', 'url', True, 'recipient', 1)
        self.assertEqual(mail.call_count, 3)
        stop.assert_called_once_with('url', True)


if __name__ == '__main__':
    unittest.main()
