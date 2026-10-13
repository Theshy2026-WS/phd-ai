import argparse
import unittest
from unittest.mock import patch
import overnight_twostage as overnight


class OvernightTests(unittest.TestCase):
    def test_command_and_resume(self):
        args = argparse.Namespace(resume=None, epochs=100)
        cmd = overnight.training_command(args)
        self.assertIn('conformer_tiny', cmd)
        self.assertNotIn('acmix50', cmd)
        self.assertEqual(cmd[-4:], ['--seeds', '42', '--epochs', '100'])
        args.resume = 'runs/suite'
        self.assertEqual(overnight.training_command(args)[-2:], ['--resume', 'runs/suite'])

    @patch.object(overnight, 'metadata', side_effect=['project', 'zones/zone', 'instance'])
    @patch.object(overnight, 'api')
    def test_preflight_never_stops(self, api, metadata):
        api.side_effect = [{'name': 'instance'}, {'permissions': ['notebooks.instances.stop']}]
        self.assertTrue(overnight.preflight().endswith('/instances/instance'))
        self.assertFalse(any(':stop' in c.args[0] for c in api.call_args_list))

    @patch.object(overnight, 'metadata', side_effect=['project', 'zones/zone', 'instance'])
    @patch.object(overnight, 'api', side_effect=[{'name': 'instance'}, {'permissions': []}])
    def test_preflight_missing_permission(self, api, metadata):
        with self.assertRaisesRegex(RuntimeError, 'No training started'):
            overnight.preflight()

    @patch.object(overnight.time, 'sleep')
    @patch.object(overnight, 'api')
    def test_stop_success(self, api, sleep):
        api.side_effect = [{'state': 'ACTIVE'}, {'name': 'projects/p/locations/z/operations/op'},
                           {'done': True}, {'state': 'STOPPED'}]
        overnight.stop_workbench('https://notebooks.googleapis.com/v2/instance')
        self.assertEqual(api.call_args_list[1].args[1], {})

    @patch.object(overnight.subprocess, 'run')
    @patch.object(overnight.time, 'sleep')
    @patch.object(overnight, 'api', side_effect=RuntimeError('offline'))
    def test_stop_failure_fallback_opt_in(self, api, sleep, run):
        with self.assertRaisesRegex(RuntimeError, 'stop the instance manually'):
            overnight.stop_workbench('url')
        run.assert_not_called()
        overnight.stop_workbench('url', fallback=True)
        run.assert_called_once_with(['sudo', '-n', '/usr/sbin/shutdown', '-h', 'now'], check=True)
