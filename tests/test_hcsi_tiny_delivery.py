import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import run_hcsi_tiny as runner


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--resume'); p.add_argument('--epochs', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--to', default='recipient@example.com')
    p.add_argument('--max-hours', type=runner.hours_argument, default=10.)
    return p


class FakeChild:
    def __init__(self, code=0):
        self.stdout = iter(['Five-arm queue finished\n'])
        self.code, self.done = code, False
        self.terminated = self.killed = False
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def poll(self): return self.code if self.done else None
    def wait(self, timeout=None): self.done = True; return self.code
    def terminate(self): self.terminated = True
    def kill(self): self.killed = True


class StalledChild(FakeChild):
    def wait(self, timeout=None):
        if not self.killed:
            raise subprocess.TimeoutExpired('fake', timeout)
        return super().wait(timeout)


class HCSIDeliveryTests(unittest.TestCase):
    def fixture(self, root, status='complete'):
        suite = root/'suite'; suite.mkdir()
        for name in ('training', 'feature_analysis'): (suite/name).mkdir()
        (suite/'completion.json').write_text(json.dumps(dict(status=status)))
        module = SimpleNamespace(make_plan=MagicMock(return_value=dict(epochs=30)),
            create_suite=MagicMock(return_value=suite), validate_resume=MagicMock(return_value=dict(epochs=30)))
        return suite, module

    def package(self):
        return dict(full=Path('reports.zip'), compact=None, attachment=Path('reports.zip'))

    def test_budget_cannot_exceed_ten_hours_and_reserves_fifteen_minutes(self):
        b = runner.budget_deadlines(100, 10)
        self.assertEqual(b['total_deadline_monotonic'], 36100)
        self.assertEqual(b['child_deadline_monotonic'], 35200)
        self.assertEqual(b['delivery_reserve_seconds'], 900)
        for invalid in (0, -1, .01, 11, float('nan'), float('inf')):
            with self.assertRaises(argparse.ArgumentTypeError): runner.hours_argument(invalid)

    def test_watchdog_stops_stalled_child_without_touching_vm(self):
        child = StalledChild()
        with patch('run_vit_augreg_overnight.stop_workbench') as stop, \
                patch.object(runner.subprocess, 'run') as command:
            watchdog = runner.ChildWatchdog(child, time.monotonic()-1, grace_seconds=.01)
            watchdog._run()
        self.assertTrue(watchdog.timed_out.is_set())
        self.assertTrue(child.terminated); self.assertTrue(child.killed)
        stop.assert_not_called(); command.assert_not_called()

    def test_cancelled_watchdog_does_not_terminate_child(self):
        child = FakeChild()
        watchdog = runner.ChildWatchdog(child, time.monotonic()+100)
        watchdog.cancelled.set(); watchdog._run()
        self.assertFalse(child.terminated); self.assertFalse(watchdog.timed_out.is_set())

    def test_owned_process_group_is_cleaned_even_after_main_exits(self):
        child = FakeChild(); child.pid = 12345
        child._resolver_owned_group = True; child.done = True
        with patch.object(runner.os, 'name', 'posix'), \
                patch.object(runner.os, 'killpg', create=True) as killpg:
            watchdog = runner.ChildWatchdog(child, time.monotonic()-1, grace_seconds=.01)
            watchdog._run()
        self.assertTrue(watchdog.timed_out.is_set())
        self.assertEqual([call.args[0] for call in killpg.call_args_list], [12345, 12345])
        self.assertEqual(killpg.call_args_list[0].args[1], runner.signal.SIGTERM)
        self.assertEqual(killpg.call_args_list[-1].args[1], getattr(runner.signal, 'SIGKILL', 9))

    def test_process_group_signal_requires_explicit_owned_new_session(self):
        child = FakeChild(); child.pid = 12345
        with patch.object(runner.os, 'name', 'posix'), patch.object(runner.os, 'killpg', create=True) as killpg:
            with self.assertRaises(ValueError): runner._signal_owned_group(child, runner.signal.SIGTERM)
        killpg.assert_not_called()

    def test_resume_subtracts_elapsed_time_and_cannot_raise_saved_cap(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)
            (suite/'wall_budget.json').write_text(json.dumps(dict(cap_seconds=18000,
                elapsed_active_seconds=7200, invocation_finished=True, invocation_start_unix=1)))
            hours, spent = runner.saved_budget(suite, 10)
            self.assertEqual(hours, 5); self.assertEqual(spent, 7200)
            budget = runner.budget_deadlines(100, hours, spent)
            self.assertEqual(budget['remaining_seconds'], 10800)
            self.assertEqual(runner.saved_budget(suite, 4)[0], 4)

    def test_unfinalized_invocation_is_conservatively_charged_after_crash(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)
            (suite/'wall_budget.json').write_text(json.dumps(dict(cap_seconds=36000,
                elapsed_active_seconds=1000, invocation_finished=False, invocation_start_unix=100)))
            hours, spent = runner.saved_budget(suite, 10, now_unix=200)
            self.assertEqual(spent, 1100)
            budget = runner.budget_deadlines(50, hours, spent)
            self.assertEqual(budget['remaining_seconds'], 34900)

    def test_success_emails_one_unified_package_and_keeps_vm_running(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, module = self.fixture(Path(temp)); events = []
            with patch.dict(sys.modules, {'hcsi_tiny_experiment':module}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.object(runner, 'check_email_ready'), \
                    patch.object(runner.subprocess, 'Popen', side_effect=lambda *a, **k:(events.append('child') or FakeChild())), \
                    patch.object(runner, 'bundle_results', side_effect=lambda *a:(events.append('package') or self.package())), \
                    patch.object(runner, 'send_mail', side_effect=lambda *a, **k:events.append('mail')) as mail, \
                    patch('run_vit_augreg_overnight.stop_workbench') as stop:
                with self.assertRaises(SystemExit) as result: runner.main([])
            self.assertEqual(result.exception.code, 0)
            self.assertEqual(events, ['child', 'package', 'mail']); mail.assert_called_once(); stop.assert_not_called()
            delivery = json.loads((suite/'delivery_status.json').read_text())
            self.assertFalse(delivery['stop_requested']); self.assertTrue(delivery['delivery_complete'])
            budget = json.loads((suite/'wall_budget.json').read_text())
            self.assertEqual(budget['max_hours'], 10.)

    def test_partial_or_failed_child_still_emails_without_shutdown(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, module = self.fixture(Path(temp), status='partial')
            with patch.dict(sys.modules, {'hcsi_tiny_experiment':module}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.object(runner, 'check_email_ready'), \
                    patch.object(runner.subprocess, 'Popen', return_value=FakeChild(code=4)), \
                    patch.object(runner, 'bundle_results', return_value=self.package()), \
                    patch.object(runner, 'send_mail') as mail, \
                    patch('run_vit_augreg_overnight.stop_workbench') as stop:
                with self.assertRaises(SystemExit) as result: runner.main([])
            self.assertEqual(result.exception.code, 4); mail.assert_called_once(); stop.assert_not_called()

    def test_timeout_subject_and_record_are_partial_even_if_completion_exists(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, _ = self.fixture(Path(temp))
            with patch.object(runner, 'bundle_results', return_value=self.package()), \
                    patch.object(runner, 'send_mail') as mail:
                self.assertFalse(runner.finish(suite, suite/'run.log', 'x@example.com', 124, timed_out=True))
            self.assertIn('PARTIAL / TIME LIMIT', mail.call_args.args[1])
            record = json.loads((suite/'delivery_status.json').read_text())
            self.assertTrue(record['time_limit_reached']); self.assertFalse(record['delivery_complete'])

    def test_cooperative_budget_exit_is_partial_without_watchdog_timeout(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, _ = self.fixture(Path(temp), status='budget_exhausted')
            with patch.object(runner, 'bundle_results', return_value=self.package()), \
                    patch.object(runner, 'send_mail') as mail:
                self.assertFalse(runner.finish(suite, suite/'run.log', 'x@example.com', 2))
            self.assertIn('PARTIAL / TIME LIMIT', mail.call_args.args[1])
            record = json.loads((suite/'delivery_status.json').read_text())
            self.assertTrue(record['time_limit_reached'])
            self.assertFalse(record['stop_requested'])

    def test_mail_failure_retries_but_never_stops_instance(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, _ = self.fixture(Path(temp))
            with patch.object(runner, 'bundle_results', return_value=self.package()), \
                    patch.object(runner, 'send_mail', side_effect=RuntimeError('offline')) as mail, \
                    patch.object(runner.time, 'sleep'), patch('run_vit_augreg_overnight.stop_workbench') as stop:
                self.assertFalse(runner.finish(suite, suite/'run.log', 'x@example.com', 0))
            self.assertEqual(mail.call_count, 3); stop.assert_not_called()

    def test_expired_delivery_budget_does_not_start_unbounded_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, _ = self.fixture(Path(temp))
            budget = runner.budget_deadlines(time.monotonic()-40000, 10)
            with patch.object(runner, 'bundle_results', return_value=self.package()), patch.object(runner, 'send_mail') as mail:
                self.assertFalse(runner.finish(suite, suite/'run.log', 'x@example.com', 0, budget=budget))
            mail.assert_not_called()

    def test_check_only_creates_no_suite_or_mail(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, module = self.fixture(Path(temp))
            with patch.dict(sys.modules, {'hcsi_tiny_experiment':module}), \
                    patch.object(runner, 'arguments', side_effect=parser), \
                    patch('torch.cuda.is_available', return_value=True), \
                    patch.object(runner, 'check_email_ready'), \
                    patch.object(runner.subprocess, 'Popen') as child, patch.object(runner, 'send_mail') as mail:
                runner.main(['--check-only'])
            module.create_suite.assert_not_called(); child.assert_not_called(); mail.assert_not_called()

    def test_resume_rejects_changed_horizon(self):
        with tempfile.TemporaryDirectory() as temp:
            suite, module = self.fixture(Path(temp))
            with patch.dict(sys.modules, {'hcsi_tiny_experiment':module}), \
                    patch.object(runner, 'arguments', side_effect=parser), patch('torch.cuda.is_available', return_value=True), \
                    patch.object(runner, 'check_email_ready') as ready:
                with self.assertRaises(ValueError): runner.main(['--resume', str(suite), '--epochs', '31'])
            ready.assert_not_called()

    def test_child_has_deadline_and_no_mail_credentials(self):
        command = runner.child_command('suite', 12345)
        self.assertEqual(Path(command[2]).name, 'hcsi_tiny_experiment.py')
        self.assertIn('--deadline-unix', command)
        with patch.dict(os.environ, {'GMAIL_USER':'private', 'GMAIL_APP_PASSWORD':'secret', 'SAFE':'visible'}):
            env = runner.child_environment()
        self.assertNotIn('GMAIL_USER', env); self.assertNotIn('GMAIL_APP_PASSWORD', env)
        self.assertEqual(env['SAFE'], 'visible')


if __name__ == '__main__': unittest.main()
