import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import recover_fusion_analysis as runner


class RecoveryTests(unittest.TestCase):
    def fixture(self, root):
        import pairwise_resolver_experiment as pairwise
        import hcsi_tiny_experiment as hcsi
        queue = root/'queue'
        queue.mkdir()
        children, child_plans, counts = {}, {}, {}
        summary = dict(status='complete', completed_epochs=2,
            best_epoch=1, best_source='ema', best_selection=dict(accuracy=.65, nll=1., n=10000))
        for name, module in (('pairwise', pairwise), ('hcsi', hcsi)):
            suite = queue/name/'suite'
            suite.mkdir(parents=True)
            children[name] = suite
            child_plans[name] = dict(epochs=2, examples=8)
            for job in module.JOBS:
                run = suite/'training'/job
                run.mkdir(parents=True)
                (run/'summary.json').write_text(json.dumps(summary))
                (run/'last.pt').write_bytes(('last-'+job).encode())
                (run/'best.pt').write_bytes(('best-'+job).encode())
                (run/'metadata.json').write_text('{}')
                counts[name+'/'+job] = dict(completed_epochs=2, planned_epochs=2)
            completion = dict(status='complete' if name == 'pairwise' else 'failed',
                failure=None if name == 'pairwise' else dict(error_type='ValueError', message='old source scope bug'))
            (suite/'completion.json').write_text(json.dumps(completion))
            (suite/'status.json').write_text(json.dumps(completion))
            (suite/'report.md').write_text('Original '+name+' report')
            if name == 'pairwise':
                (suite/'feature_analysis').mkdir()
                (suite/'feature_analysis/completion.json').write_text('{"status":"complete"}')
                (suite/'feature_analysis/report.md').write_text('Completed pairwise evidence')
            else:
                (suite/'failure.json').write_text(json.dumps(completion['failure']))
        (queue/'completion.json').write_text('{"status":"failed","exit_code":1}')
        (queue/'status.json').write_text('{"status":"failed"}')
        (queue/'report.md').write_text('Original failed parent report')
        record = dict(protocol='tiny-combined-fusion-overnight-v1', source_hashes={'old':'sha'},
            child_plans=child_plans, children=[dict(name=n, suite=str(p)) for n,p in children.items()])
        (queue/'queue.json').write_text(json.dumps(record))
        return queue, record, children, counts, runner.artifact_inventory(children)

    def loaded(self, plan, training, job, checkpoint='best.pt'):
        if checkpoint == 'last.pt':
            state = dict(epoch=2, history=[dict(epoch=i) for i in range(3)])
        else:
            state = dict(epoch=1, source='ema', selection=dict(accuracy=.65, nll=1., n=10000))
        return torch.nn.Linear(1,1), state

    def test_preflight_requires_and_reads_all_nine_completed_checkpoints(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, record, children, counts, inventory = self.fixture(Path(temp))
            with patch('run_fusion_overnight.validate_resume', return_value=record), \
                    patch('pairwise_resolver_experiment.load_resolver_model', side_effect=self.loaded) as pair_load, \
                    patch('hcsi_tiny_experiment.load_model', side_effect=self.loaded) as hcsi_load, \
                    patch('pairwise_resolver_experiment.train_job') as pair_train, \
                    patch('hcsi_tiny_experiment.train_job') as hcsi_train:
                actual = runner.validate_completed_queue(queue)
            self.assertEqual(len(actual[3]), 9)
            self.assertEqual(pair_load.call_count, 8)
            self.assertEqual(hcsi_load.call_count, 10)
            pair_train.assert_not_called(); hcsi_train.assert_not_called()
            self.assertEqual(actual[4], inventory)

    def test_incomplete_training_refuses_analysis_only_recovery(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, record, children, _, _ = self.fixture(Path(temp))
            path = children['pairwise']/'training/pairwise/summary.json'
            value = json.loads(path.read_text()); value['completed_epochs'] = 1
            path.write_text(json.dumps(value))
            with patch('run_fusion_overnight.validate_resume', return_value=record):
                with self.assertRaisesRegex(ValueError, 'did not finish'):
                    runner.validate_completed_queue(queue)

    def test_summary_and_best_checkpoint_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, record, _, _, _ = self.fixture(Path(temp))
            def wrong(*args, **kwargs):
                model, state = self.loaded(*args, **kwargs)
                if kwargs.get('checkpoint') == 'best.pt': state['epoch'] = 0
                return model, state
            with patch('run_fusion_overnight.validate_resume', return_value=record), \
                    patch('pairwise_resolver_experiment.load_resolver_model', side_effect=wrong):
                with self.assertRaisesRegex(ValueError, 'best checkpoint'):
                    runner.validate_completed_queue(queue)

    def test_recovery_keeps_models_and_pairwise_bytes_and_archives_failed_markers(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, record, children, counts, inventory = self.fixture(Path(temp))
            originals = {name: (path/'completion.json').read_bytes() for name,path in children.items()}
            old_parent = (queue/'completion.json').read_bytes()
            attempt = runner.create_attempt(queue, children, record, counts, inventory)
            runner.publish_recovery(queue, record, children, attempt, dict(status='complete'))
            runner.verify_inventory(inventory)
            self.assertEqual((children['pairwise']/'completion.json').read_bytes(), originals['pairwise'])
            self.assertEqual((attempt/'original/hcsi/completion.json').read_bytes(), originals['hcsi'])
            self.assertEqual((attempt/'original/queue/completion.json').read_bytes(), old_parent)
            self.assertFalse((children['hcsi']/'failure.json').exists())
            self.assertTrue((attempt/'original/hcsi/failure.json').is_file())
            self.assertEqual(json.loads((queue/'completion.json').read_text())['status'], 'complete')
            self.assertEqual(json.loads((children['hcsi']/'completion.json').read_text())['status'], 'complete')

    def test_inventory_detects_changed_completed_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, record, children, counts, inventory = self.fixture(Path(temp))
            (children['hcsi']/'training/hcsi/best.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(RuntimeError, 'altered'):
                runner.verify_inventory(inventory)

    def test_stale_failure_is_never_removed_without_matching_backup(self):
        with tempfile.TemporaryDirectory() as temp:
            queue, record, children, counts, inventory = self.fixture(Path(temp))
            attempt = runner.create_attempt(queue, children, record, counts, inventory)
            (attempt/'original/hcsi/failure.json').write_text('changed backup')
            with self.assertRaisesRegex(RuntimeError, 'backup or current marker changed'):
                runner.publish_recovery(queue, record, children, attempt, dict(status='complete'))
            self.assertTrue((children['hcsi']/'failure.json').exists())
            self.assertEqual(json.loads((queue/'completion.json').read_text())['status'], 'failed')

    def test_success_recovers_only_analysis_then_finalizes_before_email_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.fixture(Path(temp))
            queue, record, children, counts, inventory = fixture
            events = []
            def analysis(plan, training, output, examples):
                events.append('analysis')
                output.mkdir(exist_ok=True)
                (output/'completion.json').write_text('{"status":"complete"}')
                return dict(status='complete')
            def delivery(*args, **kwargs):
                events.append('mail')
                self.assertEqual(json.loads((queue/'completion.json').read_text())['status'], 'complete')
                kwargs['before_stop']()
                self.assertTrue(json.loads((queue/'analysis_recovery_status.json').read_text())['invocation_finished'])
                events.append('stop')
                return True
            with patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                    patch.object(runner, 'validate_completed_queue', return_value=fixture), \
                    patch.object(runner, 'check_email_ready'), patch.object(runner, 'preflight', return_value='target'), \
                    patch.object(runner, 'run_repaired_analysis', side_effect=analysis), \
                    patch.object(runner, 'finish', side_effect=delivery) as finish, \
                    patch.object(runner.subprocess, 'Popen') as process, \
                    patch('pairwise_resolver_experiment.train_job') as pair_train, \
                    patch('hcsi_tiny_experiment.train_job') as hcsi_train:
                code = runner.main(['--queue', str(queue)])
            self.assertEqual(code, 0)
            self.assertEqual(events, ['analysis','mail','stop'])
            process.assert_not_called(); pair_train.assert_not_called(); hcsi_train.assert_not_called()
            finish.assert_called_once()
            runner.verify_inventory(inventory)

    def test_failed_analysis_retains_failed_evidence_but_still_delivers_and_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.fixture(Path(temp)); queue = fixture[0]
            old = (queue/'completion.json').read_bytes()
            with patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                    patch.object(runner, 'validate_completed_queue', return_value=fixture), \
                    patch.object(runner, 'check_email_ready'), patch.object(runner, 'preflight', return_value='target'), \
                    patch.object(runner, 'run_repaired_analysis', side_effect=RuntimeError('analysis error')), \
                    patch.object(runner, 'finish', return_value=False) as finish:
                code = runner.main(['--queue', str(queue)])
            self.assertEqual(code, 1); finish.assert_called_once()
            self.assertEqual((queue/'completion.json').read_bytes(), old)
            self.assertEqual(finish.call_args.args[3], 1)
            self.assertEqual(json.loads((queue/'analysis_recovery_status.json').read_text())['status'], 'failed')

    def test_check_only_and_preflight_failure_never_create_attempt_send_or_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.fixture(Path(temp)); queue = fixture[0]
            with patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                    patch.object(runner, 'validate_completed_queue', return_value=fixture), \
                    patch.object(runner, 'check_email_ready'), patch.object(runner, 'preflight', return_value='target'), \
                    patch.object(runner, 'run_repaired_analysis') as analysis, \
                    patch.object(runner, 'finish') as finish:
                self.assertEqual(runner.main(['--queue',str(queue),'--check-only']), 0)
            analysis.assert_not_called(); finish.assert_not_called()
            self.assertFalse((queue/'analysis_recovery').exists())
            with patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                    patch.object(runner, 'validate_completed_queue', side_effect=ValueError('invalid')), \
                    patch.object(runner, 'finish') as finish:
                with self.assertRaises(ValueError): runner.main(['--queue',str(queue)])
            finish.assert_not_called()

    def test_status_record_failure_cannot_prevent_final_delivery_and_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            fixture = self.fixture(Path(temp)); queue = fixture[0]
            original_write = runner._write_json
            def write(path, value):
                if Path(path).name == 'analysis_recovery_status.json':
                    raise OSError('Simulated final record failure')
                return original_write(path, value)
            with patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                    patch.object(runner, 'validate_completed_queue', return_value=fixture), \
                    patch.object(runner, 'check_email_ready'), patch.object(runner, 'preflight', return_value='target'), \
                    patch.object(runner, 'run_repaired_analysis', return_value=dict(status='complete')), \
                    patch.object(runner, '_write_json', side_effect=write), \
                    patch.object(runner, 'finish', return_value=True) as finish:
                self.assertEqual(runner.main(['--queue',str(queue)]), 0)
            finish.assert_called_once()
            self.assertTrue(callable(finish.call_args.kwargs['before_stop']))


if __name__ == '__main__': unittest.main()
