import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import generalization_suite as runner


class Sized:
    def __init__(self, size): self.size = size
    def __len__(self): return self.size


class OvernightPlanTests(unittest.TestCase):
    def make(self, root, extra=()):
        parent = root/'parent'; parent.mkdir(exist_ok=True)
        (parent/'best.pt').write_bytes(b'parent')
        config = dict(dataset=dict(data_dir='data', download=True, num_workers=4, batch_size=64),
                      training=dict(seed=42, epochs=100, learning_rate=.0005))
        summary = dict(seed=42, split_hash='split')
        args = runner.arguments().parse_args(['--parent-run', str(parent),
            '--previous-fusion', str(root/'previous'), '--output', str(root/'new'), *extra])
        with patch.object(runner, 'load_spatial_parent', return_value=(
                parent, summary, dict(config=config), runner.file_hash(parent/'best.pt'))), \
                patch.object(runner, 'clean_datasets', return_value=(Sized(80000), Sized(10000), [], 'split', {})), \
                patch.object(runner, 'source_hashes', return_value={'source': 'hash'}), \
                patch.object(runner, 'execution_environment', return_value={'device': 'fixture'}):
            plan = runner.make_plan(args)
        return args, plan

    def test_defaults_match12h_budget_and_separate_training_horizons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, plan = self.make(root)
            self.assertEqual((plan['max_hours'], plan['warm_epochs'], plan['fresh_epochs']), (12., 30, 100))
            self.assertEqual(plan['seed'], 42)
            self.assertEqual(plan['config']['training']['epochs'], 100)
            self.assertFalse(plan['config']['dataset']['download'])
            self.assertFalse((root/'new').exists())
            path = runner.create_suite(args, plan)
            self.assertTrue((path/'suite.json').is_file())

    def test_budget_values_and_source_output_are_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for args in (['--warm-epochs', '31'], ['--fresh-epochs', '101'],
                         ['--max-hours', 'nan'], ['--reserve-minutes', '400']):
                with self.assertRaises(ValueError): self.make(root, args)
            _, plan = self.make(root, ['--smoke'])
            self.assertEqual((plan['warm_epochs'], plan['fresh_epochs']), (1, 1))
            with self.assertRaisesRegex(ValueError, 'outside'):
                runner._outside(root/'parent/child', root/'parent', root/'previous')

    def test_active_budget_retains_elapsed_runtime_across_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            clock = [100.]
            with patch.object(runner.time, 'monotonic', side_effect=lambda: clock[0]):
                budget = runner.ActiveBudget(root, 12.)
                clock[0] += 75.
                budget.write()
                resumed = runner.ActiveBudget(root, 12.)
                self.assertEqual(resumed.previous, 75.)
                self.assertEqual(resumed.remaining(), 12*3600-75)
                with self.assertRaisesRegex(ValueError, 'different'):
                    runner.ActiveBudget(root, 10.)

    def run_stages(self, root, audit=None, training=None):
        parent = root/'parent'; parent.mkdir()
        (parent/'best.pt').write_bytes(b'unchanged')
        plan = dict(parent_run=str(parent), previous_fusion=str(root/'old'), config={}, seed=42,
            parent_checkpoint_sha256=runner.file_hash(parent/'best.pt'), max_hours=12., reserve_minutes=45,
            warm_epochs=30, fresh_epochs=100, examples=0, smoke=False)
        with patch.object(runner, 'validate_resume', return_value=plan), \
                patch('generalization_audit.run_data_audit', audit or Mock(return_value=dict(status='complete', training_allowed=True))), \
                patch('generalization_last.run_previous_last', return_value=dict(status='complete')) as previous, \
                patch('generalization_experiment.run_training_study', training or Mock(return_value=dict(status='complete'))) as trained, \
                patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                patch.object(runner.torch.cuda, 'empty_cache'):
            result = runner.run_suite(root)
            return result, previous, trained

    def test_all_stages_get_parent_recipe_and_training_horizons(self):
        with tempfile.TemporaryDirectory() as temp:
            result, previous, trained = self.run_stages(Path(temp))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(trained.call_args.kwargs['epochs'], 30)
            self.assertEqual(trained.call_args.kwargs['fresh_epochs'], 100)
            self.assertEqual(previous.call_count, 1)

    def test_fatal_data_error_blocks_further_gpu_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training = Mock()
            with self.assertRaisesRegex(RuntimeError, 'failed stages'):
                self.run_stages(root, Mock(return_value=dict(status='complete', training_allowed=False)), training)
            training.assert_not_called()
            result = json.loads((root/'completion.json').read_text())
            self.assertEqual(result['outcomes']['training']['status'], 'blocked')

    def test_time_limit_is_partial_without_faking_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            result, _, _ = self.run_stages(Path(temp), training=Mock(return_value=dict(status='budget_exhausted')))
            self.assertEqual(result['status'], 'budget_exhausted')
            self.assertFalse(result['official_validation_evaluated'])


if __name__ == '__main__':
    unittest.main()
