import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import mechanism_suite as suite


class Sized:
    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def parent_config():
    return dict(dataset=dict(data_dir='./data', download=True, num_workers=4, batch_size=64),
                training=dict(epochs=100, learning_rate=0.0005, seed=42))


class MechanismSuiteTests(unittest.TestCase):
    def make(self, root, extra=()):
        parent = root/'parent'
        parent.mkdir(exist_ok=True)
        (parent/'best.pt').write_bytes(b'immutable parent')
        summary = dict(seed=42, split_hash='fixed-split', best_epoch=100, completed_epochs=100)
        args = suite.arguments().parse_args(['--parent-run', str(parent), '--output', str(root/'new'), *extra])
        with patch('mechanism_diagnostics.load_spatial_parent', return_value=(
                parent, summary, dict(config=parent_config()), suite.file_hash(parent/'best.pt'))), \
                patch.object(suite, 'clean_datasets', return_value=(Sized(80000), Sized(10000), [], 'fixed-split', {})), \
                patch.object(suite, 'source_hashes', return_value={'new.py': 'identity'}), \
                patch.object(suite, 'execution_environment', return_value={'device': 'fixture'}):
            plan = suite.make_plan(args)
        return args, plan

    def test_default_uses_original_recipe_and_three_additional30_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, plan = self.make(root)
            self.assertEqual(plan['epochs'], 30)
            self.assertEqual(plan['config']['training']['epochs'], 100)
            self.assertEqual(plan['config']['training']['learning_rate'], 0.0005)
            self.assertEqual(plan['seed'], 42)
            self.assertEqual(plan['jobs'], ['cnn8_control', 'unet32', 'unet32_noise'])
            self.assertFalse(plan['config']['dataset']['download'])
            self.assertFalse((root/'new').exists())
            result = suite.create_suite(args, plan)
            self.assertEqual(json.loads((result/'suite.json').read_text())['epochs'], 30)
            self.assertEqual((root/'parent/best.pt').read_bytes(), b'immutable parent')

    def test_budget_and_source_directory_are_bounded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for epochs in ('0', '31'):
                with self.assertRaisesRegex(ValueError, '1..30'):
                    self.make(root, ['--epochs', epochs])
            _, plan = self.make(root, ['--smoke'])
            self.assertEqual(plan['epochs'], 1)
            self.assertTrue(plan['smoke'])
            with self.assertRaisesRegex(ValueError, 'outside'):
                suite._outside_parent(root/'parent'/'child', root/'parent')

    def run_stages(self, root, diagnostic=None, training=None):
        (root/'parent').mkdir()
        (root/'parent/best.pt').write_bytes(b'unchanged')
        plan = dict(parent_run=str(root/'parent'), config=parent_config(), examples=0,
                    epochs=30, probe_every=5, smoke=False,
                    parent_checkpoint_sha256=suite.file_hash(root/'parent/best.pt'))
        with patch.object(suite, 'validate_resume', return_value=plan), \
                patch('mechanism_diagnostics.run_diagnostics', diagnostic or Mock()), \
                patch('fusion30_experiment.train_fusion_suite', training or Mock()), \
                patch('torch.cuda.is_available', return_value=True), \
                patch('torch.cuda.empty_cache'):
            return suite.run_suite(root)

    def test_both_stages_run_and_full_completion_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            diagnostic, training = Mock(), Mock()
            result = self.run_stages(root, diagnostic, training)
            self.assertEqual(result['status'], 'complete')
            self.assertFalse(result['audit_partition_evaluated'])
            self.assertFalse(result['official_validation_evaluated'])
            self.assertEqual(diagnostic.call_count, 1)
            self.assertEqual(training.call_args.kwargs['epochs'], 30)
            self.assertTrue((root/'completion.json').exists())

    def test_diagnostic_failure_preserves_independent_training_and_nonzero_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training = Mock()
            with self.assertRaisesRegex(RuntimeError, 'stages failed'):
                self.run_stages(root, Mock(side_effect=ValueError('diagnostic fixture')), training)
            self.assertEqual(training.call_count, 1)
            status = json.loads((root/'status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['outcomes']['fusion30']['status'], 'complete')
            self.assertTrue((root/'mechanism/stage_failure.json').exists())
            self.assertFalse((root/'completion.json').exists())

    def test_interrupt_does_not_start_other_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            training = Mock()
            with self.assertRaises(KeyboardInterrupt):
                self.run_stages(Path(temp), Mock(side_effect=KeyboardInterrupt()), training)
            training.assert_not_called()

    def test_resume_rejects_source_change_before_using_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, plan = self.make(root)
            (root/'suite.json').write_text(json.dumps(plan))
            with patch.object(suite, 'source_hashes', return_value={'new.py': 'changed'}):
                with self.assertRaisesRegex(ValueError, 'source files'):
                    suite.validate_resume(root)


if __name__ == '__main__':
    unittest.main()
