import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import vit_followup_suite as runner


class Sized:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n


class ViTFollowupSuiteTests(unittest.TestCase):
    def make_fixture(self, root, extra=()):
        parent = root/'parent'
        parent.mkdir(exist_ok=True)
        (parent/'best.pt').write_bytes(b'parent')
        config = dict(dataset=dict(data_dir='data', download=True, batch_size=64, num_workers=4),
                      training=dict(epochs=100, learning_rate=.0005, seed=42))
        loaded = (parent, dict(seed=42, split_hash='split', best_epoch=100), dict(config=config), 'hash')
        args = runner.arguments().parse_args(['--parent-run', str(parent), '--output', str(root/'new'), *extra])
        return args, loaded

    def plan(self, args, loaded, overlap=False):
        manifest = dict(train_clean=['train/a'], selection=['train/a' if overlap else 'selection/b'])
        with patch.object(runner, 'load_spatial_parent', return_value=loaded), \
                patch.object(runner, 'clean_datasets', return_value=(Sized(80000), Sized(10000), [], 'split', manifest)), \
                patch.object(runner, 'execution_environment', return_value={}), \
                patch.object(runner, 'source_hashes', return_value={'source': 'hash'}):
            return runner.make_plan(args)

    def test_four_jobs_have_190_epochs_and_true_native64_and_independent_branches(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, loaded = self.make_fixture(root)
            plan = self.plan(args, loaded)
            self.assertEqual(plan['total_model_epochs'], 190)
            self.assertEqual(plan['jobs'], ['vit_pretrain', 'dual_fusion', 'shallow64_attention', 'cross16_mlp'])
            self.assertEqual(plan['stages'], ['training', 'feature_analysis'])
            self.assertEqual(plan['shallow64']['tokens'], 4096)
            self.assertFalse(plan['dual_fusion']['parameter_sharing'])
            self.assertTrue(plan['dual_fusion']['cnn_trainable'])
            self.assertTrue(plan['dual_fusion']['vit_trainable'])
            self.assertEqual(plan['cross16_mlp']['depth'], 2)
            self.assertTrue(plan['complete_all'])
            self.assertIsNone(plan['max_hours'])
            self.assertEqual(plan['config']['training']['epochs'], 100)
            self.assertTrue(loaded[2]['config']['dataset']['download'])
            self.assertFalse(plan['config']['dataset']['download'])
            self.assertFalse((root/'new').exists())
            suite = runner.create_suite(args, plan)
            self.assertTrue((suite/'suite.json').is_file())

    def test_smoke_is_separate_one_epoch_protocol(self):
        with tempfile.TemporaryDirectory() as temp:
            args, loaded = self.make_fixture(Path(temp), ['--smoke'])
            plan = self.plan(args, loaded)
            self.assertEqual((plan['vit_epochs'], plan['fusion_epochs'], plan['total_model_epochs']), (1, 1, 4))
            self.assertTrue(plan['smoke'])

    def test_overlapping_development_images_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, loaded = self.make_fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'disjoint'):
                self.plan(args, loaded, overlap=True)

    def test_parent_output_and_invalid_schedule_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, loaded = self.make_fixture(Path(temp))
            args.output = str(loaded[0])
            with self.assertRaisesRegex(ValueError, 'separate directory'):
                self.plan(args, loaded)
            args.output = str(Path(temp)/'new')
            args.vit_epochs = 101
            with self.assertRaisesRegex(ValueError, 'ViT1..100'):
                self.plan(args, loaded)

    def run_stages(self, root, training=None, analysis=None):
        parent = root/'parent'
        parent.mkdir()
        (parent/'best.pt').write_bytes(b'parent')
        plan = dict(parent_run=str(parent), parent_checkpoint_sha256=runner.file_hash(parent/'best.pt'),
            config={}, vit_epochs=100, fusion_epochs=30, examples=0, smoke=False)
        trained = training or Mock(return_value=dict(status='complete'))
        analyzed = analysis or Mock(return_value=dict(status='complete'))
        with patch.object(runner, 'validate_resume', return_value=plan), \
                patch('vit_followup_experiment.run_followup_study', trained), \
                patch('vit_followup_diagnostics.run_followup_diagnostics', analyzed), \
                patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                patch.object(runner.torch.cuda, 'empty_cache'):
            result = runner.run_suite(root)
        return result, trained, analyzed

    def test_training_then_analysis_preserves_fixed_horizons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, trained, analyzed = self.run_stages(root)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(trained.call_args.kwargs['vit_epochs'], 100)
            self.assertEqual(trained.call_args.kwargs['fusion_epochs'], 30)
            self.assertTrue(trained.call_args.kwargs['resume'])
            self.assertEqual(analyzed.call_args.args[2], root/'feature_analysis')
            self.assertTrue(result['parent_checkpoint_unchanged'])
            self.assertFalse(result['audit_partition_evaluated'])

    def test_training_failure_still_runs_analysis_and_records_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            analyzed = Mock(return_value=dict(status='complete'))
            with self.assertRaisesRegex(RuntimeError, 'contains failures'):
                self.run_stages(root, training=Mock(side_effect=RuntimeError('fixture')), analysis=analyzed)
            analyzed.assert_called_once()
            result = json.loads((root/'completion.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['outcomes']['training']['status'], 'failed')

    def test_resume_rejects_edited_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'suite.json').write_text(json.dumps(dict(protocol=runner.PROTOCOL, source_hashes={'old': 'hash'})))
            with patch.object(runner, 'source_hashes', return_value={'new': 'hash'}):
                with self.assertRaisesRegex(ValueError, 'source files'):
                    runner.validate_resume(root)


if __name__ == '__main__':
    unittest.main()
