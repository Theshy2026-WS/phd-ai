import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import vit_augreg_suite as runner


class Sized:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n


class AugRegSuiteTests(unittest.TestCase):
    def fixture(self, root, extra=()):
        source, parent = root/'source', root/'parent'
        source.mkdir(exist_ok=True); parent.mkdir(exist_ok=True)
        original = parent/'best.pt'; original.write_bytes(b'parent')
        config = dict(dataset=dict(data_dir='data', download=False, batch_size=64, num_workers=4),
            augmentation=dict(crop_scale=[.6, 1.], randaugment_ops=2, randaugment_magnitude=5),
            training=dict(epochs=100, learning_rate=.0005, weight_decay=.05, seed=42))
        supplied = dict(source_suite=str(source), parent_run=str(parent), config=config, seed=42,
            split_hash='split', source_files={str(original):runner.file_hash(original)},
            provenance=dict(source_execution_environment={}))
        args = runner.arguments().parse_args(['--source-suite', str(source), '--output', str(root/'new'), *extra])
        return args, supplied

    def plan(self, args, supplied):
        with patch.object(runner, 'load_sources', return_value=supplied), \
                patch.object(runner, 'clean_datasets', return_value=(Sized(80000), Sized(10000), [], 'split',
                    dict(train_clean=['train/a'], selection=['selection/b']))), \
                patch.object(runner, 'execution_environment', return_value={}), \
                patch.object(runner, 'source_hashes', return_value={'new':'source'}):
            return runner.make_plan(args)

    def test_independent_search_and_fusion_keep_distinct_recipes(self):
        with tempfile.TemporaryDirectory() as temp:
            args, supplied = self.fixture(Path(temp))
            original = copy.deepcopy(supplied['config'])
            plan = self.plan(args, supplied)
            self.assertEqual(plan['total_new_model_epochs'], 1140)
            self.assertEqual(len(plan['jobs']), 11)
            self.assertIn('cnn_frozen_head', plan['jobs'])
            self.assertIn('new_frozen_residual_linear', plan['jobs'])
            for recipe in plan['profile_recipes'].values():
                self.assertEqual(recipe['training']['accumulation_steps']*recipe['dataset']['batch_size'], 256)
                self.assertEqual(recipe['training']['cutmix_alpha'], 0)
                self.assertEqual(recipe['training']['scheduler']['warmup_epochs'], 20)
            expected = copy.deepcopy(original)
            expected['dataset']['data_dir'] = str((runner.ROOT/'data').resolve())
            self.assertEqual(plan['config'], expected)
            self.assertEqual(supplied['config'], original)
            self.assertFalse(plan['external_pretraining'])
            self.assertIsNone(plan['max_hours'])
            self.assertFalse(plan['audit_partition_evaluated'])
            suite = runner.create_suite(args, plan)
            self.assertEqual(json.loads((suite/'suite.json').read_text())['total_new_model_epochs'], 1140)

    def test_source_and_ancestors_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); args, supplied = self.fixture(root)
            for output in (root, root/'source', root/'source'/'nested', root/'parent'):
                args.output = str(output)
                with self.assertRaisesRegex(ValueError, 'separate'):
                    self.plan(args, supplied)

    def test_smoke_has_separate_short_horizon_and_profile_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            args, supplied = self.fixture(Path(temp), ['--smoke', '--profiles', 'vit_tiny_p4_aug'])
            plan = self.plan(args, supplied)
            self.assertEqual(plan['total_new_model_epochs'], 9)
            self.assertEqual((plan['vit_epochs'], plan['fusion_epochs']), (1,1))
            self.assertEqual(plan['profile_recipes']['vit_tiny_p4_aug']['training']['accumulation_steps'], 1)
            self.assertEqual(plan['profile_recipes']['vit_tiny_p4_aug']['training']['scheduler']['warmup_epochs'], 0)

    def test_different_environment_and_invalid_horizon_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, supplied = self.fixture(Path(temp))
            supplied['provenance']['source_execution_environment'] = {'device':'other'}
            with self.assertRaisesRegex(ValueError, 'original GPU'):
                self.plan(args, supplied)
            args.vit_epochs = 0
            with self.assertRaisesRegex(ValueError, 'ViT1..600'):
                self.plan(args, supplied)

    def test_resume_rejects_profile_recipe_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); args, supplied = self.fixture(root)
            plan = self.plan(args, supplied)
            plan['profile_recipes']['vit_small_p8_aug']['training']['mixup_alpha'] = .99
            (root/'suite.json').write_text(json.dumps(plan))
            with patch.object(runner, 'source_hashes', return_value={'new':'source'}):
                with self.assertRaisesRegex(ValueError, 'profile recipes changed'):
                    runner.validate_resume(root)

    def run_stages(self, root, training=None, analysis=None):
        source = root/'original.pt'; source.write_bytes(b'original')
        plan = dict(source_suite='source', config={}, vit_epochs=300, fusion_epochs=30,
            profiles=list(runner.DEFAULT_PROFILES), total_new_model_epochs=1140,
            examples=0, smoke=False, source_files={str(source):runner.file_hash(source)},
            excluded_training_routes=['shallow64'])
        trained = training or Mock(return_value=dict(status='complete'))
        analyzed = analysis or Mock(return_value=dict(status='complete'))
        with patch.object(runner, 'validate_resume', return_value=plan), \
                patch('vit_augreg_experiment.run_augreg_study', trained), \
                patch('vit_augreg_diagnostics.run_augreg_diagnostics', analyzed), \
                patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                patch.object(runner.torch.cuda, 'empty_cache'):
            result = runner.run_suite(root)
        return result, trained, analyzed

    def test_training_and_analysis_finish_before_suite_success(self):
        with tempfile.TemporaryDirectory() as temp:
            result, trained, analyzed = self.run_stages(Path(temp))
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(result['source_checkpoints_unchanged'])
            self.assertEqual(trained.call_args.args[0]['vit_epochs'], 300)
            self.assertEqual(analyzed.call_args.args[2], Path(temp)/'feature_analysis')

    def test_training_failure_still_generates_analysis_then_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            analysis = Mock(return_value=dict(status='complete'))
            with self.assertRaisesRegex(RuntimeError, 'contains failures'):
                self.run_stages(Path(temp), Mock(side_effect=RuntimeError('fixture')), analysis)
            analysis.assert_called_once()
            self.assertEqual(json.loads((Path(temp)/'completion.json').read_text())['status'], 'failed')


if __name__ == '__main__': unittest.main()
