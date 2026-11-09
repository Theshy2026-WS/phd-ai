import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch import nn

import best_pair_joint as runner


class Sized:
    def __init__(self, n): self.n = n
    def __len__(self): return self.n


class BestPairJointTests(unittest.TestCase):
    def fixture(self, root, extra=()):
        cnn, vit, parent = root/'cnn', root/'vit', root/'cnn'/'training'/'cnn_head_continue'
        parent.mkdir(parents=True); vit.mkdir()
        original = parent/'best.pt'; original.write_bytes(b'best-cnn')
        config = dict(dataset=dict(data_dir='data', download=False, batch_size=64, num_workers=4),
            augmentation=dict(crop_scale=[.6, 1.], randaugment_ops=2, randaugment_magnitude=5),
            training=dict(epochs=100, learning_rate=5e-4, weight_decay=.05, seed=42))
        selected = dict(profile='vit_small_p8_augreg', checkpoint=str(vit/'best.pt'),
            checkpoint_sha256='vit-sha', state=dict(model={}), best_epoch=273, best_source='raw')
        sources = dict(cnn_suite=str(cnn), vit_suite=str(vit), source_suite=str(vit),
            parent_run=str(parent), parent_model={}, parent_completed_epochs=213,
            selected=selected, config=config, seed=42, split_hash='split',
            source_files={str(original):runner.file_hash(original)},
            provenance=dict(parent_checkpoint_sha256='cnn-sha', source_execution_environment={}))
        args = runner.arguments().parse_args(['--cnn-suite', str(cnn), '--vit-suite', str(vit),
            '--output', str(root/'new'), *extra])
        return args, sources

    def plan(self, args, sources):
        with patch.object(runner, 'load_sources', return_value=sources), \
                patch.object(runner, 'clean_datasets', return_value=(Sized(80000), Sized(10000), [], 'split',
                    dict(train_clean=['train/a'], selection=['selection/b']))), \
                patch.object(runner, 'execution_environment', return_value={}), \
                patch.object(runner, 'source_hashes', return_value={'new':'sha'}):
            return runner.make_plan(args)

    def test_single_fresh_joint30_keeps_best_weights_and_original_view_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            args, sources = self.fixture(Path(temp))
            before = copy.deepcopy(sources['config'])
            plan = self.plan(args, sources)
            self.assertEqual(plan['jobs'], ['new_joint_concat'])
            self.assertEqual(plan['epochs'], 30)
            self.assertEqual(plan['fusion_epochs'], 30)
            self.assertEqual(plan['total_new_model_epochs'], 30)
            self.assertEqual(plan['inherited_cnn_best_training_epochs'], 213)
            self.assertEqual(plan['fusion_augmentation_epoch_offset'], 100)
            self.assertEqual(plan['fusion_policy']['new_augmentation_epochs'], [101,130])
            self.assertFalse(plan['shutdown_after_completion'])
            self.assertFalse(plan['audit_partition_evaluated'])
            self.assertEqual(sources['config'], before)
            self.assertNotIn('state', plan['selected_vit'])
            suite = runner.create_suite(args, plan)
            self.assertEqual(json.loads((suite/'suite.json').read_text())['epochs'], 30)

    def test_output_cannot_overlap_either_source_or_ancestor(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); args, sources = self.fixture(root)
            for target in (root, root/'cnn', root/'vit'/'nested', Path(sources['parent_run'])):
                args.output = str(target)
                with self.assertRaisesRegex(ValueError, 'separate'):
                    self.plan(args, sources)

    def test_smoke_is_one_separate_epoch_and_invalid_horizons_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, sources = self.fixture(Path(temp), ['--smoke'])
            plan = self.plan(args, sources)
            self.assertEqual(plan['epochs'], 1)
            self.assertEqual(plan['fusion_policy']['new_augmentation_epochs'], [101,101])
            args.smoke = False; args.epochs = 31
            with self.assertRaisesRegex(ValueError, 'Need1..30'):
                self.plan(args, sources)

    def test_source_identity_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, sources = self.fixture(Path(temp))
            plan = self.plan(args, sources)
            runner._validate_source_identity(plan, sources)
            changed = copy.deepcopy(sources)
            changed['selected']['checkpoint_sha256'] = 'other'
            with self.assertRaisesRegex(ValueError, 'sources'):
                runner._validate_source_identity(plan, changed)

    def test_resume_rejects_shutdown_or_view_policy_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); args, sources = self.fixture(root)
            plan = self.plan(args, sources)
            suite = root/'suite'; suite.mkdir()
            for key, value, message in [('shutdown_after_completion',True,'policy'),
                                        ('fusion_augmentation_epoch_offset',213,'augmentation')]:
                changed = dict(plan, **{key:value})
                (suite/'suite.json').write_text(json.dumps(changed))
                with patch.object(runner, 'source_hashes', return_value={'new':'sha'}):
                    with self.assertRaisesRegex(ValueError, message):
                        runner.validate_resume(suite)

    def test_training_calls_exactly_one_new_job_without_changing_actual_cnn_history(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args,sources=self.fixture(root)
            plan=self.plan(args,sources)
            trained=Mock(return_value=dict(status='complete',completed_epochs=30))
            with patch.object(runner,'load_sources',return_value=sources), \
                    patch('vit_augreg_experiment._make_reference') as reference, \
                    patch('vit_augreg_experiment._train_job',trained), \
                    patch.object(runner,'training_report'):
                result=runner.run_training(plan,root/'training')
            self.assertEqual(result['status'],'complete')
            self.assertEqual(trained.call_count,1)
            self.assertEqual(trained.call_args.args[1],'new_joint_concat')
            self.assertEqual(trained.call_args.args[3]['parent_completed_epochs'],100)
            self.assertEqual(sources['parent_completed_epochs'],213)
            self.assertIs(trained.call_args.args[4],sources['selected'])
            self.assertEqual(reference.call_args.args[0]['parent_checkpoint_sha256'],'cnn-sha')
            changed=dict(plan,fusion_epochs=29)
            with patch.object(runner,'load_sources',return_value=sources):
                with self.assertRaisesRegex(ValueError,'recorded single'):
                    runner.run_training(changed,root/'training')

    def test_analysis_loader_does_not_select_or_retrain_vit(self):
        import vit_augreg_experiment as training
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args,sources=self.fixture(root)
            plan=self.plan(args,sources)
            run=root/'training'/runner.JOB;run.mkdir(parents=True)
            model=nn.Linear(2,2)
            config=training.job_config(plan,runner.JOB)
            metadata=dict(training._failure_identity(plan,runner.JOB),
                parent_checkpoint_sha256='cnn-sha',vit_checkpoint_sha256='vit-sha')
            state=dict(model=model.state_dict(),epoch=2,metadata=metadata,config=config)
            torch.save(state,run/'best.pt');(run/'metadata.json').write_text(json.dumps(metadata))
            with patch.object(training,'build_job_model',side_effect=lambda *a:nn.Linear(2,2)), \
                    patch.object(training,'_load_selected',side_effect=AssertionError('must not select')):
                loaded,record=runner.load_best_pair_model(plan,root/'training',sources=sources)
            self.assertEqual(record['epoch'],2)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,loaded.state_dict()[key])

    def test_failed_training_still_runs_analysis_and_never_changes_no_stop_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args,sources=self.fixture(root)
            plan=self.plan(args,sources)
            analyze=Mock(return_value=dict(status='unavailable'))
            with patch.object(runner,'validate_resume',return_value=plan), \
                    patch.object(runner,'run_training',side_effect=RuntimeError('fixture failure')), \
                    patch('best_pair_diagnostics.run_best_pair_diagnostics',analyze), \
                    patch.object(torch.cuda,'is_available',return_value=True), \
                    patch.object(torch.cuda,'empty_cache'):
                with self.assertRaisesRegex(RuntimeError,'incomplete work'):
                    runner.run_suite(root)
            analyze.assert_called_once()
            done=json.loads((root/'completion.json').read_text())
            self.assertEqual(done['status'],'failed')
            self.assertFalse(done['shutdown_after_completion'])


if __name__=='__main__': unittest.main()
