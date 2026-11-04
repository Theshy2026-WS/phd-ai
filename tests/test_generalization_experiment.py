import copy
import json
from pathlib import Path
import random
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import generalization_experiment as runner
from models.generalization_preact import (GeneralizationCNN, build_generalization_model,
                                         description, initialization_family)
from models.spatial_preact import build_spatial_model
from models.attention_correction import state_digest
from test_baseline_diagnostics import config as original_config, fixture_datasets, toy_model


class OptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_parameter_matching_init_and_phase_loss_one_bn_update(self):
        parent = build_spatial_model('cnn8', 42).eval()
        control = build_generalization_model('warm_control', parent.state_dict()).eval()
        dense = build_generalization_model('warm_full8', parent.state_dict()).eval()
        self.assertEqual(state_digest(control), state_digest(parent))
        self.assertEqual(sum(p.numel() for p in dense.parameters()), sum(p.numel() for p in parent.parameters()))
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            torch.testing.assert_close(parent(x), control(x), atol=0, rtol=0)
        a = build_generalization_model('fresh_control')
        b = build_generalization_model('fresh_phasebalanced')
        c = build_generalization_model('fresh_sam')
        self.assertEqual(state_digest(a), state_digest(b))
        self.assertEqual(state_digest(a), state_digest(c))
        b.train()
        count = int(b.head.bn.num_batches_tracked)
        targets = nn.functional.one_hot(torch.tensor([0, 199]), 200).float()
        loss = b.training_loss(x, targets)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(int(b.head.bn.num_batches_tracked), count+1)
        self.assertGreater(float(b.stem.weight.grad.abs().sum()), 0.)

    def small_bn_model(self):
        return nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 3)).train()

    def test_sam_uses_same_batch_updates_bn_once_and_changes_weights(self):
        model = self.small_bn_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        x = torch.randn(4, 3, 8, 8)
        target = nn.functional.one_hot(torch.tensor([0, 1, 2, 0]), 3).float()
        original = model[0].weight.detach().clone()
        first, second = runner.sam_step(model, x, target, optimizer)
        self.assertTrue(np.isfinite([first, second]).all())
        self.assertEqual(int(model[1].num_batches_tracked), 1)
        self.assertTrue(model[1].track_running_stats)
        self.assertFalse(torch.equal(original, model[0].weight))

    def test_sam_restores_exact_weights_and_bn_tracking_after_second_pass_failure(self):
        model = self.small_bn_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        calls = []
        original_loss = runner._loss
        def fail_on_second(*args):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('perturbed evaluation failed')
            return original_loss(*args)
        with patch.object(runner, '_loss', side_effect=fail_on_second):
            with self.assertRaisesRegex(RuntimeError, 'perturbed evaluation failed'):
                runner.sam_step(model, torch.randn(4, 3, 8, 8),
                    nn.functional.one_hot(torch.tensor([0, 1, 2, 0]), 3).float(), optimizer)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], atol=0, rtol=0)
        self.assertTrue(model[1].track_running_stats)
        self.assertEqual(int(model[1].num_batches_tracked), 1)

    def test_ema_updates_float_parameters_buffers_and_copies_integer_counter(self):
        model = self.small_bn_model()
        ema = copy.deepcopy(model).requires_grad_(False)
        with torch.no_grad():
            model[0].weight.add_(2.)
            model[1].running_mean.add_(4.)
            model[1].num_batches_tracked.fill_(5)
        old = ema[0].weight.detach().clone()
        runner.update_ema(ema, model, .5)
        torch.testing.assert_close(ema[0].weight, old+1)
        torch.testing.assert_close(ema[1].running_mean, torch.full((4,), 2.))
        self.assertEqual(int(ema[1].num_batches_tracked), 5)

    def test_job_recipes_and_precision_control(self):
        base = original_config()
        base['training']['mixed_precision'] = True
        original = copy.deepcopy(base)
        control = runner.job_config(base, 'warm_control', 30)
        gentle = runner.job_config(base, 'warm_gentle', 30)
        sam = runner.job_config(base, 'warm_sam', 30)
        fp32 = runner.job_config(base, 'warm_fp32_control', 30)
        fresh = runner.job_config(base, 'fresh_control', 100)
        self.assertEqual(base, original)
        self.assertEqual(control['training']['learning_rate'], 1e-4)
        self.assertEqual(fresh['training']['learning_rate'], 5e-4)
        self.assertEqual(fresh['training']['epochs'], 100)
        self.assertEqual(gentle['augmentation']['crop_scale'], [.85, 1.])
        self.assertEqual(gentle['training']['cutmix_alpha'], 0.)
        self.assertEqual(sam, fp32)
        self.assertFalse(sam['training']['mixed_precision'])

    def test_expanded_individual_factor_recipes_and_matched_fresh_sam(self):
        base = original_config()
        base['training']['mixed_precision'] = True
        control = runner.job_config(base, 'fresh_control', 100)
        no_cutmix = runner.job_config(base, 'fresh_no_cutmix', 100)
        self.assertEqual(no_cutmix['training']['cutmix_alpha'], 0.)
        no_cutmix['training']['cutmix_alpha'] = control['training']['cutmix_alpha']
        self.assertEqual(no_cutmix, control)
        no_ra = runner.job_config(base, 'fresh_no_randaugment', 100)
        self.assertEqual(no_ra['augmentation']['randaugment_ops'], 0)
        no_ra['augmentation']['randaugment_ops'] = control['augmentation']['randaugment_ops']
        self.assertEqual(no_ra, control)
        self.assertEqual(runner.job_config(base, 'fresh_sam', 100),
                         runner.job_config(base, 'fresh_fp32_control', 100))
        self.assertEqual(runner.job_config(base, 'fresh_wd_low', 100)['training']['weight_decay'], .01)
        self.assertEqual(runner.job_config(base, 'fresh_wd_high', 100)['training']['weight_decay'], .1)
        sgd = runner.job_config(base, 'fresh_sgd', 100)
        self.assertEqual(sgd['training']['learning_rate'], .1)
        self.assertEqual(sgd['training']['weight_decay'], 5e-4)
        self.assertEqual(len(runner.DEFAULT_JOBS), 19)

    def test_depth_and_groupnorm_controls_declare_their_different_state_layouts(self):
        base = build_generalization_model('fresh_control').eval()
        shallow = build_generalization_model('fresh_shallow26').eval()
        grouped = build_generalization_model('fresh_groupnorm').eval()
        self.assertEqual([len(stage) for stage in shallow.stages], [2, 2, 2, 2])
        self.assertLess(sum(p.numel() for p in shallow.parameters()), sum(p.numel() for p in base.parameters()))
        self.assertEqual(sum(p.numel() for p in grouped.parameters()), sum(p.numel() for p in base.parameters()))
        self.assertFalse(any(isinstance(m, nn.BatchNorm2d) for m in grouped.modules()))
        self.assertTrue(any(isinstance(m, nn.GroupNorm) for m in grouped.modules()))
        torch.testing.assert_close(base.stem.weight, grouped.stem.weight, atol=0, rtol=0)
        torch.testing.assert_close(base.stages[2][0].conv2[2].weight,
                                   shallow.stages[2][0].conv2[2].weight, atol=0, rtol=0)
        x = torch.randn(2, 3, 64, 64)
        for model in (shallow, grouped):
            self.assertEqual(model(x).shape, (2, 200))
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertNotEqual(initialization_family('fresh_groupnorm'), initialization_family('fresh_control'))
        self.assertFalse(description('fresh_shallow26')['parameter_matched'])
        self.assertEqual(description('fresh_groupnorm')['attention_blocks'], 0)

    def test_fresh_dense_readout_is_full8_with_same_initial_weights(self):
        control = build_generalization_model('fresh_control')
        dense = build_generalization_model('fresh_full8')
        self.assertEqual(state_digest(control), state_digest(dense))
        self.assertEqual(dense.readout, 'full8')
        self.assertIsInstance(dense.final_reduction, nn.Identity)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def supplied(self, root):
        values, manifest, split = fixture_datasets(root)
        from datasets.utility_tiny import UtilityDataset
        base = values['train']
        values['train_probe'] = UtilityDataset(base.folder, [0, 4, 8], 'train_probe', 42)
        return values, manifest, split

    def plan(self):
        config = original_config()
        config['training'].update(mixed_precision=False, gradient_clip_norm=5.)
        config['dataset'].update(batch_size=2, num_workers=0)
        return dict(config=config, epochs=3, fresh_epochs=4, seed=42,
            source_hashes=runner.source_hashes(), parent_run='parent',
            parent_checkpoint_sha256='same-parent', parent_completed_epochs=100,
            split_hash='fixture-split', examples=0, estimated_epoch_seconds=.001)

    def run_fake(self, root, *, interrupted=False, race=False, deadline=None, variant='warm_control'):
        root.mkdir(parents=True, exist_ok=True)
        supplied = self.supplied(root)
        plan = self.plan()
        state = dict(epoch=0, interrupt=interrupted, race=race)
        data_epochs = []
        def factory(*args):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                return toy_model()
        def train(model, ema, batches, config, seed, epoch, device, amp, dtype, optimizer, scaler, **kwargs):
            additional = epoch-100 if variant.startswith('warm_') else epoch
            if additional == 2 and state['interrupt']:
                state['interrupt'] = False
                raise KeyboardInterrupt()
            state['epoch'] = additional
            data_epochs.append(epoch)
            optimizer.zero_grad()
            coefficient = random.random()+np.random.rand()
            loss = sum((p*torch.randn_like(p)*coefficient).sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            runner.update_ema(ema, model)
            return dict(train_soft_target_loss=1., sam_perturbed_loss=None, skipped_amp_steps=0)
        def evaluate(model, batches, *args):
            if batches.split == 'train_probe':
                random.random(); np.random.rand(); torch.rand(3)
                return dict(accuracy=1., nll=.01, top5_accuracy=1.)
            return dict(accuracy={0:.7, 1:.8, 2:.9, 3:.6, 4:.6}[state['epoch']], nll=.5, top5_accuracy=1.)
        original_save = runner.atomic_save
        def save(path, value):
            if Path(path).name == 'last.pt' and value['epoch'] == 2 and state['race']:
                state['race'] = False
                raise KeyboardInterrupt()
            return original_save(path, value)
        with patch.object(runner, 'prepare_datasets', return_value=supplied), \
             patch.object(runner, 'build_generalization_model', side_effect=factory), \
             patch.object(runner, 'loader', side_effect=lambda data,*args:data), \
             patch.object(runner, 'train_one_epoch', side_effect=train), \
             patch.object(runner, 'evaluate', side_effect=evaluate), \
             patch('spatial_experiment.evaluate', side_effect=evaluate), \
             patch('spatial_experiment.loader', side_effect=lambda data,*args:data), \
             patch.object(runner, 'atomic_save', side_effect=save):
            if interrupted or race:
                with self.assertRaises(KeyboardInterrupt):
                    runner._train_job(plan, variant, root/'run', {}, deadline)
            result = runner._train_job(plan, variant, root/'run', {}, deadline)
        last = torch.load(root/'run/last.pt', weights_only=False)
        self.assertEqual(supplied[0]['audit'].reads, 0)
        self.assertEqual(supplied[0]['official_validation'].reads, 0)
        return result, last, data_epochs, root/'run', plan

    def test_last_and_best_are_distinct_and_fresh_horizon_is_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary, last, epochs, run, _ = self.run_fake(root/'warm')
            self.assertEqual(summary['best_epoch'], 2)
            self.assertEqual(summary['completed_epochs'], 3)
            self.assertEqual(epochs, [101, 102, 103])
            self.assertNotEqual(summary['last_diagnostic_directory'], summary['diagnostic_directory'])
            for key in ('last_diagnostic_directory', 'diagnostic_directory'):
                self.assertTrue((run/summary[key]/'selection/predictions.npz').exists())
            fresh, _, epochs, _, _ = self.run_fake(root/'fresh', variant='fresh_control')
            self.assertEqual(fresh['completed_epochs'], 4)
            self.assertEqual(epochs, [1, 2, 3, 4])

    def test_resume_and_best_write_race_reproduce_final_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, full, _, _, _ = self.run_fake(root/'full')
            resumed_result, resumed, _, _, _ = self.run_fake(root/'resumed', interrupted=True)
            race_result, race, _, _, _ = self.run_fake(root/'race', race=True)
            self.assertEqual(resumed_result['best_epoch'], 2)
            self.assertEqual(race_result['best_epoch'], 2)
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
                torch.testing.assert_close(full['model'][key], race['model'][key], rtol=0, atol=0)
                torch.testing.assert_close(full['ema_model'][key], race['ema_model'][key], rtol=0, atol=0)

    def test_deadline_retains_epoch_zero_and_marks_partial(self):
        with tempfile.TemporaryDirectory() as temp:
            result, state, epochs, _, _ = self.run_fake(Path(temp), deadline=time.monotonic()-1)
            self.assertEqual(result['status'], 'budget_exhausted')
            self.assertEqual(result['completed_epochs'], 0)
            self.assertEqual(epochs, [])
            self.assertEqual(state['epoch'], 0)

    def test_raw_ema_checkpoint_selection_ties_keep_earlier_raw(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            config = runner.job_config(original_config(), 'warm_control', 3)
            meta = dict(known='metadata')
            history = [dict(epoch=0, selection_accuracy=.7, ema_selection_accuracy=.7),
                       dict(epoch=1, selection_accuracy=.8, ema_selection_accuracy=.8),
                       dict(epoch=2, selection_accuracy=.8, ema_selection_accuracy=.8)]
            runner._save_best(run, dict(epoch=1, source='raw', selection=dict(accuracy=.8),
                                       metadata=meta, config=config, model={}))
            self.assertEqual(runner._recover_best(run, history, meta, config), (.8,1,'raw'))

    def test_api_epoch_bounds_and_known_jobs(self):
        for kw in (dict(epochs=31), dict(fresh_epochs=101), dict(jobs=['unknown']), dict(jobs=[])):
            with self.assertRaises(ValueError):
                runner.run_training_study('not-needed', 'not-needed', **kw)

    def test_study_continues_other_arms_after_failure_and_reports_nonzero(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root/'parent'; parent.mkdir()
            loaded = (parent, dict(seed=42, split_hash='fixed', completed_epochs=100),
                      dict(config=original_config(), model={}), 'parent-hash')
            calls = []
            def job(plan, variant, path, parent_state, deadline):
                calls.append(variant)
                if variant == 'warm_control':
                    raise RuntimeError('one arm failed')
                return dict(status='complete', completed_epochs=30, initial_state_sha256='same')
            with patch.object(runner, 'load_spatial_parent', return_value=loaded), \
                 patch.object(runner, 'clean_datasets', return_value=(None,None,None,'fixed',None)), \
                 patch.object(runner, 'source_hashes', return_value={}), \
                 patch.object(runner, 'execution_environment', return_value={}), \
                 patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                 patch.object(runner.torch.cuda, 'empty_cache'), \
                 patch.object(runner, '_make_reference'), \
                 patch.object(runner, '_train_job', side_effect=job), \
                 patch.object(runner, 'comparison'), \
                 patch.object(runner, 'file_hash', return_value='parent-hash'):
                with self.assertRaisesRegex(RuntimeError, 'arms failed'):
                    runner.run_training_study(parent, root/'study', jobs=['warm_control','warm_full8'])
            self.assertEqual(calls, ['warm_control','warm_full8'])
            status = json.loads((root/'study/completion.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['jobs']['warm_full8']['status'], 'complete')

    def test_study_budget_partial_is_explicit_and_skips_remaining_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root/'parent'; parent.mkdir()
            loaded = (parent, dict(seed=42, split_hash='fixed', completed_epochs=100),
                      dict(config=original_config(), model={}), 'parent-hash')
            result = dict(status='budget_exhausted', completed_epochs=12, initial_state_sha256='same')
            with patch.object(runner, 'load_spatial_parent', return_value=loaded), \
                 patch.object(runner, 'clean_datasets', return_value=(None,None,None,'fixed',None)), \
                 patch.object(runner, 'source_hashes', return_value={}), \
                 patch.object(runner, 'execution_environment', return_value={}), \
                 patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                 patch.object(runner.torch.cuda, 'empty_cache'), \
                 patch.object(runner, '_make_reference'), \
                 patch.object(runner, '_train_job', return_value=result) as train, \
                 patch.object(runner, 'comparison'), \
                 patch.object(runner, 'file_hash', return_value='parent-hash'):
                report = runner.run_training_study(parent, root/'study', jobs=['warm_control','warm_full8'])
            self.assertEqual(train.call_count, 1)
            self.assertEqual(report['status'], 'budget_exhausted')
            self.assertEqual(report['skipped_jobs'], ['warm_full8'])


if __name__ == '__main__':
    unittest.main()
