import copy
from contextlib import contextmanager
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import vit_followup_experiment as runner
import vit_followup_diagnostics as diagnostics
from test_baseline_diagnostics import config as original_config, fixture_datasets, toy_model


class ToyFollowup(nn.Module):
    def __init__(self, variant='cross16_mlp'):
        super().__init__()
        self.variant = variant
        self.analysis_mode = 'learned'
        self.encoder = nn.Identity() if variant == 'vit' else toy_model()
        if variant in ('vit', 'dual_fusion'):
            self.vit = toy_model()
            self.class_token = nn.Parameter(torch.zeros(1, 1, 3))
            self.position_embedding = nn.Parameter(torch.zeros(1, 2, 3))
        if variant != 'vit':
            self.fusion = nn.Linear(6 if variant == 'dual_fusion' else 3, 3)
            self.gamma = nn.Parameter(torch.tensor(.1))
        self.attention_flops_per_image = 64

    def forward(self, images):
        if self.variant == 'vit':
            return self.vit(images)+self.class_token.flatten()+self.position_embedding.mean((0, 1))
        cnn = self.encoder(images)
        if self.variant == 'dual_fusion':
            vit = self.vit(images)+self.class_token.flatten()+self.position_embedding.mean((0, 1))
            if self.analysis_mode == 'cnn_only':
                vit = torch.zeros_like(vit)
            elif self.analysis_mode == 'vit_only':
                cnn = torch.zeros_like(cnn)
            return cnn+self.gamma*self.fusion(torch.cat((cnn, vit), 1))
        if self.analysis_mode == 'removed':
            return cnn
        return cnn+self.gamma*self.fusion(images.mean((-2, -1)))

    @contextmanager
    def analysis(self, mode):
        previous = self.analysis_mode
        self.analysis_mode = mode
        try:
            yield self
        finally:
            self.analysis_mode = previous

    def forward_decomposed(self, images):
        logits = self(images)
        before = self.vit(images) if self.variant == 'vit' else self.encoder(images)
        return dict(logits=logits, features_before=before, features_after=logits,
                    gamma=getattr(self, 'gamma', torch.tensor(0.)), retrieval_maps={})


def factory(variant, parent_state=None, vit_state=None, seed=42):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return ToyFollowup(variant)


class OptimizerTests(unittest.TestCase):
    def test_all_independent_parameters_have_correct_roles_and_schedule(self):
        model = ToyFollowup('dual_fusion')
        config = runner.job_config(original_config(), 'dual_fusion', 30)
        optimizer = runner.make_optimizer(model, config)
        schedule = runner.make_scheduler(optimizer, config)
        seen, roles = set(), set()
        no_decay_ids = {id(model.class_token), id(model.position_embedding)}
        for group in optimizer.param_groups:
            roles.add(group['role'])
            for p in group['params']:
                self.assertNotIn(id(p), seen)
                seen.add(id(p))
                self.assertEqual(group['role'], runner.parameter_role(model, p))
                if id(p) in no_decay_ids or p.ndim <= 1:
                    self.assertEqual(group['weight_decay'], 0.)
        self.assertEqual(roles, {'encoder', 'vit', 'fusion'})
        self.assertEqual(seen, {id(p) for p in model.parameters()})
        for _ in range(31):
            rates = runner._rates(optimizer)
            self.assertEqual(rates['encoder_lr'], rates['vit_lr'])
            self.assertAlmostEqual(rates['fusion_lr']/rates['encoder_lr'], 10.)
            optimizer.step()
            schedule.step()

    def test_standalone_has_only_vit_role_and_no_recipe_mutation(self):
        base = original_config()
        before = copy.deepcopy(base)
        config = runner.job_config(base, 'vit_pretrain', 100)
        model = ToyFollowup('vit')
        optimizer = runner.make_optimizer(model, config)
        self.assertEqual({g['role'] for g in optimizer.param_groups}, {'vit'})
        self.assertEqual(base, before)
        self.assertEqual(config['augmentation'], base['augmentation'])
        self.assertEqual(config['training']['early_stopping_patience'], 0)
        self.assertEqual(config['training']['vit_learning_rate'], 5e-4)
        self.assertEqual(config['training']['scheduler']['warmup_epochs'], 2)

    def test_exact_user_queue_and_invalid_horizons(self):
        self.assertEqual(runner.DEFAULT_JOBS,
            ('vit_pretrain', 'dual_fusion', 'shallow64_attention', 'cross16_mlp'))
        self.assertEqual(runner.split_job('shallow64_attention'), ('joint', 'shallow64'))
        for job in ('warm_cross16', 'vit', 'unknown'):
            with self.assertRaises(ValueError):
                runner.split_job(job)
        for kwargs in (dict(vit_epochs=101), dict(fusion_epochs=31), dict(jobs=[]),
                       dict(jobs=['dual_fusion']), dict(jobs=['dual_fusion', 'vit_pretrain'])):
            with self.assertRaises(ValueError):
                runner.run_followup_study('unused', 'unused', **kwargs)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def plan(self):
        config = original_config()
        config['dataset'].update(batch_size=2, num_workers=0)
        config['training'].update(mixed_precision=False, gradient_clip_norm=5.)
        return dict(config=config, vit_epochs=4, fusion_epochs=3, seed=42,
            source_hashes=runner.source_hashes(), parent_run='original-cnn8',
            parent_checkpoint_sha256='parent-hash', parent_completed_epochs=100,
            split_hash='fixture-split', examples=0)

    def supplied(self, root):
        supplied, manifest, fingerprint = fixture_datasets(root)
        from datasets.utility_tiny import UtilityDataset
        supplied['train_probe'] = UtilityDataset(supplied['train'].folder, [0, 4, 8], 'train_probe', 42)
        return supplied, manifest, fingerprint

    def fake_run(self, root, job='cross16_mlp', interrupted=False, race=False, improve=True,
                 prior_failure=None):
        root.mkdir(parents=True, exist_ok=True)
        plan, supplied = self.plan(), self.supplied(root)
        progress = dict(epoch=0, interrupt=interrupted, race=race)
        epochs = []
        if prior_failure in ('only', 'foreign', 'stray'):
            self.assertTrue(runner._record_failure(plan, job, root/'run',
                dict(error_type='RuntimeError', traceback='synthetic early failure')))
            if prior_failure == 'foreign':
                path = root/'run/failure.json'
                saved = json.loads(path.read_text())
                saved['identity']['source_hashes'] = {'another': 'source'}
                path.write_text(json.dumps(saved), encoding='utf-8')
            if prior_failure == 'stray':
                (root/'run/notes.txt').write_text('unrelated user file', encoding='utf-8')

        def train(model, ema, batches, config, seed, epoch, device, amp, dtype, optimizer, scaler):
            local = epoch if job == 'vit_pretrain' else epoch-100
            if local == 2 and progress['interrupt']:
                progress['interrupt'] = False
                raise KeyboardInterrupt()
            progress['epoch'] = local
            epochs.append(epoch)
            optimizer.zero_grad()
            coefficient = random.random()+np.random.rand()
            loss = sum((p*torch.randn_like(p)*coefficient).sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            runner.update_ema(ema, model)
            return dict(train_soft_target_loss=1., skipped_amp_steps=0)

        def evaluate(model, batches, *args):
            if batches.split == 'train_probe':
                random.random(); np.random.rand(); torch.rand(3)
                return dict(accuracy=1., top5_accuracy=1., nll=.01)
            scores = {0:.7, 1:.8, 2:.9, 3:.6, 4:.6} if improve else {0:.7, 1:.6, 2:.5, 3:.4, 4:.3}
            return dict(accuracy=scores[progress['epoch']], top5_accuracy=1., nll=.5)

        real_save = runner.atomic_save

        def save(path, value):
            if Path(path).name == 'last.pt' and value['epoch'] == 2 and progress['race']:
                progress['race'] = False
                raise KeyboardInterrupt()
            return real_save(path, value)

        with patch.object(runner, 'prepare_datasets', return_value=supplied), \
             patch.object(runner, 'build_followup_model', side_effect=factory), \
             patch.object(runner, 'loader', side_effect=lambda ds, *args: ds), \
             patch.object(runner, 'train_one_epoch', side_effect=train), \
             patch('generalization_experiment.evaluate', side_effect=evaluate), \
             patch('generalization_experiment.loader', side_effect=lambda ds, *args: ds), \
             patch('spatial_experiment.evaluate', side_effect=evaluate), \
             patch('spatial_experiment.loader', side_effect=lambda ds, *args: ds), \
             patch.object(runner, 'estimate_flops', return_value=1000), \
             patch.object(runner, 'atomic_save', side_effect=save):
            if prior_failure == 'metadata_before_last':
                with patch.object(runner, '_isolated_evaluate', side_effect=RuntimeError('initial evaluation failed')):
                    with self.assertRaisesRegex(RuntimeError, 'initial evaluation failed'):
                        runner._train_job(plan, job, root/'run', {})
                self.assertTrue((root/'run/metadata.json').is_file())
                self.assertFalse((root/'run/last.pt').exists())
                self.assertTrue(runner._record_failure(plan, job, root/'run',
                    dict(error_type='RuntimeError', traceback='synthetic initial evaluation failure')))
            if interrupted or race:
                with self.assertRaises(KeyboardInterrupt):
                    runner._train_job(plan, job, root/'run', {}, {}, 'vit-dependency' if job == 'dual_fusion' else None)
            summary = runner._train_job(plan, job, root/'run', {}, {}, 'vit-dependency' if job == 'dual_fusion' else None)
        last = torch.load(root/'run/last.pt', weights_only=False)
        self.assertEqual(supplied[0]['audit'].reads, 0)
        self.assertEqual(supplied[0]['official_validation'].reads, 0)
        return summary, last, epochs, root/'run', plan

    def test_complete_pretrain_and_joint_keep_best_last_and_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pre, _, epochs, _, _ = self.fake_run(root/'vit', job='vit_pretrain')
            self.assertEqual(epochs, [1, 2, 3, 4])
            self.assertEqual(pre['completed_epochs'], 4)
            joint, _, epochs, run, _ = self.fake_run(root/'dual', job='dual_fusion')
            self.assertEqual(epochs, [101, 102, 103])
            self.assertEqual(joint['vit_checkpoint_sha256'], 'vit-dependency')
            self.assertEqual(joint['best_epoch'], 2)
            self.assertEqual(joint['best_trained_selection']['epoch'], 2)
            self.assertEqual(joint['forward_flops_estimate'], 1064)
            self.assertNotEqual(joint['diagnostic_directory'], joint['last_diagnostic_directory'])
            self.assertTrue((run/joint['diagnostic_directory']/'selection/predictions.npz').is_file())

    def test_epoch0_never_reported_as_learned_improvement(self):
        with tempfile.TemporaryDirectory() as temp:
            result, _, _, run, _ = self.fake_run(Path(temp), improve=False)
            self.assertEqual(result['best_epoch'], 0)
            self.assertEqual(result['best_trained_selection']['epoch'], 1)
            self.assertEqual(result['last_epoch_selection']['accuracy'], .4)
            self.assertEqual(torch.load(run/'best_trained.pt', weights_only=False)['epoch'], 1)

    def test_early_owned_failure_directories_retry_and_foreign_files_remain_untouched(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for prior in ('only', 'metadata_before_last'):
                result, _, epochs, run, plan = self.fake_run(root/prior, prior_failure=prior)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(epochs, [101, 102, 103])
                self.assertFalse((run/'failure.json').exists())
                self.assertTrue(runner._record_failure(plan, 'cross16_mlp', run,
                    dict(error_type='RuntimeError', traceback='stale failure after completed summary')))
                supplied = self.supplied(root/(prior+'_stale_data'))
                with patch.object(runner, 'prepare_datasets', return_value=supplied), \
                     patch.object(runner, 'build_followup_model', side_effect=factory):
                    saved = runner._train_job(plan, 'cross16_mlp', run, {})
                self.assertEqual(saved, result)
                self.assertFalse((run/'failure.json').exists())
            for prior in ('foreign', 'stray'):
                with self.assertRaises(ValueError):
                    self.fake_run(root/prior, prior_failure=prior)
                run = root/prior/'run'
                self.assertTrue((run/'failure.json').exists())
                self.assertFalse((run/'metadata.json').exists())
                self.assertFalse((run/'last.pt').exists())
                if prior == 'foreign':
                    saved = json.loads((run/'failure.json').read_text())
                    self.assertEqual(saved['identity']['source_hashes'], {'another': 'source'})
                    self.assertFalse(runner._record_failure(self.plan(), 'cross16_mlp', run,
                        dict(error_type='ValueError', traceback='must not replace foreign record')))
                else:
                    self.assertEqual((run/'notes.txt').read_text(), 'unrelated user file')

    def test_interrupt_and_commit_race_resume_exactly(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, full, _, _, _ = self.fake_run(root/'full')
            _, resumed, _, _, _ = self.fake_run(root/'interrupted', interrupted=True)
            _, race, _, _, _ = self.fake_run(root/'race', race=True)
            for name in full['model']:
                torch.testing.assert_close(full['model'][name], resumed['model'][name], rtol=0, atol=0)
                torch.testing.assert_close(full['model'][name], race['model'][name], rtol=0, atol=0)
                torch.testing.assert_close(full['ema_model'][name], race['ema_model'][name], rtol=0, atol=0)
            self.assertEqual(full['history'][-1]['fusion_lr'], race['history'][-1]['fusion_lr'])

    def test_vit_dependency_requires_completed_source_matched_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, _, _, run, plan = self.fake_run(root/'fixture', job='vit_pretrain')
            run.rename(root/'vit_pretrain')
            state, digest = runner.load_vit_dependency(plan, root)
            self.assertEqual(set(state), set(torch.load(root/'vit_pretrain/best.pt', weights_only=False)['model']))
            self.assertEqual(digest, runner.file_hash(root/'vit_pretrain/best.pt'))
            for changes in (dict(vit_epochs=5), dict(source_hashes={'different':'source'}), dict(seed=43)):
                changed = dict(plan, **changes)
                with self.assertRaises(ValueError):
                    runner.load_vit_dependency(changed, root)

    def test_failure_of_vit_does_not_skip_independent_attention_jobs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root/'parent'; parent.mkdir()
            loaded = (parent, dict(seed=42, split_hash='split', completed_epochs=100),
                      dict(config=original_config(), model={}), 'parent-hash')
            calls = []

            def train(plan, job, run, state, vit_state, vit_hash):
                calls.append(job)
                if job == 'vit_pretrain':
                    raise RuntimeError('synthetic pretraining failure')
                return dict(status='complete', completed_epochs=30)

            with patch.object(runner, 'load_spatial_parent', return_value=loaded), \
                 patch.object(runner, 'clean_datasets', return_value=(None, None, None, 'split', None)), \
                 patch.object(runner, 'source_hashes', return_value={}), \
                 patch.object(runner, 'execution_environment', return_value={}), \
                 patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                 patch.object(runner.torch.cuda, 'empty_cache'), \
                 patch.object(runner, '_train_job', side_effect=train), \
                 patch.object(runner, '_make_reference'), \
                 patch.object(runner, 'comparison'), \
                 patch.object(runner, 'file_hash', return_value='parent-hash'):
                with self.assertRaisesRegex(RuntimeError, 'follow-up jobs failed'):
                    runner.run_followup_study(parent, root/'study')
            self.assertEqual(calls, ['vit_pretrain', 'shallow64_attention', 'cross16_mlp'])
            result = json.loads((root/'study/completion.json').read_text())
            self.assertEqual(result['status'], 'failed')
            self.assertIn('dual_fusion', result['failures'])
            failure = json.loads((root/'study/dual_fusion/failure.json').read_text())
            self.assertEqual(failure['protocol'], runner.PROTOCOL)
            self.assertEqual(failure['job'], 'dual_fusion')
            self.assertEqual(failure['identity']['source_hashes'], {})
            self.assertEqual(failure['identity']['parent_checkpoint_sha256'], 'parent-hash')

    def test_real_postdiagnostics_record_all_modes_and_reuse_identical_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result, _, _, run, plan = self.fake_run(root/'fixture', improve=False)
            training = root/'training'; training.mkdir()
            run.rename(training/'cross16_mlp')
            plan['jobs'] = ['cross16_mlp']
            runner.save_json(training/'suite.json', plan)
            supplied = self.supplied(root/'diagnostic_data')
            parent = root/'parent'; parent.mkdir()
            loaded = (parent, dict(seed=42, split_hash='fixture-split'),
                      dict(config=plan['config'], model={}), 'parent-hash')
            output = root/'analysis'
            with patch.object(diagnostics, 'load_spatial_parent', return_value=loaded), \
                 patch.object(diagnostics, 'clean_datasets', return_value=(supplied[0]['train_probe'],
                     supplied[0]['selection'], None, 'fixture-split', supplied[1])), \
                 patch.object(diagnostics, 'build_followup_model', side_effect=factory), \
                 patch.object(diagnostics, 'file_hash', side_effect=lambda p: 'parent-hash'
                     if Path(p) == parent/'best.pt' else runner.file_hash(p)):
                first = diagnostics.run_followup_diagnostics(parent, training, output, examples=0)
                second = diagnostics.run_followup_diagnostics(parent, training, output, examples=0)
            self.assertEqual(first, second)
            self.assertEqual(first['status'], 'complete')
            self.assertEqual(len(list(output.glob('attempt_*'))), 1)
            import pandas as pd
            rows = pd.read_csv(output/first['analysis_directory']/'comparison.csv')
            self.assertEqual(set(rows['checkpoint']), {'best', 'last_raw', 'best_trained'})
            self.assertEqual(set(rows['mode']), set(diagnostics.MODES['cross16_mlp']))
            self.assertEqual(len(rows), 3*2*4)
            self.assertEqual(len(first['comparisons']), 3*2*3)
            self.assertEqual(supplied[0]['audit'].reads, 0)
            self.assertEqual(supplied[0]['official_validation'].reads, 0)


class DiagnosticPolicyTests(unittest.TestCase):
    def test_modes_target_sources_and_mlp_separately(self):
        self.assertEqual(diagnostics.MODES['vit'], ('learned',))
        self.assertEqual(diagnostics.MODES['dual_fusion'], ('learned', 'cnn_only', 'vit_only'))
        self.assertEqual(diagnostics.MODES['shallow64'], ('learned', 'uniform', 'removed'))
        self.assertEqual(diagnostics.MODES['cross16_mlp'], ('learned', 'uniform', 'removed', 'no_mlp'))


if __name__ == '__main__':
    unittest.main()
