import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import vit_augreg_experiment as runner
from datasets.utility_tiny import UtilityDataset
from test_baseline_diagnostics import config as original_config, fixture_datasets


PROFILE = 'vit_tiny_p4_aug'


def base_config():
    config = original_config()
    config['dataset'].update(batch_size=2, num_workers=0)
    config['training'].update(mixed_precision=False, gradient_clip_norm=100.,
        accumulation_steps=4, learning_rate=5e-4, vit_learning_rate=5e-4,
        weight_decay=.03, mixup_alpha=.2, cutmix_alpha=0., mix_probability=1.,
        betas=[.9, .999], eps=1e-8)
    config['training']['scheduler'] = dict(warmup_epochs=1, min_lr=1e-6,
        warmup_start_factor=.01)
    return config


def plan():
    config = base_config()
    return dict(config=config, profile_recipes={PROFILE: copy.deepcopy(config)},
        profiles=[PROFILE], vit_epochs=3, fusion_epochs=2, seed=42,
        source_hashes={'source': 'fixture'}, source_files={},
        source_suite='old-suite', source_provenance={'parent_checkpoint_sha256': 'parent'},
        parent_run='original-cnn8', split_hash='fixture-split', examples=0, smoke=False,
        jobs=['pretrain_'+PROFILE]+list(runner.FUSION_JOBS))


class LinearClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(3, 3)
        self.class_token = nn.Parameter(torch.zeros(1, 1, 3))
        self.position_embedding = nn.Parameter(torch.zeros(1, 2, 3))
        self.attention_flops_per_image = 0

    def forward(self, images):
        features = images.mean((-2, -1)) if images.ndim == 4 else images
        return self.classifier(features)+self.class_token.flatten()+self.position_embedding.mean((0, 1))


def toy_factory(job, sources, selected=None, seed=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42 if seed is None else seed)
        return LinearClassifier()


class AugRegOptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_queue_and_config_leave_original_fusion_recipe_unchanged(self):
        supplied = plan()
        before = copy.deepcopy(supplied)
        pretrain = runner.job_config(supplied, 'pretrain_'+PROFILE)
        joint = runner.job_config(supplied, 'new_joint_concat')
        self.assertEqual(supplied, before)
        self.assertEqual(pretrain['training']['accumulation_steps'], 4)
        self.assertEqual(joint['training']['accumulation_steps'], 1)
        self.assertEqual(joint['augmentation'], supplied['config']['augmentation'])
        self.assertEqual(joint['training']['fusion_learning_rate'], 5e-4)
        self.assertEqual(joint['training']['vit_learning_rate'], 5e-5)
        self.assertEqual(joint['training']['encoder_learning_rate'], 5e-5)
        self.assertEqual(len(runner.FUSION_JOBS), 8)
        self.assertIn('new_frozen_residual_linear', runner.FUSION_JOBS)
        for unknown in ('shallow64', 'new_frozen_unknown', 'pretrain_unknown'):
            with self.assertRaises(ValueError):
                runner.split_job(unknown)

    def test_full_horizon_schedule_and_no_decay_position_cls(self):
        supplied = plan()
        supplied['vit_epochs'] = 300
        supplied['profile_recipes'][PROFILE]['training']['scheduler']['warmup_epochs'] = 20
        config = runner.job_config(supplied, 'pretrain_'+PROFILE)
        model = LinearClassifier()
        optimizer = runner.make_optimizer(model, config, standalone=True)
        seen = []
        for group in optimizer.param_groups:
            self.assertEqual(group['role'], 'vit')
            for parameter in group['params']:
                seen.append(id(parameter))
                if parameter is model.class_token or parameter is model.position_embedding or parameter.ndim < 2:
                    self.assertEqual(group['weight_decay'], 0.)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {id(p) for p in model.parameters()})
        scheduler = runner.make_scheduler(optimizer, config)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-6)
        for _ in range(20):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 5e-4)
        for _ in range(280):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 1e-6)

    def test_accumulation_matches_actual_large_batch_including_short_final_window(self):
        torch.manual_seed(17)
        model = LinearClassifier()
        reference = copy.deepcopy(model)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        config = base_config()
        config['training'].update(accumulation_steps=3, weight_decay=0.)
        optimizer = torch.optim.SGD(model.parameters(), lr=.03)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.03)
        images, labels = torch.randn(9, 3), torch.tensor([0, 1, 2]*3)
        batches = [(images[a:b], labels[a:b], ['x']*(b-a)) for a, b in [(0, 2), (2, 4), (4, 6), (6, 8), (8, 9)]]
        _, amp, dtype, scaler = runner.runtime(config, 42)
        def unchanged_mix(x, y, *unused):
            return x, F.one_hot(y, 3).float()
        with patch('resnet_experiment.mix_batch', side_effect=unchanged_mix):
            result = runner.train_one_epoch(model, ema, batches, config, 42, 1,
                'cpu', amp, dtype, optimizer, scaler)
        for a, b in [(0, 6), (6, 9)]:
            expected_optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(reference(images[a:b]), labels[a:b]).backward()
            expected_optimizer.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertEqual(result['optimizer_steps'], 2)
        self.assertEqual(result['skipped_amp_steps'], 0)

    def test_frozen_cnn_head_changes_without_feature_or_buffer_drift(self):
        from models.spatial_preact import build_spatial_model
        parent = build_spatial_model('cnn8', 42)
        model = runner.FrozenCNNHeadControl(parent.state_dict(), 42)
        before = runner.frozen_digest(model)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        config = runner.job_config(plan(), 'cnn_frozen_head')
        optimizer = runner.make_optimizer(model, config)
        self.assertEqual({g['role'] for g in optimizer.param_groups}, {'fusion'})
        self.assertEqual({id(p) for g in optimizer.param_groups for p in g['params']},
                         {id(p) for p in model.encoder.head.dense.parameters()})
        _, amp, dtype, scaler = runner.runtime(config, 42)
        previous_head = model.encoder.head.dense.weight.detach().clone()
        batches = [(torch.randn(2, 3, 64, 64), torch.tensor([0, 1]), ['a', 'b'])]
        runner.train_one_epoch(model, ema, batches, config, 42, 101, 'cpu', amp, dtype, optimizer, scaler)
        self.assertEqual(before, runner.frozen_digest(model))
        self.assertFalse(torch.equal(previous_head, model.encoder.head.dense.weight))
        for name, value in runner._frozen_state(model).items():
            torch.testing.assert_close(ema.state_dict()[name], value, rtol=0, atol=0)
        self.assertFalse(model.encoder.training)

    def test_single_batch_scales_mean_loss_without_batch_size_amplification(self):
        model = LinearClassifier()
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        config = base_config()
        config['training']['accumulation_steps'] = 1
        optimizer = torch.optim.SGD(model.parameters(), lr=0.)
        images, labels = torch.randn(7, 3), torch.arange(7) % 3
        expected = float(F.cross_entropy(model(images), labels).detach())
        _, amp, dtype, scaler = runner.runtime(config, 42)
        original_scale, seen = scaler.scale, []
        def scale(loss):
            seen.append(float(loss.detach()))
            return original_scale(loss)
        def mix(x, y, *unused):
            return x, F.one_hot(y, 3).float()
        with patch('resnet_experiment.mix_batch', side_effect=mix), patch.object(scaler, 'scale', side_effect=scale):
            runner.train_one_epoch(model, ema, [(images, labels, ['x']*7)], config,
                42, 1, 'cpu', amp, dtype, optimizer, scaler)
        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(seen[0], expected, places=6)


class AugRegTransactionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def supplied(self, root):
        supplied, manifest, fingerprint = fixture_datasets(root)
        supplied['train_probe'] = UtilityDataset(supplied['train'].folder, [0, 4, 8], 'train_probe', 42)
        return supplied, manifest, fingerprint

    def run_fixture(self, root, interrupt=False, race=False, improve=True):
        root.mkdir(parents=True, exist_ok=True)
        requested = plan()
        supplied = self.supplied(root)
        progress = dict(epoch=0, interrupt=interrupt, race=race)
        sources = dict(seed=42, parent_model={}, parent_completed_epochs=100,
                       provenance={'parent_checkpoint_sha256': 'parent'})
        def train(model, ema, batches, config, seed, epoch, device, amp, dtype, optimizer, scaler):
            progress['epoch'] = epoch
            if epoch == 2 and progress['interrupt']:
                progress['interrupt'] = False
                random.random(); np.random.rand(); torch.rand(1)
                raise RuntimeError('synthetic interruption')
            optimizer.zero_grad(set_to_none=True)
            loss = sum(p.square().sum() for p in model.parameters())
            loss = loss*(.5+random.random()+np.random.rand()+float(torch.rand(1)))
            loss.backward()
            optimizer.step()
            runner.update_ema(ema, model)
            return dict(train_soft_target_loss=float(loss.detach()), skipped_amp_steps=0, optimizer_steps=1)
        def evaluate(*args):
            score = .4+.1*progress['epoch'] if improve else .8-.1*progress['epoch']
            return dict(accuracy=score, nll=1-score, top5_accuracy=1.)
        def diagnose(model, supplied, config, seed, run, examples, label):
            target = run/label/'diagnostics'
            target.mkdir(parents=True, exist_ok=True)
            metrics = dict(accuracy=evaluate()['accuracy'], nll=evaluate()['nll'], top5_accuracy=1.)
            return target.relative_to(run).as_posix(), dict(train_clean=metrics, selection=metrics)
        original_save = runner._save_history
        def save_history(run, history):
            if history[-1]['epoch'] == 2 and progress['race']:
                progress['race'] = False
                raise RuntimeError('synthetic interruption after last checkpoint')
            return original_save(run, history)
        patches = [patch.object(runner, 'build_job_model', side_effect=toy_factory),
            patch.object(runner, 'prepare_datasets', return_value=supplied),
            patch.object(runner, 'train_one_epoch', side_effect=train),
            patch.object(runner, '_isolated_evaluate', side_effect=evaluate),
            patch.object(runner, '_probe', side_effect=evaluate),
            patch.object(runner, '_diagnose_at', side_effect=diagnose),
            patch.object(runner, '_save_history', side_effect=save_history),
            patch.object(runner, '_cost', return_value=dict(forward_flops_estimate=0))]
        from contextlib import ExitStack
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            if interrupt or race:
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    runner._train_job(requested, 'pretrain_'+PROFILE, root/'run', sources)
            result = runner._train_job(requested, 'pretrain_'+PROFILE, root/'run', sources)
        state = torch.load(root/'run/last.pt', map_location='cpu', weights_only=False)
        return result, state

    def test_full_resume_preserves_optimizer_ema_rng_and_horizon(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            full, a = self.run_fixture(root/'full')
            resumed, b = self.run_fixture(root/'resumed', interrupt=True)
            self.assertEqual(full['completed_epochs'], 3)
            self.assertEqual(resumed['completed_epochs'], 3)
            for name, value in a['model'].items():
                torch.testing.assert_close(value, b['model'][name], rtol=0, atol=0)
                torch.testing.assert_close(a['ema_model'][name], b['ema_model'][name], rtol=0, atol=0)
            for key in a['optimizer']['state']:
                for field, value in a['optimizer']['state'][key].items():
                    torch.testing.assert_close(value, b['optimizer']['state'][key][field], rtol=0, atol=0)
            self.assertEqual([x['epoch'] for x in b['history']], [0, 1, 2, 3])
            self.assertEqual(a['scheduler'], b['scheduler'])

    def test_resume_recovers_committed_best_after_report_write_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, state = self.run_fixture(Path(temporary), race=True)
            self.assertEqual(result['best_epoch'], 3)
            self.assertEqual(state['epoch'], 3)

    def test_epoch_zero_winner_is_separate_from_best_actually_trained(self):
        with tempfile.TemporaryDirectory() as temporary:
            result, state = self.run_fixture(Path(temporary), improve=False)
            self.assertEqual(result['best_epoch'], 0)
            self.assertEqual(result['best_trained_selection']['epoch'], 1)
            self.assertTrue((Path(temporary)/'run/best_trained.pt').is_file())

    def test_failure_only_recovery_and_foreign_directory_protection(self):
        with tempfile.TemporaryDirectory() as temporary:
            requested = plan()
            run = Path(temporary)/'owned'
            self.assertTrue(runner._record_failure(requested, 'pretrain_'+PROFILE, run, {'error_type': 'Synthetic'}))
            self.assertTrue(runner._owned_failure(requested, 'pretrain_'+PROFILE, run))
            changed = copy.deepcopy(requested)
            changed['vit_epochs'] = 4
            with self.assertRaisesRegex(ValueError, 'another experiment'):
                runner._owned_failure(changed, 'pretrain_'+PROFILE, run)
            foreign = Path(temporary)/'foreign'
            foreign.mkdir()
            (foreign/'user-file.txt').write_text('keep')
            self.assertFalse(runner._record_failure(requested, 'pretrain_'+PROFILE, foreign, {'error_type': 'Synthetic'}))
            self.assertEqual((foreign/'user-file.txt').read_text(), 'keep')

    def test_selection_requires_all_full_candidates_and_uses_nll_tie_break(self):
        from models.vit_augreg import PROFILES
        requested = plan()
        requested['profiles'] = list(PROFILES)[:2]
        requested['profile_recipes'] = {name: base_config() for name in requested['profiles']}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, profile in enumerate(requested['profiles']):
                job, run = 'pretrain_'+profile, root/('pretrain_'+profile)
                run.mkdir()
                metadata = runner._failure_identity(requested, job)
                score = dict(accuracy=.6, nll=1.-index*.1, top5_accuracy=.9)
                state = dict(model={}, epoch=2, source='ema', selection=score,
                             metadata=metadata, config=runner.job_config(requested, job))
                torch.save(state, run/'best.pt')
                summary = dict(**metadata, status='complete', completed_epochs=3,
                    best_epoch=2, best_source='ema', best_selection=score)
                runner.save_json(run/'summary.json', summary)
                if index == 0:
                    with self.assertRaises(FileNotFoundError):
                        runner.select_vit(requested, root)
            selected = runner.select_vit(requested, root)
            self.assertEqual(selected['profile'], requested['profiles'][1])
            self.assertTrue(selected['complete'])
            self.assertEqual(runner.select_vit(requested, root), selected)
            summary['completed_epochs'] = 2
            runner.save_json(run/'summary.json', summary)
            with self.assertRaisesRegex(ValueError, 'must complete'):
                runner.select_vit(requested, root)

    def test_study_selects_completed_vit_before_all_fusion_jobs_and_retains_partial_failures(self):
        from models.vit_augreg import PROFILES
        requested = plan()
        requested['profiles'] = list(PROFILES)[:2]
        requested['profile_recipes'] = {name: base_config() for name in requested['profiles']}
        requested['jobs'] = ['pretrain_'+p for p in requested['profiles']]+list(runner.FUSION_JOBS)
        sources = dict(seed=42, parent_model={}, parent_completed_epochs=100,
            source_suite='old-suite', parent_run='original-cnn8', split_hash='fixture-split',
            provenance=requested['source_provenance'], source_files={})
        seen = []
        def fake_train(plan, job, run, sources, selected=None):
            seen.append((job, None if selected is None else selected['profile']))
            run.mkdir(parents=True, exist_ok=True)
            metadata = runner._failure_identity(plan, job)
            scope, profile = runner.split_job(job)
            score = dict(accuracy=.6, nll=.9 if profile == plan['profiles'][-1] else 1., top5_accuracy=.9)
            state = dict(model={}, epoch=2, source='ema', selection=score,
                         metadata=metadata, config=runner.job_config(plan, job))
            runner.atomic_save(run/'best.pt', state)
            result = dict(**metadata, status='complete',
                completed_epochs=plan['vit_epochs'] if scope == 'pretrain' else plan['fusion_epochs'],
                best_epoch=2, best_source='ema', best_selection=score)
            runner.save_json(run/'summary.json', result)
            return result
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(runner, 'load_sources', return_value=sources), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(runner, '_make_reference'), patch.object(runner, 'comparison'), \
                 patch.object(runner, '_train_job', side_effect=fake_train):
                result = runner.run_augreg_study(requested, root/'complete', examples=0)
                self.assertEqual(result['status'], 'complete')
                self.assertEqual(len(seen), len(requested['jobs']))
                for job, selected in seen:
                    if job in runner.FUSION_JOBS and job != 'cnn_frozen_head':
                        self.assertEqual(selected, requested['profiles'][-1])
                self.assertTrue((root/'complete/selected_vit.json').is_file())
                changed = copy.deepcopy(requested)
                changed['fusion_epochs'] += 1
                with self.assertRaisesRegex(ValueError, 'Resume requires'):
                    runner.run_augreg_study(changed, root/'complete', examples=0)
            seen.clear()
            def fails_first(plan, job, run, sources, selected=None):
                if job == 'pretrain_'+plan['profiles'][0]:
                    raise RuntimeError('synthetic candidate failure')
                return fake_train(plan, job, run, sources, selected)
            with patch.object(runner, 'load_sources', return_value=sources), \
                 patch.object(torch.cuda, 'is_available', return_value=True), \
                 patch.object(runner, '_make_reference'), patch.object(runner, 'comparison'), \
                 patch.object(runner, '_train_job', side_effect=fails_first):
                with self.assertRaisesRegex(RuntimeError, 'incomplete jobs'):
                    runner.run_augreg_study(requested, root/'partial', examples=0)
            completion = json.loads((root/'partial/completion.json').read_text())
            self.assertEqual(completion['status'], 'failed')
            self.assertIn('cnn_frozen_head', completion['jobs'])
            self.assertNotIn('new_frozen_concat', completion['jobs'])
            self.assertFalse((root/'partial/selected_vit.json').exists())


if __name__ == '__main__':
    unittest.main()
