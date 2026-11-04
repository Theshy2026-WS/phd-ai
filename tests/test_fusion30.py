import copy
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import fusion30_experiment as runner
from models.attention_correction import state_digest
from models.spatial_fusion import build_fusion_model
from models.spatial_preact import build_spatial_model
from test_baseline_diagnostics import config as original_config, fixture_datasets, toy_model


class FixedTrainingImages:
    split, seed, epoch = 'train', 42, 0
    indices, folder = [10, 11], None

    def __len__(self):
        return 2

    def __getitem__(self, position):
        mean = torch.tensor([.485, .456, .406])[:, None, None]
        std = torch.tensor([.229, .224, .225])[:, None, None]
        return (torch.full((3, 64, 64), .5)-mean)/std, position, f'image{position}'


class FusionModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initial_logits_encoder_and_noisy_arm_weights_match(self):
        parent = build_spatial_model('cnn8').eval()
        rng = torch.get_rng_state().clone()
        fusion = build_fusion_model('unet32', parent.state_dict()).eval()
        noisy = build_fusion_model('unet32_noise', parent.state_dict()).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(state_digest(fusion.encoder), state_digest(parent))
        self.assertEqual(state_digest(fusion), state_digest(noisy))
        self.assertTrue(all(p.requires_grad for p in fusion.parameters()))
        sizes = []
        handles = [block.register_forward_hook(lambda m, i, o: sizes.append(tuple(o.shape[-2:])))
                   for block in (fusion.deep, fusion.refine16, fusion.refine32)]
        x = torch.randn(2, 3, 64, 64)
        with torch.inference_mode():
            expected, observed = parent(x), fusion(x)
        for handle in handles:
            handle.remove()
        self.assertTrue(torch.equal(expected, observed))
        self.assertEqual(sizes, [(8, 8), (16, 16), (32, 32)])
        self.assertEqual(sum(p.numel() for p in fusion.parameters())-sum(p.numel() for p in parent.parameters()), 681672)

    def test_zero_head_learns_then_gradients_reach_all_laterals(self):
        parent = build_spatial_model('cnn8')
        model = build_fusion_model('unet32', parent.state_dict()).train()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        x, target = torch.randn(2, 3, 64, 64), torch.tensor([0, 199])
        original_stem = model.encoder.stem.weight.detach().clone()
        nn.functional.cross_entropy(model(x), target).backward()
        self.assertGreater(float(model.correction.weight.grad.abs().sum()), 0.)
        self.assertEqual(float(model.deep.weight.grad.abs().sum()), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(original_stem, model.encoder.stem.weight))
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(x), target).backward()
        for layer in (model.deep, model.lateral16, model.lateral32):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0.)

    def test_noise_is_training_only_deterministic_rgb_space_and_rng_isolated(self):
        base = FixedTrainingImages()
        noisy = runner.NoisyTrainingDataset(base, probability=1., std_rgb=.02)
        before = torch.get_rng_state().clone()
        a, label, identifier = noisy[0]
        b, _, _ = noisy[0]
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        clean, _, _ = base[0]
        rgb_noise = (a-clean)*torch.tensor([.229, .224, .225])[:, None, None]
        self.assertAlmostEqual(float(rgb_noise.std()), .02, delta=.001)
        self.assertEqual((label, identifier), (0, 'image0'))
        noisy.epoch = 2
        self.assertEqual(base.epoch, 2)
        self.assertFalse(torch.equal(a, noisy[0][0]))
        disabled = runner.NoisyTrainingDataset(base, probability=0.)
        self.assertTrue(torch.equal(clean, disabled[0][0]))
        base.split = 'selection'
        with self.assertRaisesRegex(ValueError, 'training'):
            runner.NoisyTrainingDataset(base)


def configuration():
    config = original_config()
    config['dataset'].update(batch_size=2, num_workers=0)
    config['training']['mixed_precision'] = False
    return runner.training_config(config, None, 3)


def fake_factory(variant, parent_state, seed):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return toy_model()


class FusionRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def plan(self):
        config = configuration()
        return dict(config=config, seed=42, split_hash='fixture-split',
            parent_run='original-cnn8', parent_best_epoch=100, parent_checkpoint_sha256='parent-hash',
            source_hashes=runner.source_hashes(), protocol=runner.PROTOCOL,
            execution_environment=runner.execution_environment(config, 42), probe_every=2, examples=0)

    def supplied(self, root):
        supplied, manifest, fingerprint = fixture_datasets(root)
        base = supplied['train']
        from datasets.utility_tiny import UtilityDataset
        supplied['train_probe'] = UtilityDataset(base.folder, [0, 4, 8], 'train_probe', 42)
        return supplied, manifest, fingerprint

    def run_job(self, root, interrupted=False, improve=True, best_write_race=False, augmentation_offset=0):
        root.mkdir(parents=True, exist_ok=True)
        supplied = self.supplied(root)
        plan, state = self.plan(), dict(epoch=0, interrupt=interrupted, race=best_write_race)
        plan['augmentation_epoch_offset'] = augmentation_offset
        run, probes = root/'run', []

        def train(model, batches, config, seed, epoch, device, amp, dtype, optimizer, scaler):
            epoch -= augmentation_offset
            if epoch == 2 and state['interrupt']:
                state['interrupt'] = False
                raise KeyboardInterrupt()
            state['epoch'] = epoch
            optimizer.zero_grad()
            coefficient = random.random()+np.random.rand()
            loss = sum((p*torch.randn_like(p)*coefficient).sum() for p in model.parameters())
            loss.backward()
            optimizer.step()
            return 1.

        def evaluate(model, batches, *args):
            if batches.split == 'train_probe':
                probes.append(state['epoch'])
                random.random(); np.random.rand(); torch.rand(9)
                return dict(accuracy=1., top5_accuracy=1., nll=.01)
            scores = {1: .8, 2: .9 if best_write_race else .6, 3: .7} if improve else {1: .6, 2: .5, 3: .4}
            return dict(accuracy=scores[state['epoch']], top5_accuracy=1., nll=.5)

        original_save = runner.atomic_save

        def save(path, value):
            if Path(path).name == 'last.pt' and value['epoch'] == 2 and state['race']:
                state['race'] = False
                raise KeyboardInterrupt()
            return original_save(path, value)

        with patch.object(runner, 'prepare_datasets', return_value=supplied), \
                patch.object(runner, 'build_fusion_model', side_effect=fake_factory), \
                patch.object(runner, 'loader', side_effect=lambda ds, *args: ds), \
                patch.object(runner, 'train_epoch', side_effect=train), \
                patch.object(runner, 'evaluate', side_effect=evaluate), \
                patch('spatial_experiment.evaluate', side_effect=evaluate), \
                patch('spatial_experiment.loader', side_effect=lambda ds, *args: ds), \
                patch.object(runner, 'estimate_flops', return_value=1000), \
                patch.object(runner, 'atomic_save', side_effect=save):
            if interrupted or best_write_race:
                with self.assertRaises(KeyboardInterrupt):
                    runner._train_job(plan, 'cnn8_control', run, {})
            summary = runner._train_job(plan, 'cnn8_control', run, {})
        last = torch.load(run/'last.pt', weights_only=False)
        self.assertEqual(supplied[0]['audit'].reads, 0)
        self.assertEqual(supplied[0]['official_validation'].reads, 0)
        return summary, last, probes, run, plan

    def test_config_budget_preserves_parent_augmentation(self):
        original = original_config()
        unchanged = copy.deepcopy(original)
        config = runner.training_config(original, None, 30)
        self.assertEqual(original, unchanged)
        self.assertEqual(config['augmentation'], original['augmentation'])
        for key in ('mix_probability', 'mixup_alpha', 'cutmix_alpha', 'label_smoothing', 'weight_decay'):
            self.assertEqual(config['training'][key], original['training'][key])
        self.assertEqual(config['training']['learning_rate'], 5e-5)
        self.assertEqual(config['training']['epochs'], 30)
        self.assertEqual(config['training']['scheduler']['warmup_epochs'], 1)
        self.assertEqual(config['training']['scheduler']['min_lr'], 1e-6)

    def test_epoch_zero_can_win_and_final_epoch_is_reported(self):
        with tempfile.TemporaryDirectory() as temp:
            summary, last, probes, run, plan = self.run_job(Path(temp), improve=False)
            self.assertEqual(summary['best_epoch'], 0)
            self.assertEqual(summary['completed_epochs'], 3)
            self.assertEqual(summary['last_epoch_selection']['accuracy'], .4)
            self.assertEqual([row['epoch'] for row in last['history']], [0, 1, 2, 3])
            self.assertEqual(probes, [0, 1, 2, 3])
            self.assertTrue((run/'diagnostics/selection/predictions.npz').exists())

    def test_warm_start_uses_fresh_augmentation_epoch_stream(self):
        with tempfile.TemporaryDirectory() as temp:
            summary, last, probes, run, plan = self.run_job(Path(temp), augmentation_offset=100)
            self.assertEqual([row['epoch'] for row in last['history']], [0, 1, 2, 3])
            self.assertEqual([row['augmentation_epoch'] for row in last['history']], [100, 101, 102, 103])
            self.assertEqual(summary['augmentation_epoch_offset'], 100)
            self.assertEqual(summary['completed_epochs'], 3)

    def test_interrupted_resume_and_best_checkpoint_race_reproduce_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            a, full, _, _, _ = self.run_job(root/'full')
            b, resumed, _, _, _ = self.run_job(root/'resumed', interrupted=True)
            for name in full['model']:
                torch.testing.assert_close(full['model'][name], resumed['model'][name], rtol=0, atol=0)
            c, race, _, run, _ = self.run_job(root/'race', best_write_race=True)
            for name in full['model']:
                torch.testing.assert_close(full['model'][name], race['model'][name], rtol=0, atol=0)
            self.assertEqual(c['best_epoch'], 2)
            self.assertEqual([p.name for p in run.glob('best_epoch*.pt')], ['best_epoch2.pt'])
            self.assertEqual(a['best_epoch'], b['best_epoch'])

    def test_report_pairing_and_zoomed_curves_are_generated(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary, last, _, run, plan = self.run_job(root/'fixture')
            output = root/'analysis'; output.mkdir()
            shutil.copytree(run, output/'cnn8_control')
            shutil.copytree(run, output/'unet32')
            shutil.copytree(run, output/'unet32_noise')
            result = runner.comparison(output, plan)
            self.assertEqual(len(result['rows']), 3)
            self.assertIn('unet32_noise_vs_unet32', result['paired'])
            self.assertEqual(result['paired']['unet32_noise_vs_unet32']['corrected'], 0)
            for filename in ('learning_curves.png', 'selection_top1_zoom.png', 'selection_nll.png',
                             'comparison.csv', 'paired_summary.json', 'report.md'):
                self.assertGreater((output/filename).stat().st_size, 0)
            self.assertIn('Last selection Top-1', (output/'report.md').read_text())

    def test_resume_metadata_rejects_parent_and_recipe_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            summary, last, _, _, plan = self.run_job(Path(temp))
            runner._checkpoint_matches(last, plan, 'cnn8_control')
            bad = copy.deepcopy(plan)
            bad['parent_checkpoint_sha256'] = 'different'
            with self.assertRaisesRegex(ValueError, 'differs'):
                runner._checkpoint_matches(last, bad, 'cnn8_control')
            bad = copy.deepcopy(last)
            bad['config']['training']['learning_rate'] = 1.
            with self.assertRaisesRegex(ValueError, 'configuration'):
                runner._checkpoint_matches(bad, plan, 'cnn8_control')

    def test_pairing_counts_and_order_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shared = dict(image_ids=np.array(['a','b','c','d']), targets=np.array([0,1,0,1]),
                          class_names=np.array(['zero','one']))
            a = np.array([[.8,.2],[.6,.4],[.7,.3],[.6,.4]], np.float32)
            b = np.array([[.4,.6],[.2,.8],[.7,.3],[.3,.7]], np.float32)
            np.savez(root/'a.npz', **shared, probabilities=a, predicted_labels=a.argmax(1))
            np.savez(root/'b.npz', **shared, probabilities=b, predicted_labels=b.argmax(1))
            result = runner.paired_predictions(root/'a.npz', root/'b.npz', root/'paired.csv')
            self.assertEqual((result['corrected'], result['harmed'], result['delta_accuracy_pp']), (2, 1, 25.))
            shared['image_ids'] = shared['image_ids'][::-1]
            np.savez(root/'b.npz', **shared, probabilities=b, predicted_labels=b.argmax(1))
            with self.assertRaisesRegex(ValueError, 'identical ordered'):
                runner.paired_predictions(root/'a.npz', root/'b.npz', root/'paired.csv')

    def test_suite_attempts_remaining_independent_arms_after_failure(self):
        import mechanism_diagnostics
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root/'parent'; parent.mkdir()
            config = original_config()
            loaded = (parent.resolve(), dict(seed=42, split_hash='fixture-split', best_epoch=100, completed_epochs=100),
                      dict(config=config, model={}), 'parent-hash')
            calls = []
            def job(plan, variant, path, state):
                self.assertEqual(plan['augmentation_epoch_offset'], 100)
                calls.append(variant)
                if variant == 'cnn8_control':
                    raise RuntimeError('synthetic controlled failure')
                return dict(encoder_initial_state_sha256='same-encoder', initial_state_sha256='same-fusion')
            with patch.object(mechanism_diagnostics, 'load_spatial_parent', return_value=loaded), \
                    patch.object(runner, 'clean_datasets', return_value=(None,None,None,'fixture-split',None)), \
                    patch.object(runner, 'execution_environment', return_value={}), \
                    patch.object(runner.torch.cuda, 'is_available', return_value=True), \
                    patch.object(runner.torch.cuda, 'empty_cache'), \
                    patch.object(runner, 'file_hash', return_value='parent-hash'), \
                    patch.object(runner, '_train_job', side_effect=job), \
                    patch.object(runner, 'comparison'):
                with self.assertRaisesRegex(RuntimeError, 'One or more'):
                    runner.train_fusion_suite(parent, root/'new')
            self.assertEqual(calls, ['cnn8_control', 'unet32', 'unet32_noise'])
            completion = json.loads((root/'new/completion.json').read_text())
            self.assertEqual(completion['status'], 'failed')
            self.assertEqual(completion['runs'], 2)


if __name__ == '__main__':
    unittest.main()
