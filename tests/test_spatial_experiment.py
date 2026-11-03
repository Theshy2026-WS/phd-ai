import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import spatial_experiment as runner
from test_baseline_diagnostics import config as original_config, fixture_datasets, toy_model, write_parent


def configuration(epochs=3):
    config = original_config()
    config['dataset'].update(batch_size=2, num_workers=0)
    config['training'].update(epochs=epochs, mixed_precision=False, probe_per_class=1)
    config['training']['scheduler']['warmup_epochs'] = 0
    config['profiling'] = dict(warmup_iterations=0, benchmark_iterations=1)
    return config


def model_factory(variant, seed):
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return toy_model()


class SpatialRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def plan(self, config=None):
        config = configuration() if config is None else config
        return dict(protocol=runner.PROTOCOL, config=config, split_hash='fixture-split',
                    probe_every=2, examples=0, source_hashes=runner.source_hashes(),
                    execution_environment=runner.execution_environment(config, 42))

    def supplied(self, root):
        datasets, manifest, fingerprint = fixture_datasets(root)
        base = datasets['train']
        datasets['train_probe'] = runner.UtilityDataset(base.folder, [0, 4, 8], 'train_probe', 42)
        return datasets, manifest, fingerprint

    def test_plan_is_fresh_fixed_budget_and_preflight_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parent = root/'parent'
            write_parent(parent)
            supplied = fixture_datasets(root)
            args = runner.arguments().parse_args(['--base-run', str(parent), '--output', str(root/'new')])
            with patch('baseline_diagnostics.make_datasets', return_value=supplied):
                plan = runner.make_plan(args)
            self.assertEqual(plan['config']['training']['epochs'], 100)
            self.assertEqual(plan['config']['training']['probe_per_class'], 20)
            self.assertEqual(plan['probe_every'], 5)
            self.assertEqual(plan['config']['training']['early_stopping_patience'], 0)
            self.assertFalse(plan['config']['dataset']['download'])
            self.assertTrue(plan['fresh_training'])
            self.assertEqual([j['variant'] for j in plan['jobs']], ['cnn4', 'cnn8'])
            self.assertFalse((root/'new').exists())
            args.smoke = True
            with patch('baseline_diagnostics.make_datasets', return_value=supplied):
                self.assertEqual(runner.make_plan(args)['config']['training']['epochs'], 1)
            args.output = str(parent/'nested')
            with self.assertRaisesRegex(ValueError, 'outside'):
                runner.make_plan(args)

    def test_probe_preserves_training_rng(self):
        config = configuration()
        runner.runtime(config, 42)
        model = toy_model()
        before = runner.rng_state('cpu')
        expected = (random.random(), np.random.rand(), torch.rand(3))
        runner.restore_rng(before, 'cpu')
        def noisy(*args):
            random.random()
            np.random.rand()
            torch.rand(200)
            return dict(accuracy=.9, nll=.2, top5_accuracy=1.)
        with patch.object(runner, 'evaluate', side_effect=noisy), patch.object(runner, 'loader', return_value=[]):
            runner._probe(model, [], config, 42, 'cpu', torch.float16)
        observed = (random.random(), np.random.rand(), torch.rand(3))
        self.assertEqual(expected[:2], observed[:2])
        torch.testing.assert_close(expected[2], observed[2], rtol=0, atol=0)

    def run_fake_training(self, root, interrupted=False, interrupt_epoch=2):
        config = configuration()
        plan = self.plan(config)
        supplied = self.supplied(root)
        run = root/'run'
        job = dict(variant='cnn4', seed=42)
        state = dict(epoch=0, interrupt=interrupted)
        probe_epochs = []
        def train(model, batches, config, seed, epoch, device, amp, dtype, optimizer, scaler):
            if epoch == interrupt_epoch and state['interrupt']:
                state['interrupt'] = False
                raise KeyboardInterrupt()
            state['epoch'] = epoch
            optimizer.zero_grad()
            coefficient = random.random() + np.random.rand()
            loss = sum((parameter*torch.randn_like(parameter)*coefficient).sum()
                       for parameter in model.parameters())
            loss.backward()
            optimizer.step()
            return 1.
        def evaluate(model, batches, *args):
            if batches.split == 'train_probe':
                probe_epochs.append(state['epoch'])
                # Deliberately consume every RNG: the caller must isolate probes.
                random.random();np.random.rand();torch.rand(9)
                return dict(accuracy=1., top5_accuracy=1., nll=.01)
            return dict(accuracy={1: .8, 2: .6, 3: .7}[state['epoch']],
                        top5_accuracy=1., nll=.5)
        with patch.object(runner, 'prepare_datasets', return_value=supplied), \
                patch.object(runner, 'build_spatial_model', side_effect=model_factory), \
                patch.object(runner, 'loader', side_effect=lambda dataset, *args: dataset), \
                patch.object(runner, 'train_epoch', side_effect=train), \
                patch.object(runner, 'evaluate', side_effect=evaluate):
            if interrupted:
                with self.assertRaises(KeyboardInterrupt):
                    runner.train_run(plan, job, run)
                if interrupt_epoch > 1:
                    checkpoint = torch.load(run/'last.pt', weights_only=False)
                    self.assertEqual(checkpoint['epoch'], interrupt_epoch-1)
                else:
                    self.assertFalse((run/'last.pt').exists())
            summary = runner.train_run(plan, job, run)
        last = torch.load(run/'last.pt', weights_only=False)
        self.assertEqual(supplied[0]['audit'].reads, 0)
        self.assertEqual(supplied[0]['official_validation'].reads, 0)
        return summary, last, probe_epochs, run, plan

    def test_selection_only_checkpoint_and_probe_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            summary, last, probes, run, plan = self.run_fake_training(Path(temp))
            self.assertEqual(summary['best_epoch'], 1)
            self.assertEqual(summary['completed_epochs'], 3)
            self.assertEqual(probes, [1, 2, 3])  # Every2 plusfirst/final.
            self.assertEqual(len(last['history']), 3)
            self.assertTrue(all('clean_accuracy_gap_pp' in row for row in last['history']))
            self.assertTrue((run/'diagnostics/selection/predictions.npz').exists())
            self.assertFalse(summary['official_validation_evaluated'])
            self.assertFalse(summary['audit_partition_evaluated'])

    def test_interrupted_resume_reproduces_last_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'full').mkdir();(root/'resumed').mkdir()
            a, full, _, _, _ = self.run_fake_training(root/'full')
            b, resumed, _, _, _ = self.run_fake_training(root/'resumed', interrupted=True)
            self.assertEqual(a['best_epoch'], b['best_epoch'])
            self.assertEqual(full['history'][-1]['lr'], resumed['history'][-1]['lr'])
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)

    def test_first_epoch_restart_is_validated_and_unrelated_directory_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'first').mkdir()
            summary, _, _, _, _ = self.run_fake_training(root/'first', interrupted=True, interrupt_epoch=1)
            self.assertEqual(summary['completed_epochs'], 3)
            unrelated=root/'unrelated';unrelated.mkdir();(unrelated/'keep.txt').write_text('keep')
            config=configuration();supplied=self.supplied(root/'data')
            with patch.object(runner,'prepare_datasets',return_value=supplied), \
                    patch.object(runner,'build_spatial_model',side_effect=model_factory):
                with self.assertRaisesRegex(ValueError,'Refusing to overwrite'):
                    runner.train_run(self.plan(config),dict(variant='cnn4',seed=42),unrelated)
            self.assertEqual((unrelated/'keep.txt').read_text(),'keep')

    def test_resume_rejects_source_environment_and_split_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp)
            plan = self.plan()
            plan['jobs'] = [dict(variant='cnn4', seed=42, directory='cnn4_seed42')]
            runner.save_json(suite/'suite.json', plan)
            with patch.object(runner, 'clean_datasets'):
                self.assertEqual(runner.validate_resume(suite)['protocol'], runner.PROTOCOL)
            changed = copy.deepcopy(plan);changed['source_hashes'] = {}
            runner.save_json(suite/'suite.json', changed)
            with self.assertRaisesRegex(ValueError, 'source'):
                runner.validate_resume(suite)
            changed = copy.deepcopy(plan);changed['execution_environment']['device'] = 'differentGPU'
            runner.save_json(suite/'suite.json', changed)
            with self.assertRaisesRegex(ValueError, 'environment'):
                runner.validate_resume(suite)
            runner.save_json(suite/'suite.json', plan)
            with patch.object(runner, 'clean_datasets', side_effect=ValueError('split changed')):
                with self.assertRaisesRegex(ValueError, 'split'):
                    runner.validate_resume(suite)

    def test_paired_correction_harm_and_order_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ids=np.array(['a','b','c','d']);labels=np.array([0,1,0,1])
            shared=dict(image_ids=ids,targets=labels,class_names=np.array(['a','b']))
            a=np.array([[.8,.2],[.6,.4],[.7,.3],[.6,.4]],np.float32)
            b=np.array([[.4,.6],[.2,.8],[.7,.3],[.3,.7]],np.float32)
            np.savez(root/'a.npz',**shared,probabilities=a,predicted_labels=np.array([0,0,0,0]))
            np.savez(root/'b.npz',**shared,probabilities=b,predicted_labels=np.array([1,1,0,1]))
            score=runner.paired_selection(root/'a.npz',root/'b.npz',root/'paired.csv')
            self.assertEqual(score['corrected'],2);self.assertEqual(score['harmed'],1)
            self.assertEqual(score['delta_accuracy_pp'],25.)
            shared['image_ids']=ids[::-1]
            np.savez(root/'b.npz',**shared,probabilities=b,predicted_labels=np.array([1,1,0,1]))
            with self.assertRaisesRegex(ValueError,'identical ordered'):
                runner.paired_selection(root/'a.npz',root/'b.npz',root/'bad.csv')


if __name__ == '__main__':
    unittest.main()
